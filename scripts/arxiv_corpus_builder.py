#!/usr/bin/env python
"""Build curated arXiv paper corpora using LLM-evaluated filtering.

Downloads papers from arXiv one at a time, evaluates each for relevance
using an LLM-as-judge, and builds a quality corpus of on-topic papers.

Uses cursor-based iteration with persistent state for proper resume capability.

Usage:
    uv run python arxiv_corpus_builder.py \
        --config ../corpus_specs/ml_interpretability.yaml \
        --corpus ml_interpretability \
        --data-dir ../data
"""

import argparse
import calendar
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import arxiv  # type: ignore[import-untyped]  # arxiv library has no type stubs
import fitz  # type: ignore[import-untyped]  # pymupdf has no type stubs
import httpx
import yaml
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, SecretStr, ValidationError
from tqdm import tqdm

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ARXIV_DELAY_SECONDS = 3.0
PDF_DELAY_SECONDS = 1.0
SAVE_INTERVAL = 10


@dataclass
class Cursor:
    """Tracks position in arXiv search results for resume capability."""

    year: int
    month: int
    offset: int = 0

    def to_dict(self) -> dict[str, int]:
        return {"year": self.year, "month": self.month, "offset": self.offset}

    @classmethod
    def from_dict(cls, data: dict[str, int]) -> "Cursor":
        return cls(year=data["year"], month=data["month"], offset=data.get("offset", 0))


@dataclass
class BuildState:
    """Persistent state for corpus building."""

    corpus_name: str
    query: str
    cursor: Cursor
    accepted: list[dict[str, Any]] = field(default_factory=list)
    processed_ids: set[str] = field(default_factory=set)
    total_evaluated: int = 0

    def save(self, state_path: Path) -> None:
        """Save state to disk."""
        data = {
            "corpus_name": self.corpus_name,
            "query": self.query,
            "cursor": self.cursor.to_dict(),
            "accepted": self.accepted,
            "processed_ids": list(self.processed_ids),
            "total_evaluated": self.total_evaluated,
        }
        # Write atomically
        tmp_path = state_path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        tmp_path.rename(state_path)

    @classmethod
    def load(cls, state_path: Path) -> "BuildState | None":
        """Load state from disk, or return None if not found."""
        if not state_path.exists():
            return None
        try:
            with state_path.open(encoding="utf-8") as f:
                data = json.load(f)
            return cls(
                corpus_name=data["corpus_name"],
                query=data["query"],
                cursor=Cursor.from_dict(data["cursor"]),
                accepted=data.get("accepted", []),
                processed_ids=set(data.get("processed_ids", [])),
                total_evaluated=data.get("total_evaluated", 0),
            )
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"Could not load state: {e}")
            return None


class RelevanceEvaluation(BaseModel):
    """Structured output from the LLM relevance evaluation."""

    relevant: bool = Field(description="Whether the paper is primarily about the topic")
    confidence: float = Field(
        description="Confidence in the decision (0.0 to 1.0)", ge=0.0, le=1.0
    )
    reasoning: str = Field(description="Brief explanation of the decision")


def extract_pdf_text(pdf_path: Path, max_pages: int = 3) -> str:
    """Extract text from the first N pages of a PDF."""
    try:
        doc = fitz.open(pdf_path)
        text_parts: list[str] = []
        for page_num in range(min(max_pages, len(doc))):
            page = doc[page_num]
            text_parts.append(page.get_text())
        doc.close()
        return "\n\n".join(text_parts)
    except Exception as e:
        logger.warning(f"Failed to extract text from {pdf_path}: {e}")
        return ""


EVALUATION_PROMPT = """
You are evaluating whether an academic paper is relevant to a specific corpus topic.

CORPUS TOPIC DESCRIPTION:
{corpus_description}

PAPER INFORMATION:
Title: {title}
arXiv ID: {arxiv_id}

Abstract:
{abstract}

First pages content (excerpt):
{first_pages}

EVALUATION TASK:
Determine if this paper's PRIMARY FOCUS is the corpus topic described above.

IMPORTANT CRITERIA:
- The paper must be PRIMARILY ABOUT the topic, not just mention it in passing
- Papers that use the topic as a tool for something else are NOT relevant
- Papers that mention the topic only in related work or future work are NOT relevant
- Be strict: when in doubt, mark as NOT relevant

Respond with:
- relevant: true/false
- confidence: your confidence in this decision (0.0 to 1.0)
- reasoning: brief explanation (1-2 sentences)"""


def create_evaluator(
    model_name: str = "gpt-5-mini",
    temperature: float = 0.0,
) -> ChatOpenAI:
    """Create an LLM instance for paper evaluation."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")

    base_url = os.environ.get("OPENAI_BASE_URL")

    return ChatOpenAI(
        model=model_name,
        temperature=temperature,
        api_key=SecretStr(api_key),
        base_url=base_url,
    )


def evaluate_paper(
    pdf_path: Path,
    title: str,
    abstract: str,
    arxiv_id: str,
    corpus_description: str,
    llm: ChatOpenAI,
    confidence_threshold: float = 0.7,
) -> RelevanceEvaluation | None:
    """Evaluate whether a paper is relevant to the corpus topic.

    Returns None if LLM call fails.
    """
    first_pages = extract_pdf_text(pdf_path, max_pages=3)[:4000]

    prompt = EVALUATION_PROMPT.format(
        corpus_description=corpus_description,
        title=title,
        arxiv_id=arxiv_id,
        abstract=abstract,
        first_pages=first_pages,
    )

    try:
        structured_llm = llm.with_structured_output(RelevanceEvaluation)
        raw_result = structured_llm.invoke(prompt)

        if raw_result is None:
            logger.warning(f"LLM returned None for {arxiv_id}")
            return None

        result = cast(RelevanceEvaluation, raw_result)

        # Low confidence on positive = reject
        if result.relevant and result.confidence < confidence_threshold:
            return RelevanceEvaluation(
                relevant=False,
                confidence=result.confidence,
                reasoning=f"Below threshold ({result.confidence:.2f}). "
                f"{result.reasoning}",
            )

        return result

    except (ValidationError, Exception) as e:
        logger.warning(f"LLM evaluation failed for {arxiv_id}: {e}")
        return None


def load_corpus_config(config_path: Path, corpus_name: str) -> dict[str, Any]:
    """Load corpus configuration from YAML file."""
    with config_path.open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if corpus_name not in config:
        available = ", ".join(config.keys())
        raise ValueError(f"Corpus '{corpus_name}' not found. Available: {available}")

    corpus = config[corpus_name]

    required = ["category", "description", "target_count"]
    for req_field in required:
        if req_field not in corpus:
            raise ValueError(f"Corpus config missing required field: {req_field}")

    return cast(dict[str, Any], corpus)


def build_search_query(corpus_config: dict[str, Any]) -> str:
    """Build arXiv search query string from corpus config."""
    category = corpus_config["category"]
    query_parts: list[str] = [f"cat:{category}"]

    secondary_categories = corpus_config.get("secondary_categories", [])
    if secondary_categories:
        cat_clause = " OR ".join(f"cat:{cat}" for cat in secondary_categories)
        query_parts[0] = f"({query_parts[0]} OR {cat_clause})"

    query_terms = corpus_config.get("query_terms", [])
    if query_terms:
        if len(query_terms) == 1:
            query_parts.append(query_terms[0])
        else:
            terms_clause = " OR ".join(query_terms)
            query_parts.append(f"({terms_clause})")

    query = " AND ".join(query_parts)

    exclude_terms = corpus_config.get("exclude_terms", [])
    for term in exclude_terms:
        query = f"{query} ANDNOT {term}"

    return query


def iter_arxiv_papers(
    base_query: str,
    cursor: Cursor,
    processed_ids: set[str],
    min_year: int = 2020,
) -> Iterator[tuple[dict[str, Any], Cursor]]:
    """Iterate through arXiv papers starting from cursor position.

    Yields (paper_metadata, updated_cursor) tuples. The cursor is updated
    after each paper, allowing the caller to persist state for resume.

    Uses monthly windows to handle categories with >500 papers/year.
    """
    client = arxiv.Client(
        page_size=100,
        delay_seconds=ARXIV_DELAY_SECONDS,
        num_retries=3,
    )

    year = cursor.year
    month = cursor.month
    offset = cursor.offset

    while year >= min_year:
        # Build date range for this month
        days_in_month = calendar.monthrange(year, month)[1]
        start_date = f"{year}{month:02d}01000000"
        end_date = f"{year}{month:02d}{days_in_month:02d}235959"
        date_filter = f"submittedDate:[{start_date} TO {end_date}]"
        month_query = f"{base_query} AND {date_filter}"

        logger.info(f"Searching arXiv: {year}-{month:02d} (offset {offset})")

        search = arxiv.Search(
            query=month_query,
            max_results=500,  # arXiv limit per query
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        month_count = 0
        skipped_count = 0

        for result in client.results(search):
            arxiv_id = result.entry_id.split("/abs/")[-1]

            # Skip papers we've already processed
            if arxiv_id in processed_ids:
                skipped_count += 1
                offset += 1
                continue

            paper = {
                "arxiv_id": arxiv_id,
                "title": result.title,
                "authors": [{"name": a.name} for a in result.authors],
                "abstract": result.summary,
                "primary_category": result.primary_category,
                "categories": list(result.categories),
                "published": result.published.isoformat() if result.published else None,
                "updated": result.updated.isoformat() if result.updated else None,
                "pdf_url": result.pdf_url,
            }

            offset += 1
            month_count += 1

            # Update cursor to current position
            new_cursor = Cursor(year=year, month=month, offset=offset)
            yield paper, new_cursor

        if skipped_count > 0:
            logger.info(f"  Skipped {skipped_count} already-processed papers")
        logger.info(f"  Found {month_count} new papers in {year}-{month:02d}")

        # Move to previous month
        month -= 1
        if month < 1:
            month = 12
            year -= 1
        offset = 0

    logger.info("Reached minimum year, no more papers to fetch")


def download_pdf(pdf_url: str, dest_path: Path) -> bool:
    """Download a PDF file. Returns True on success."""
    time.sleep(PDF_DELAY_SECONDS)
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.get(pdf_url, follow_redirects=True)
            response.raise_for_status()
            dest_path.write_bytes(response.content)
        return True
    except httpx.HTTPError as e:
        logger.warning(f"Download failed: {e}")
        return False


def write_final_metadata(
    corpus_dir: Path,
    corpus_name: str,
    query: str,
    accepted: list[dict[str, Any]],
    total_evaluated: int,
) -> None:
    """Write final corpus metadata file."""
    metadata = {
        "corpus": corpus_name,
        "source": "arxiv",
        "search_query": query,
        "curated_at": datetime.now(UTC).isoformat(),
        "total_papers": len(accepted),
        "papers_evaluated": total_evaluated,
        "acceptance_rate": len(accepted) / total_evaluated if total_evaluated else 0,
        "papers": accepted,
    }

    with (corpus_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def build_corpus(
    config_path: Path,
    corpus_name: str,
    data_dir: Path,
    limit: int | None = None,
    fresh: bool = False,
) -> None:
    """Build a curated corpus by downloading and evaluating papers.

    Uses cursor-based iteration with persistent state for proper resume.
    """
    corpus_config = load_corpus_config(config_path, corpus_name)

    corpus_dir = data_dir / corpus_name
    papers_dir = corpus_dir / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)

    state_path = corpus_dir / "build_state.json"
    temp_dir = Path(tempfile.mkdtemp(prefix="arxiv_build_"))

    target_count = limit if limit is not None else corpus_config["target_count"]
    corpus_description = corpus_config["description"]
    confidence_threshold = corpus_config.get("confidence_threshold", 0.7)
    evaluator_model = corpus_config.get("evaluator_model", "gpt-5-mini")

    query = build_search_query(corpus_config)

    # Load or create state
    state: BuildState | None = None
    if not fresh:
        state = BuildState.load(state_path)
        if state and state.query != query:
            logger.warning("Query changed, starting fresh")
            state = None

    if state is None:
        # Start from current month
        now = datetime.now(UTC)
        state = BuildState(
            corpus_name=corpus_name,
            query=query,
            cursor=Cursor(year=now.year, month=now.month, offset=0),
        )

    if len(state.accepted) >= target_count:
        logger.info(f"Target already reached: {len(state.accepted)}/{target_count}")
        return

    logger.info(f"Building corpus: {corpus_name}")
    logger.info(f"Target: {target_count} papers (have {len(state.accepted)})")
    logger.info(f"Resuming from: {state.cursor.year}-{state.cursor.month:02d}")

    llm = create_evaluator(model_name=evaluator_model)
    papers_since_save = 0

    try:
        paper_iter = iter_arxiv_papers(
            base_query=query,
            cursor=state.cursor,
            processed_ids=state.processed_ids,
            min_year=2020,
        )

        pbar = tqdm(desc="Evaluating", unit="paper")

        for paper, new_cursor in paper_iter:
            if len(state.accepted) >= target_count:
                break

            arxiv_id = paper["arxiv_id"]
            pbar.set_postfix(
                accepted=len(state.accepted),
                evaluated=state.total_evaluated,
            )

            pdf_url = paper.get("pdf_url")
            if not pdf_url:
                logger.warning(f"No PDF URL for {arxiv_id}, skipping")
                state.processed_ids.add(arxiv_id)
                state.cursor = new_cursor
                continue

            logger.info(f"Downloading {arxiv_id}: {paper['title'][:60]}...")

            temp_pdf = temp_dir / f"{arxiv_id.replace('/', '_')}.pdf"
            if not download_pdf(pdf_url, temp_pdf):
                state.processed_ids.add(arxiv_id)
                state.cursor = new_cursor
                continue

            evaluation = evaluate_paper(
                pdf_path=temp_pdf,
                title=paper["title"],
                abstract=paper["abstract"],
                arxiv_id=arxiv_id,
                corpus_description=corpus_description,
                llm=llm,
                confidence_threshold=confidence_threshold,
            )

            # Handle evaluation failure
            if evaluation is None:
                temp_pdf.unlink(missing_ok=True)
                state.processed_ids.add(arxiv_id)
                state.cursor = new_cursor
                continue

            state.processed_ids.add(arxiv_id)
            state.cursor = new_cursor
            state.total_evaluated += 1
            papers_since_save += 1
            pbar.update(1)

            if evaluation.relevant:
                filename = f"{arxiv_id.replace('/', '_')}.pdf"
                final_path = papers_dir / filename
                try:
                    shutil.move(str(temp_pdf), str(final_path))
                    paper["file"] = f"papers/{filename}"
                    state.accepted.append(paper)
                    logger.info(
                        f"  ✓ ACCEPT ({len(state.accepted)}/{target_count}) "
                        f"[{evaluation.confidence:.2f}]: {evaluation.reasoning[:60]}"
                    )
                except OSError as e:
                    logger.error(f"Failed to move PDF: {e}")
                    temp_pdf.unlink(missing_ok=True)
            else:
                temp_pdf.unlink(missing_ok=True)
                logger.info(
                    f"  ✗ REJECT [{evaluation.confidence:.2f}]: "
                    f"{evaluation.reasoning[:60]}"
                )

            # Incremental save
            if papers_since_save >= SAVE_INTERVAL:
                state.save(state_path)
                papers_since_save = 0

        pbar.close()

        # Final save
        state.save(state_path)

        # Write final metadata
        write_final_metadata(
            corpus_dir=corpus_dir,
            corpus_name=corpus_name,
            query=query,
            accepted=state.accepted,
            total_evaluated=state.total_evaluated,
        )

        rate = (
            len(state.accepted) / state.total_evaluated if state.total_evaluated else 0
        )

        logger.info("=" * 60)
        logger.info(f"Build complete: {len(state.accepted)}/{target_count} papers")
        logger.info(f"Evaluated: {state.total_evaluated}, Acceptance rate: {rate:.1%}")
        logger.info(f"Output: {corpus_dir}")
        logger.info("=" * 60)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build curated arXiv paper corpus with LLM evaluation",
    )
    parser.add_argument(
        "--config", type=Path, required=True, help="Path to corpus config YAML"
    )
    parser.add_argument(
        "--corpus", type=str, required=True, help="Name of corpus to build"
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None, help="Data directory path"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Override target count (for testing)"
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing progress and start fresh",
    )

    args = parser.parse_args()

    if not args.config.exists():
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)

    if not os.environ.get("OPENAI_API_KEY"):
        logger.error("OPENAI_API_KEY environment variable not set")
        sys.exit(1)

    data_dir = args.data_dir or (Path(__file__).parent.parent / "data")
    data_dir.mkdir(parents=True, exist_ok=True)

    try:
        build_corpus(
            config_path=args.config,
            corpus_name=args.corpus,
            data_dir=data_dir,
            limit=args.limit,
            fresh=args.fresh,
        )
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)
    except KeyboardInterrupt:
        logger.warning("\nInterrupted")
        sys.exit(130)


if __name__ == "__main__":
    main()
