#!/usr/bin/env python3
"""Download papers from a curated corpus.

Reads metadata.json and downloads all PDFs listed.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import httpx


def download_corpus(corpus_dir: Path, delay: float = 1.0) -> None:
    """Download all papers in a corpus."""
    metadata_path = corpus_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: {metadata_path} not found", file=sys.stderr)
        sys.exit(1)

    with open(metadata_path) as f:
        metadata = json.load(f)

    papers_dir = corpus_dir / "papers"
    papers_dir.mkdir(exist_ok=True)

    papers = metadata.get("papers", [])
    print(f"Downloading {len(papers)} papers to {papers_dir}")

    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        for i, paper in enumerate(papers, 1):
            arxiv_id = paper["arxiv_id"]
            pdf_url = paper["pdf_url"]
            pdf_path = papers_dir / f"{arxiv_id}.pdf"

            if pdf_path.exists():
                print(f"[{i}/{len(papers)}] {arxiv_id} - already exists")
                continue

            print(f"[{i}/{len(papers)}] {arxiv_id} - downloading...")
            try:
                response = client.get(pdf_url)
                response.raise_for_status()
                pdf_path.write_bytes(response.content)
            except httpx.HTTPError as e:
                print(f"  Error: {e}", file=sys.stderr)

            time.sleep(delay)

    print("Done")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download papers from a curated corpus")
    parser.add_argument("corpus", help="Corpus directory (e.g., ml_interpretability)")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests (seconds)")
    args = parser.parse_args()

    # Find corpus directory relative to repo root
    script_dir = Path(__file__).parent
    repo_root = script_dir.parent
    corpus_dir = repo_root / args.corpus

    if not corpus_dir.exists():
        print(f"Error: Corpus directory not found: {corpus_dir}", file=sys.stderr)
        sys.exit(1)

    download_corpus(corpus_dir, args.delay)


if __name__ == "__main__":
    main()
