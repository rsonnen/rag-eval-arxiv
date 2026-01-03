# rag-eval-arxiv

Evaluation corpus of curated arXiv academic papers for testing RAG (Retrieval-Augmented Generation) systems.

## What This Is

This repository contains **evaluation data for RAG systems**, not a document archive. It includes:

1. **corpus.yaml** - Evaluation configuration defining domain context and testing scenarios
2. **Generated questions** - Validated Q/A pairs for evaluation (where available)
3. **metadata.json** - Paper inventory with download URLs
4. **Build tools** - Scripts for curating new paper collections

The actual PDF papers are not included. Each paper has its own license governing redistribution - use the download helper to fetch them directly from arXiv.

## Sub-Corpora

| Corpus | Category | Papers | Questions |
|--------|----------|--------|-----------|
| `computational_biology` | q-bio | ~200 | graduate_exam |
| `computer_vision` | cs.CV | ~200 | paper_review |
| `information_retrieval` | cs.IR | ~200 | - |
| `ml_interpretability` | cs.LG | ~200 | - |
| `nlp` | cs.CL | ~200 | - |
| `quantum_computing` | quant-ph | ~200 | - |
| `reinforcement_learning` | cs.LG | ~200 | - |
| `robotics` | cs.RO | ~200 | - |

## Quick Start

### 1. Download Papers

```bash
cd scripts
uv sync
uv run python download_papers.py computational_biology
```

This reads `metadata.json` and fetches all papers to the `papers/` directory.

### 2. Use for Evaluation

Each sub-corpus has a `corpus.yaml` defining evaluation scenarios:

```yaml
name: "Computational Biology Papers"
corpus_context: >
  200 arXiv papers in computational biology...

scenarios:
  graduate_exam:
    name: "Graduate Qualifying Exam"
    description: >
      Questions for a computational biology qualifying exam...
  rag_eval:
    name: "RAG System Evaluation"
    description: >
      Questions to test whether a retrieval system actually read this paper...
```

Pre-generated questions (where available) are in `{scenario}_{mode}_questions.json` files.

## Directory Structure

```
computational_biology/
    corpus.yaml                              # Evaluation configuration
    metadata.json                            # Paper list with URLs
    graduate_exam_textual_questions.json     # Generated Q/A (if available)
    papers/                                  # Downloaded PDFs (gitignored)

scripts/
    download_papers.py                       # Fetch papers from URLs in metadata
    arxiv_corpus_builder.py                  # Build new corpora (discovery + curation)

corpus_specs/
    *.yaml                                   # Build configurations for arxiv_corpus_builder
```

## Building New Corpora

The corpus builder discovers papers via arXiv API, downloads them, and curates using LLM evaluation.

### Config Format (corpus_specs/*.yaml)

```yaml
corpus_name:
  category: cs.LG                # Primary arXiv category
  secondary_categories:          # Optional
    - cs.AI
  query_terms:                   # Keywords (OR logic)
    - interpretability
    - explainability
  exclude_terms:                 # Optional filter
    - survey
  max_results: 200
  sort_by: submittedDate
  sort_order: descending
```

### Usage

```bash
cd scripts

# Build corpus (discovery + LLM curation + download)
uv run python arxiv_corpus_builder.py --config ../corpus_specs/ml_interpretability.yaml \
    --corpus ml_interpretability

# Dry run (search only, no download)
uv run python arxiv_corpus_builder.py --config ../corpus_specs/ml_interpretability.yaml \
    --corpus ml_interpretability --dry-run
```

Building is resumable - re-run to continue interrupted builds.

## Features

- **Resumable downloads**: Re-run commands to continue interrupted downloads
- **Rate limiting**: 3 second delay between requests (respects arXiv terms)
- **Dry run mode**: Search and display results without downloading
- **Metadata tracking**: JSON metadata for all downloaded papers
- **Category + keyword search**: Flexible query building from config

## Metadata Format

```json
{
  "corpus": "computational_biology",
  "source": "arxiv",
  "search_query": "cat:q-bio AND (genomics OR bioinformatics)",
  "curated_at": "2025-12-26T...",
  "total_papers": 200,
  "papers": [
    {
      "arxiv_id": "2312.12345v2",
      "title": "Paper Title",
      "authors": [{"name": "Author Name"}],
      "abstract": "...",
      "primary_category": "q-bio.GN",
      "categories": ["q-bio.GN", "cs.LG"],
      "published": "2023-12-15T...",
      "updated": "2024-01-10T...",
      "pdf_url": "https://arxiv.org/pdf/2312.12345v2",
      "file": "papers/2312.12345v2.pdf"
    }
  ]
}
```

## arXiv Categories

| Category | Field |
|----------|-------|
| cs.LG | Machine Learning |
| cs.AI | Artificial Intelligence |
| cs.CL | Computation and Language (NLP) |
| cs.CV | Computer Vision |
| cs.IR | Information Retrieval |
| cs.RO | Robotics |
| quant-ph | Quantum Physics |
| q-bio.* | Quantitative Biology |

Full taxonomy: https://arxiv.org/category_taxonomy

## API Reference

### arXiv API

- **Library**: `arxiv` (Python wrapper)
- **Rate limit**: 3 seconds between requests
- **Max results**: 2,000 per query (paginated)
- **Docs**: https://info.arxiv.org/help/api/

### Query Syntax

- `cat:cs.LG` - Category filter
- `au:Smith` - Author search
- `ti:neural` - Title search
- `abs:explanation` - Abstract search
- `AND`, `OR`, `ANDNOT` - Boolean operators

## Licensing

**This repository** (scripts, configurations): MIT License

**Papers**: Individual licenses vary (arXiv license, CC BY, CC BY-NC-SA, etc.)
- Papers are preprints, not final published versions
- For research/evaluation use only
- Check individual paper metadata for license details

## Requirements

- Python 3.11+
- uv package manager
- Dependencies in `scripts/pyproject.toml`
