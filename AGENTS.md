# AGENTS.md — Spec-Driven Development & Agent Operating Principles

> **Note**: This file mirrors [`ANTIGRAVITY.md`](ANTIGRAVITY.md) so that any AI coding agent or contributor working in `omniclaims_lakehouse_multimodal` adheres to the same Spec-Driven Development (SDD) methodology, anti-over-engineering discipline, and Life Insurance domain standards.

## 1. Project Purpose & The 3 Core Deliverables
This is an **educational POC project** demonstrating how Google Cloud Lakehouse (BigLake Managed Apache Iceberg + BigLake Object Tables + BigQuery ML / Vertex AI) unifies structured tables, medical X-rays, contract/clinical PDFs, and call center audio for **Life & Living Benefits Insurance Claims Adjudication**.

We optimize strictly for **three deliverables** and **zero over-engineering**:
1. **Simple, Reproducible Open-Source GitHub Repo** (`README.md`, `generate_data.py`, `setup_gcp.sh`, `run_lakehouse.py`, `sql/`).
2. **Interactive Multi-Modal Demo UI** (`app.py` in Streamlit) that visually renders the multi-modal insurance files and queries/displays the unified BigQuery adjudication results.
3. **High-Impact Technical Blog** (`blog/multimodal-life-insurance-lakehouse.md` + `blog/images/`), following the narrative style of `Tunix_Gemma4_TPU/blog/fine-tuning-gemma4-insurance.md`.

## 2. Quick Reference Commands
```bash
# 1. Sync Python environment with uv
uv sync

# 2. Generate the 5 Hero Life Insurance multi-modal cases locally (< 15s)
uv run python generate_data.py

# 3. Provision GCP Infrastructure (GCS Bucket, BigQuery Dataset, Cloud Resource Connection, IAM)
bash setup_gcp.sh

# 4. Upload to GCS, build Iceberg & Object Tables, and run BQML Vector Search + Gemini Multimodal SQL
uv run python run_lakehouse.py

# 5. Launch the OmniClaim Life 360 Demo UI (supports both Local Preview & Live BigQuery Querying)
uv run streamlit run app.py
```
