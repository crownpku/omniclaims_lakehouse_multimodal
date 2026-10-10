# ANTIGRAVITY.md — Project Principles & Agent Guardrails

## Mode Defaults
- **Planning / Spec-Driven Mode**: Recommended for multi-file workflow setup, BigQuery Iceberg & BigLake Object Table SQL verification, and end-to-end cloud execution.
- **Fast Mode**: Suitable for single-file edits, UI tweaks, or blog polishing.

---

## Core Project Goal & The 3 Deliverables
This project is an **educational Proof-of-Concept (POC)** showcasing Google Cloud’s newest 2026 multi-modal lakehouse capabilities in a rich **Life & Living Benefits Insurance** context.
Every design decision must serve our **three concrete deliverables** while **strictly avoiding over-engineering**:

1. **A Clean, Simple Open-Source GitHub Repository**:
   - Flat, minimal file structure (mirroring `Tunix_Gemma4_TPU`) that any developer can read and understand in 5 minutes.
   - Only **3 core Python scripts** (`generate_data.py`, `run_lakehouse.py`, `app.py`), **1 setup shell script** (`setup_gcp.sh`), and **2 readable SQL files** (`sql/01_lakehouse_ddl.sql`, `sql/02_multimodal_query.sql`).
2. **An Interactive Demo System / UI (`app.py`)**:
   - A single-file Streamlit workbench that visually showcases the multi-modal life insurance assets (Structured Iceberg Policy/Claim rows, Chest/Trauma X-Ray PNGs, Policy & Attending Physician Statement PDFs, and Beneficiary Call Audio WAVs) and displays the live/cached BigQuery cross-modal query & Gemini 3.8 Flash adjudication results.
3. **A Practitioner Technical Blog (`blog/multimodal-life-insurance-lakehouse.md` + `blog/images/`)**:
   - Written in the same candid, high-signal engineering style as `Tunix_Gemma4_TPU/blog/fine-tuning-gemma4-insurance.md`—walking through the Life Insurance unstructured data divide, BigLake Iceberg + Object Tables architecture, real SQL code snippets, technical gotchas we solved, and visual demo results.

---

## Newest 2026 Google Cloud & AI Technology Stack
Always use the **latest generation (October 2026)** Google Cloud models and SQL primitives (note: Gemini 2.5 series reached retirement in Oct 2026 and legacy `multimodalembedding@001` has been superseded by Gemini Embedding 2):

- **Multimodal Reasoning Model**: **Gemini 3.8 Flash (`gemini-3.8-flash`)** (with automatic endpoint fallback check across `gemini-3.8-flash` / `gemini-3.5-flash` / `gemini-3-flash-preview` in `run_lakehouse.py`).
- **Multimodal Embedding Model**: **Gemini Embedding 2 (`gemini-embedding-2-preview`)** — 3072-dimensional native multimodal embedding model supporting Images, PDFs, Audio, Video, and Text in a unified vector space (with automatic fallback to `multimodalembedding@001` if needed).
- **Modern BigQuery AI SQL Functions**:
  - `AI.GENERATE` / `ML.GENERATE_TEXT` with native `ObjectRef` (`OBJ.MAKE_REF`, `OBJ.FETCH_METADATA`) to process PDFs and `.wav` Audio directly from GCS.
  - `AI.GENERATE_EMBEDDING` / `AI.EMBED` + `VECTOR_SEARCH` (`COSINE` distance).
  - `OBJ.GET_ACCESS_URL(ref, 'r')` for zero-egress signed URLs in the UI.
- **Open Lakehouse Storage**:
  - **BigLake Managed Apache Iceberg Tables** (`table_format = 'ICEBERG'`, `file_format = 'PARQUET'`, `storage_uri = 'gs://...'`).
  - **BigLake Object Tables** (`object_metadata = 'SIMPLE'`).
- **Speech Synthesis**: **Google Cloud Text-to-Speech Chirp 3: HD Voices (`en-US-Chirp3-HD-*`)** for realistic beneficiary FNOL recorded statements.

---

## Anti-Over-Engineering Guardrails (Keep It Simple!)
- **No Bloated Datasets**: Generate **only the 5 curated Hero Life Insurance Claims** (`CLM-LIFE-1001` to `CLM-CI-1005`) and **6 Reference Medical Atlas Images**. Do *not* generate 50+ noisy background rows or 5,000 padded vectors. Five vivid cases run in seconds, cost pennies on Vertex AI, and are 10x easier to explain in a blog and demo UI.
- **No Deep Package Nesting or Boilerplate**: Keep scripts at the root directory (`generate_data.py`, `run_lakehouse.py`, `app.py`) rather than deeply nested enterprise modules, abstract factory classes, or Docker/Terraform sprawl.
- **Readable, Blog-Ready SQL**: Keep SQL queries self-contained and well-commented in `sql/01_lakehouse_ddl.sql` and `sql/02_multimodal_query.sql` so snippets can be pasted directly into the technical blog.

---

## Domain Context (Strict Life Insurance Focus)
- **Products**: Term Life (20-Year), Whole Life, Universal Life with Living Benefits, Accidental Death & Dismemberment (**AD&D**) Double-Indemnity Riders, and Critical Illness (**CI**) Accelerated Benefit Riders.
- **Core Life Adjudication Scenarios (The 5 Hero Cases)**:
  1. `CLM-LIFE-1001` (**Clean Fast-Track**): 20-Yr Term Life ($500K), 68 months in force (past the 24-month Contestability Window), consistent acute pneumonia X-ray, APS PDF, and beneficiary call audio $\rightarrow$ `AUTO_APPROVE_PAYOUT`.
  2. `CLM-LIFE-1002` (**Contestable Material Misrepresentation**): Whole Life ($1.5M), **7 months in force** (inside the 24-month Contestability Window). Applicant declared *"Preferred Plus, zero cardiac history"*, but X-ray vector search matches **Severe Cardiomegaly + Pacemaker**, APS PDF confirms a 4-year cardiomyopathy history, and spouse mentions his *"cardiologist of 4 years"* on the FNOL audio $\rightarrow$ `SIU_RESCISSION_HOLD`.
  3. `CLM-ADD-1003` (**AD&D Rider Exclusion**): Executive Term ($1M Base + $1M AD&D Rider). Coroner APS PDF and beneficiary audio reveal unroped free-solo ice climbing above 4,000m, triggering Section 4.2 AD&D Mountaineering Exclusion in the Policy PDF $\rightarrow$ `PARTIAL_APPROVE_BASE_DENY_ADD_RIDER` (Pay $1M Base Life; Deny $1M AD&D Rider).
  4. `CLM-CI-1004` (**Syndicated Radiograph Reuse Fraud**): Universal Life + CI Rider ($350K). `VECTOR_SEARCH` flags a near-zero cosine distance (`< 0.02`) against a radiograph submitted 9 months ago under a different policy (`PRIOR-CLM-8821`) $\rightarrow$ `SIU_FRAUD_RING_REJECT`.
  5. `CLM-CI-1005` (**Valid Critical Illness Living Benefit**): Universal Life + CI Rider ($250K), 29 months in force, biopsy & pulmonary radiograph meet Section 2.1 CI severity criteria $\rightarrow$ `APPROVE_LIVING_BENEFIT`.

---

## Developer Guidelines & Agent Rules

1. **Environment & Package Management**:
   - Always use `uv` (`uv sync`, `uv run`) to manage Python dependencies.
   - Always check official Google Cloud documentation before writing BigQuery / BigLake / BQML code.
   - Never modify installed third-party packages inside `.venv/`.
2. **Local-First Workflow**:
   - Author and test all code locally in the repository root first.
   - Ensure `uv run python generate_data.py` and `uv run streamlit run app.py` work out-of-the-box locally even before running the cloud pipeline (using local `data/` preview + cached/live BigQuery toggle).
