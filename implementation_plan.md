# OmniClaim Life Lakehouse — Streamlined Spec-Driven Implementation Plan (`implementation_plan.md`)

This document is the step-by-step **Spec-Driven Development (SDD) Plan** for building **OmniClaim Life Lakehouse** on Google Cloud (`GCP_PROJECT_ID=your-gcp-project-id`). Following [`ANTIGRAVITY.md`](ANTIGRAVITY.md), this plan uses the **newest October 2026 Google Cloud AI & Lakehouse stack** and is intentionally **simple, flat, educational, and free of over-engineering**, targeting three concrete deliverables:

1. **A Simple, Reproducible Open-Source GitHub Repo** (`README.md`, `generate_data.py`, `setup_gcp.sh`, `run_lakehouse.py`, `sql/`)
2. **An Interactive Demo UI (`app.py`)** showcasing multi-modal Life Insurance content & live BigQuery cross-modal querying
3. **A Technical Blog (`blog/multimodal-life-insurance-lakehouse.md` + `blog/images/`)** explaining the architecture, SQL, and lessons learned

---

## 1. Newest 2026 Google Cloud Technology Stack

| Layer | 2026 Google Cloud Technology | Why It Matters (vs. Legacy 2024/2025 Stack) |
| :--- | :--- | :--- |
| **Structured Table Format** | **BigLake Managed Apache Iceberg Tables** (`table_format = 'ICEBERG'`, `file_format = 'PARQUET'`) | Open Parquet + Iceberg metadata stored directly in your GCS bucket (`gs://.../iceberg_warehouse/`) with BigQuery automated compaction & ACID governance. |
| **Unstructured Media Catalog** | **BigLake Object Tables** (`object_metadata = 'SIMPLE'`) + **`ObjectRef`** (`OBJ.MAKE_REF`, `OBJ.FETCH_METADATA`) | Exposes raw Chest X-Ray PNGs, Policy/APS PDFs, and Beneficiary FNOL `.wav` files in GCS as SQL rows with native `ObjectRef` pointers—zero ETL copying. |
| **Multimodal Embeddings** | **Gemini Embedding 2 (`gemini-embedding-2-preview`)** via `AI.GENERATE_EMBEDDING` / `ML.GENERATE_EMBEDDING` | Newest 2026 flagship **3072-dimensional** native multimodal embedding model across Images, PDFs, Audio, and Text (superseding legacy 1408-D `multimodalembedding@001`). |
| **Vector Similarity Search** | **BigQuery `VECTOR_SEARCH`** (`COSINE` distance) | In-engine cosine similarity comparing submitted claim radiographs against a Reference Medical & Prior-Claims Atlas without external vector DBs (Pinecone/Milvus). |
| **Multimodal Reasoning LLM** | **Gemini 3.8 Flash (`gemini-3.8-flash`)** via `AI.GENERATE` / `ML.GENERATE_TEXT` | Newest October 2026 Gemini 3 generation (replacing retiring `gemini-2.5-flash`), natively reading Policy & APS PDFs and listening to `.wav` beneficiary call recordings via `ObjectRef` inside SQL. |
| **Secure Media Delivery** | **`OBJ.GET_ACCESS_URL(ref, 'r')`** | Generates time-limited signed GCS URLs directly inside SQL so Streamlit renders private X-rays, PDFs, and audio with `--public-access-prevention` enforced. |
| **Voice Synthesis** | **Cloud Text-to-Speech Chirp 3: HD Voices** (`en-US-Chirp3-HD-*`) | Synthesizes lifelike beneficiary First Notice of Loss (FNOL) recorded statements. |

---

## 2. Streamlined Repository Layout (Zero Over-Engineering)

```text
omniclaims_lakehouse_multimodal/
├── ANTIGRAVITY.md                              # Project principles & 2026 tech stack guardrails
├── AGENTS.md                                   # Quick reference & SDD operating rules
├── omniclaim-lakehouse-project-description-and-implementation-p.md  # Product & Architecture Spec
├── implementation_plan.md                      # This streamlined engineering plan
├── pyproject.toml                              # uv dependencies (bigquery, storage, tts, fpdf2, pillow, streamlit)
├── .env.example                                # GCP_PROJECT_ID=your-gcp-project-id, models, bucket
├── README.md                                   # Clean GitHub landing page & 4-step reproduction guide
├── setup_gcp.sh                                # Step 1: Creates GCS bucket, BQ dataset, Cloud Resource Connection & IAM
├── generate_data.py                            # Step 2: Generates the 5 Hero Life Insurance cases locally into data/
├── run_lakehouse.py                            # Step 3: Uploads data/ to GCS and executes sql/01 & sql/02 in BigQuery
├── app.py                                      # Step 4: Streamlit Demo UI (shows X-rays, PDFs, Audio & runs/displays SQL)
├── sql/
│   ├── 01_lakehouse_ddl.sql                    # Managed Apache Iceberg tables + BigLake Object Tables + BQML Models
│   └── 02_multimodal_query.sql                 # Gemini Embedding 2 + VECTOR_SEARCH + Gemini 3.8 Flash ObjectRef SQL
└── blog/
    ├── multimodal-life-insurance-lakehouse.md  # Full technical blog post (Medium / Dev.to ready)
    └── images/                                 # Architecture diagrams & Demo UI screenshots for the blog
```

---

## 3. Step-by-Step Implementation Plan

### Step 1: Local Multi-Modal Life Insurance Data Generator (`generate_data.py`)
`generate_data.py` deterministically synthesizes **5 vivid, domain-authentic Life Insurance claims** (`CLM-LIFE-1001` through `CLM-CI-1005`) and **6 Reference Medical Atlas radiographs** into `data/` in `< 15 seconds`:

| Claim ID | Life Insurance Scenario | Structured Iceberg Data (`claims_policy_core.csv`) | Radiograph Image (`claim_radiographs/*.png`) | Policy & Clinical PDF (`pdfs/*.pdf`) | Beneficiary Audio (`audio/*.wav`) | Expected Lakehouse Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`CLM-LIFE-1001`** | **Clean Fast-Track Death Claim** | *Eleanor Vance*, 20-Yr Term (`$500K`), **68 mos** in force (Incontestable >24m), Preferred Non-Smoker | Matches `REF_PNEUMONIA_001` (acute right-lower-lobe pneumonia) | `POL-TERM-STD-2025.pdf` + `CLM-LIFE-1001_aps.pdf` (acute pneumonia, no prior history) | Daughter reports sudden 4-day pneumonia hospitalization | ✅ `AUTO_APPROVE_PAYOUT` (`$500,000`) |
| **`CLM-LIFE-1002`** | **24-Month Contestability Material Misrepresentation** | *Robert Sterling*, Whole Life (`$1.5M`), **7 mos** in force ⚠️ (**Contestable <24m**), declared *"Zero cardiac history"* | Matches `REF_CARDIOMEGALY_001` (severe enlarged heart + pacemaker wire) | `POL-WL-PREF-2025.pdf` + `CLM-LIFE-1002_aps.pdf` (ICD-10 I42.0 Cardiomyopathy diagnosed 4 yrs ago) | Spouse says: *"His cardiologist of 4 years rushed him to the ER..."* | 🚨 `SIU_RESCISSION_HOLD` (`$0` - Contestable Misrepresentation) |
| **`CLM-ADD-1003`** | **AD&D Double-Indemnity Hazardous Sport Exclusion** | *Marcus Thorne*, Exec Term (`$1M` Base + `$1M` AD&D Rider = `$2M`), **38 mos** in force | Matches `REF_TRAUMA_001` (high-energy thoracic/clavicular trauma) | `POL-ADD-EXEC-2025.pdf` (Sec 4.2 excludes unroped climbing >4,000m) + Coroner APS PDF | Brother admits insured was *free-solo ice climbing at 4,250m on Mt. Rainier* | ⚠️ `PARTIAL_APPROVE_BASE_DENY_ADD_RIDER` (`$1,000,000` Base only) |
| **`CLM-CI-1004`** | **Syndicated Radiograph Reuse Fraud Ring** | *Victor Krum*, UL + Critical Illness Rider (`$350K`), **14 mos** in force ⚠️ (**Contestable <24m**) | **99.5% duplicate** (1px shift) of `PRIOR_CLM_8821_FRAUD.png` submitted 9 mos ago by another policy | `POL-UL-CI-2025.pdf` + `CLM-CI-1004_aps.pdf` (unverifiable clinic license) | Evasive caller refuses to name biopsy hospital | 🛑 `SIU_FRAUD_RING_REJECT` (`$0` - Duplicate X-Ray Fraud) |
| **`CLM-CI-1005`** | **Valid Critical Illness Living Benefit** | *Diana Chen*, UL + Critical Illness Rider (`$250K`), **29 mos** in force (past waiting & contestability) | Matches `REF_PULM_MASS_001` (distinct 2.8cm RUL nodule, zero prior claim match) | `POL-UL-CI-2025.pdf` (Sec 2.1 threshold) + Oncology biopsy APS PDF | Insured confirms oncology diagnosis & treatment plan | ✅ `APPROVE_LIVING_BENEFIT` (`$250,000`) |

---

### Step 2: GCP Setup & Readable Lakehouse SQL (`setup_gcp.sh`, `sql/`, `run_lakehouse.py`)
1. **`setup_gcp.sh`**:
   - Configures project `your-gcp-project-id`, enables BigQuery, BigLake, Connection, Storage, Vertex AI, and Cloud TTS APIs, creates the private GCS bucket `gs://your-gcp-project-id-bucket`, creates dataset `omniclaim_lakehouse`, creates Cloud Resource Connection `lakehouse-omniclaim-conn`, and grants `roles/storage.objectAdmin` + `roles/aiplatform.user` to the connection service account.
2. **`sql/01_lakehouse_ddl.sql`**:
   - Creates the **BigQuery Managed Apache Iceberg Table** (`omniclaim_lakehouse.iceberg_claims_policies`) + reference atlas metadata table (`dim_ref_medical_atlas_meta`).
   - Creates the **BigLake Object Tables** (`obj_ref_medical_atlas`, `obj_claim_radiographs`, `obj_policy_and_aps_pdfs`, `obj_beneficiary_fnol_audio`).
   - Registers the **BQML Remote Models** pointing to **`gemini-embedding-2-preview`** and **`gemini-3.8-flash`**.
3. **`sql/02_multimodal_query.sql`**:
   - Generates multimodal embeddings over the radiograph Object Tables using `AI.GENERATE_EMBEDDING` / `ML.GENERATE_EMBEDDING`.
   - Runs `VECTOR_SEARCH` (`COSINE`, `use_brute_force => true`) comparing `claims_image_embeddings` against `ref_medical_atlas_embeddings`.
   - Runs **Gemini 3.8 Flash** (`AI.GENERATE` / `ML.GENERATE_TEXT`) over the **Policy PDF**, **APS PDF**, and **Beneficiary `.wav` Audio** `ObjectRef` structs (`OBJ.FETCH_METADATA(OBJ.MAKE_REF(...))`) and builds the final view `vw_claim_adjudication_360` with `OBJ.GET_ACCESS_URL` signed URLs.
4. **`run_lakehouse.py`**:
   - Uploads `data/unstructured/` to GCS, automatically verifies the newest supported Gemini 3.x & Embedding 2 model endpoints in the region, runs `sql/01_lakehouse_ddl.sql` and `sql/02_multimodal_query.sql`, caches the live BigQuery output to `data/live_bq_results.json`, and prints a clean verification table.

---

### Step 3: Interactive Multi-Modal Demo UI (`app.py`)
- Single-file Streamlit app showing:
  1. **Apache Iceberg Policy & 24-Month Contestability Banner**
  2. **3-Column Multi-Modal Evidence Viewer**:
     - **Radiograph Vector Search Forensics** (Submitted X-ray vs. Matched Atlas/Prior X-ray + Cosine Similarity)
     - **Policy Contract & Attending Physician Statement PDFs** (Inline preview / signed link + extracted clauses)
     - **Beneficiary FNOL Recorded Audio** (`st.audio` player + caller transcript & contradiction highlights)
  3. **Gemini 3.8 Flash Unified Verdict & Live BigQuery SQL Explorer**

---

### Step 4: Technical Blog & README (`blog/multimodal-life-insurance-lakehouse.md` & `README.md`)
- Complete technical blog post with real BigQuery outputs, SQL snippets, architecture diagrams, and UI screenshots.
