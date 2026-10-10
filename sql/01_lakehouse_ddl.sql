-- ============================================================================
-- OmniClaim Life Lakehouse — DDL (`sql/01_lakehouse_ddl.sql`)
-- 1. BigLake Managed Apache Iceberg Table (Structured Life Policy & Claim Core)
-- 2. BigLake Object Tables over Google Cloud Storage (X-Rays, PDFs, Audio)
-- 3. BigQuery ML Remote Models (Gemini Embedding 2 & Gemini 3.8 Flash)
-- ============================================================================

-- 1A. Structured Core: BigQuery Managed Apache Iceberg Table
CREATE TABLE IF NOT EXISTS `__PROJECT_ID__.__DATASET_ID__.iceberg_claims_policies` (
    claim_id STRING NOT NULL,
    policy_id STRING NOT NULL,
    policyholder_id STRING NOT NULL,
    policy_code STRING NOT NULL,
    product_type STRING,
    insured_name STRING,
    dob DATE,
    gender STRING,
    uw_health_class STRING,
    declared_smoker BOOL,
    declared_medical_history STRING,
    issue_date DATE,
    event_date DATE,
    months_since_issue INT64,
    is_within_contestability BOOL,
    claim_type STRING,
    base_sum_assured FLOAT64,
    has_add_rider BOOL,
    add_rider_sum_assured FLOAT64,
    has_ci_rider BOOL,
    ci_rider_sum_assured FLOAT64,
    claimed_benefit_amount FLOAT64,
    beneficiary_name STRING,
    beneficiary_relationship STRING,
    reported_cause_of_event STRING,
    image_gcs_uri STRING,
    aps_pdf_gcs_uri STRING,
    policy_pdf_gcs_uri STRING,
    fnol_audio_gcs_uri STRING,
    expected_action STRING,
    expected_payout FLOAT64
)
CLUSTER BY claim_id, policy_id
WITH CONNECTION `__PROJECT_ID__.__REGION__.__CONNECTION_ID__`
OPTIONS (
    file_format = 'PARQUET',
    table_format = 'ICEBERG',
    storage_uri = 'gs://__BUCKET_NAME__/iceberg_warehouse/iceberg_claims_policies'
);

-- 1B. Reference Medical Atlas Metadata Table
CREATE OR REPLACE TABLE `__PROJECT_ID__.__DATASET_ID__.dim_ref_medical_atlas_meta` (
    ref_image_id STRING NOT NULL,
    image_gcs_uri STRING NOT NULL,
    reference_category STRING,
    pathology_label STRING,
    prior_claim_id STRING,
    prior_policyholder_id STRING,
    clinical_notes STRING
);

-- ============================================================================
-- 2. BIGLAKE OBJECT TABLES (Zero-Copy Unstructured Media Catalog on GCS)
-- ============================================================================

-- 2A. Reference Medical Atlas & Prior Claims Radiographs (PNG)
CREATE OR REPLACE EXTERNAL TABLE `__PROJECT_ID__.__DATASET_ID__.obj_ref_medical_atlas`
WITH CONNECTION `__PROJECT_ID__.__REGION__.__CONNECTION_ID__`
OPTIONS (
    object_metadata = 'SIMPLE',
    uris = ['gs://__BUCKET_NAME__/unstructured_evidence/ref_medical_atlas/*.png']
);

-- 2B. Submitted Claim Radiographs (PNG)
CREATE OR REPLACE EXTERNAL TABLE `__PROJECT_ID__.__DATASET_ID__.obj_claim_radiographs`
WITH CONNECTION `__PROJECT_ID__.__REGION__.__CONNECTION_ID__`
OPTIONS (
    object_metadata = 'SIMPLE',
    uris = ['gs://__BUCKET_NAME__/unstructured_evidence/claim_radiographs/*.png']
);

-- 2C. Master Policy Contracts & Attending Physician Statements (PDF)
CREATE OR REPLACE EXTERNAL TABLE `__PROJECT_ID__.__DATASET_ID__.obj_policy_and_aps_pdfs`
WITH CONNECTION `__PROJECT_ID__.__REGION__.__CONNECTION_ID__`
OPTIONS (
    object_metadata = 'SIMPLE',
    uris = ['gs://__BUCKET_NAME__/unstructured_evidence/policy_and_aps_pdfs/*.pdf']
);

-- 2D. Beneficiary First Notice of Loss (FNOL) Recorded Statements (WAV Audio)
CREATE OR REPLACE EXTERNAL TABLE `__PROJECT_ID__.__DATASET_ID__.obj_beneficiary_fnol_audio`
WITH CONNECTION `__PROJECT_ID__.__REGION__.__CONNECTION_ID__`
OPTIONS (
    object_metadata = 'SIMPLE',
    uris = ['gs://__BUCKET_NAME__/unstructured_evidence/beneficiary_fnol_audio/*.wav']
);

-- ============================================================================
-- 3. VERTEX AI REMOTE MODELS IN BIGQUERY ML
-- ============================================================================

-- 3A. Multimodal Embedding Model (Gemini Embedding 2 / Multimodal Embedding)
CREATE OR REPLACE MODEL `__PROJECT_ID__.__DATASET_ID__.model_multimodal_embed`
REMOTE WITH CONNECTION `__PROJECT_ID__.__REGION__.__CONNECTION_ID__`
OPTIONS (ENDPOINT = '__EMBEDDING_MODEL_ENDPOINT__');

-- 3B. Multimodal Reasoning & Extraction Model (Gemini 3.8 Flash)
CREATE OR REPLACE MODEL `__PROJECT_ID__.__DATASET_ID__.model_gemini_auditor`
REMOTE WITH CONNECTION `__PROJECT_ID__.__REGION__.__CONNECTION_ID__`
OPTIONS (ENDPOINT = '__GEMINI_MODEL_ENDPOINT__');
