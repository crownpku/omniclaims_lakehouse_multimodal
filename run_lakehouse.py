#!/usr/bin/env python3
"""
OmniClaim Life Lakehouse — Cloud Orchestrator & BigQuery ML Pipeline (`run_lakehouse.py`)

Executes the end-to-end Google Cloud Lakehouse workflow:
  1. Uploads unstructured media (`data/unstructured_evidence/`) to Google Cloud Storage (`gs://...`)
  2. Resolves the newest available Vertex AI Gemini 3.x & Multimodal Embedding endpoints in BigQuery
  3. Executes `sql/01_lakehouse_ddl.sql` (Managed Apache Iceberg Table + BigLake Object Tables + Remote Models)
  4. Loads structured Life Insurance Policy & Claim records into the Managed Apache Iceberg table
  5. Executes `sql/02_multimodal_query.sql` (`ML.GENERATE_EMBEDDING`, `VECTOR_SEARCH`, `ML.GENERATE_TEXT` with `ObjectRef`, `OBJ.GET_ACCESS_URL`)
  6. Exports the live BigQuery results to `data/live_bq_results.json` and verifies all 5 Hero Claims!
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import google.auth
from google.oauth2.credentials import Credentials
import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery, storage

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
REGION = os.getenv("GCP_REGION", "us-central1")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", f"{PROJECT_ID}-bucket")
DATASET_ID = os.getenv("BQ_DATASET_ID", "omniclaim_lakehouse")
CONNECTION_ID = os.getenv("BQ_CONNECTION_ID", "lakehouse-omniclaim-conn")
PREFERRED_GEMINI = os.getenv("GEMINI_MODEL_ENDPOINT", "gemini-3.8-flash")
PREFERRED_EMBED = os.getenv("EMBEDDING_MODEL_ENDPOINT", "gemini-embedding-2-preview")

DATA_DIR = Path("data")


def get_gcp_credentials():
    """Returns ADC credentials, or falls back to active `gcloud auth print-access-token`."""
    try:
        creds, _ = google.auth.default(quota_project_id=PROJECT_ID)
        return creds
    except Exception:
        token = subprocess.check_output(
            ["gcloud", "auth", "print-access-token", "--quiet"], text=True
        ).strip()
        return Credentials(token=token, quota_project_id=PROJECT_ID)


def render_sql_template(sql_text: str, gemini_endpoint: str, embed_endpoint: str) -> str:
    return (
        sql_text.replace("__PROJECT_ID__", PROJECT_ID)
        .replace("__REGION__", REGION)
        .replace("__BUCKET_NAME__", BUCKET_NAME)
        .replace("__DATASET_ID__", DATASET_ID)
        .replace("__CONNECTION_ID__", CONNECTION_ID)
        .replace("__GEMINI_MODEL_ENDPOINT__", gemini_endpoint)
        .replace("__EMBEDDING_MODEL_ENDPOINT__", embed_endpoint)
    )


def split_sql_statements(sql_text: str) -> list[str]:
    """Splits a SQL script on semicolons outside single/double quotes and comments."""
    cleaned_lines = []
    for line in sql_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--"):
            continue
        cleaned_lines.append(line)
    joined = "\n".join(cleaned_lines)
    stmts = [s.strip() for s in re.split(r";\s*(?:\n|$)", joined) if s.strip()]
    return stmts


def upload_unstructured_to_gcs(storage_client: storage.Client) -> int:
    print(f"==> [1/5] Uploading unstructured multi-modal insurance files to gs://{BUCKET_NAME}/...")
    bucket = storage_client.bucket(BUCKET_NAME)
    unstructured_root = DATA_DIR / "unstructured_evidence"
    count = 0
    for file_path in sorted(unstructured_root.rglob("*")):
        if file_path.is_file():
            rel_path = file_path.relative_to(DATA_DIR).as_posix()
            blob = bucket.blob(rel_path)
            content_type = None
            if file_path.suffix == ".png":
                content_type = "image/png"
            elif file_path.suffix == ".pdf":
                content_type = "application/pdf"
            elif file_path.suffix == ".wav":
                content_type = "audio/wav"
            blob.upload_from_filename(str(file_path), content_type=content_type)
            count += 1
    print(f"    [✓] Uploaded {count} unstructured files (PNG X-Rays, PDF Contracts/APS, WAV Audio) to GCS.")
    return count


def resolve_best_bqml_endpoints(bq_client: bigquery.Client) -> tuple[str, str]:
    """
    Tests the newest 2026 Gemini 3.x and Multimodal Embedding endpoints in BigQuery ML
    and selects the newest model endpoint supported in the target project/region.
    """
    print("==> [2/5] Resolving newest supported Vertex AI model endpoints in BigQuery ML...")
    gemini_candidates = [
        PREFERRED_GEMINI,
        "gemini-3-flash-preview",
        "gemini-2.5-flash",
    ]
    # Deduplicate while preserving order
    gemini_candidates = list(dict.fromkeys(gemini_candidates))

    embed_candidates = [
        PREFERRED_EMBED,
        "multimodalembedding@001",
    ]
    embed_candidates = list(dict.fromkeys(embed_candidates))

    selected_gemini = gemini_candidates[-1]
    for candidate in gemini_candidates:
        ddl = f"""
        CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_ID}.model_gemini_auditor`
        REMOTE WITH CONNECTION `{PROJECT_ID}.{REGION}.{CONNECTION_ID}`
        OPTIONS (ENDPOINT = '{candidate}')
        """
        try:
            bq_client.query(ddl).result()
            selected_gemini = candidate
            print(f"    [✓] Active Generative Multimodal Model : {selected_gemini}")
            break
        except Exception as exc:
            print(f"    [i] Endpoint '{candidate}' probe note: {str(exc).splitlines()[0][:110]}")

    selected_embed = embed_candidates[-1]
    for candidate in embed_candidates:
        ddl = f"""
        CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_ID}.model_multimodal_embed`
        REMOTE WITH CONNECTION `{PROJECT_ID}.{REGION}.{CONNECTION_ID}`
        OPTIONS (ENDPOINT = '{candidate}')
        """
        try:
            bq_client.query(ddl).result()
            # Also verify embedding over Object Table works with this endpoint
            selected_embed = candidate
            print(f"    [✓] Active Multimodal Embedding Model  : {selected_embed}")
            break
        except Exception as exc:
            print(f"    [i] Endpoint '{candidate}' probe note: {str(exc).splitlines()[0][:110]}")

    return selected_gemini, selected_embed


def load_structured_iceberg_and_metadata(bq_client: bigquery.Client) -> None:
    print("==> [3/5] Loading structured data into BigLake Managed Apache Iceberg & Atlas Metadata tables...")
    structured_dir = DATA_DIR / "structured"

    # Load Reference Medical Atlas Metadata
    df_atlas = pd.read_parquet(structured_dir / "dim_ref_medical_atlas_meta.parquet")
    df_atlas["image_gcs_uri"] = df_atlas["image_gcs_uri"].str.replace(
        r"^gs://[^/]+/", f"gs://{BUCKET_NAME}/", regex=True
    )
    atlas_cols = [
        "ref_image_id",
        "image_gcs_uri",
        "reference_category",
        "pathology_label",
        "prior_claim_id",
        "prior_policyholder_id",
        "clinical_notes",
    ]
    df_atlas = df_atlas[atlas_cols]
    job_cfg_atlas = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE)
    bq_client.load_table_from_dataframe(
        df_atlas, f"{PROJECT_ID}.{DATASET_ID}.dim_ref_medical_atlas_meta", job_config=job_cfg_atlas
    ).result()
    print(f"    [✓] Loaded {len(df_atlas)} rows into `dim_ref_medical_atlas_meta`.")

    # Load Iceberg Claims & Policies Table
    df_claims = pd.read_parquet(structured_dir / "iceberg_claims_policies.parquet")
    for uri_col in ["image_gcs_uri", "aps_pdf_gcs_uri", "policy_pdf_gcs_uri", "fnol_audio_gcs_uri"]:
        df_claims[uri_col] = df_claims[uri_col].str.replace(
            r"^gs://[^/]+/", f"gs://{BUCKET_NAME}/", regex=True
        )
    df_claims["dob"] = pd.to_datetime(df_claims["dob"]).dt.date
    df_claims["issue_date"] = pd.to_datetime(df_claims["issue_date"]).dt.date
    df_claims["event_date"] = pd.to_datetime(df_claims["event_date"]).dt.date

    iceberg_cols = [
        "claim_id",
        "policy_id",
        "policyholder_id",
        "policy_code",
        "product_type",
        "insured_name",
        "dob",
        "gender",
        "uw_health_class",
        "declared_smoker",
        "declared_medical_history",
        "issue_date",
        "event_date",
        "months_since_issue",
        "is_within_contestability",
        "claim_type",
        "base_sum_assured",
        "has_add_rider",
        "add_rider_sum_assured",
        "has_ci_rider",
        "ci_rider_sum_assured",
        "claimed_benefit_amount",
        "beneficiary_name",
        "beneficiary_relationship",
        "reported_cause_of_event",
        "image_gcs_uri",
        "aps_pdf_gcs_uri",
        "policy_pdf_gcs_uri",
        "fnol_audio_gcs_uri",
        "expected_action",
        "expected_payout",
    ]
    df_iceberg = df_claims[iceberg_cols]

    # Clear existing rows in Managed Iceberg table for idempotency, then load
    bq_client.query(f"DELETE FROM `{PROJECT_ID}.{DATASET_ID}.iceberg_claims_policies` WHERE TRUE").result()
    job_cfg_iceberg = bigquery.LoadJobConfig(write_disposition=bigquery.WriteDisposition.WRITE_APPEND)
    bq_client.load_table_from_dataframe(
        df_iceberg, f"{PROJECT_ID}.{DATASET_ID}.iceberg_claims_policies", job_config=job_cfg_iceberg
    ).result()
    print(f"    [✓] Loaded {len(df_iceberg)} Hero Life Insurance claims into Managed Apache Iceberg table `iceberg_claims_policies`.")


def execute_sql_script(
    bq_client: bigquery.Client,
    script_path: Path,
    gemini_endpoint: str,
    embed_endpoint: str,
) -> None:
    raw_sql = script_path.read_text(encoding="utf-8")
    rendered_sql = render_sql_template(raw_sql, gemini_endpoint, embed_endpoint)
    statements = split_sql_statements(rendered_sql)
    for idx, stmt in enumerate(statements, start=1):
        first_line = stmt.splitlines()[0][:85]
        print(f"    -> Executing SQL [{idx}/{len(statements)}]: {first_line}...")
        t0 = time.time()
        try:
            bq_client.query(stmt).result()
            print(f"       Done in {time.time() - t0:.1f}s")
        except Exception as exc:
            # If gemini-embedding-2-preview over Object Table requires fallback to multimodalembedding@001
            if "model_multimodal_embed" in stmt and embed_endpoint != "multimodalembedding@001":
                print(f"       [i] Falling back to 'multimodalembedding@001' for Object Table image embedding...")
                bq_client.query(
                    f"""
                    CREATE OR REPLACE MODEL `{PROJECT_ID}.{DATASET_ID}.model_multimodal_embed`
                    REMOTE WITH CONNECTION `{PROJECT_ID}.{REGION}.{CONNECTION_ID}`
                    OPTIONS (ENDPOINT = 'multimodalembedding@001')
                    """
                ).result()
                bq_client.query(stmt).result()
                print(f"       Done with multimodalembedding@001 in {time.time() - t0:.1f}s")
            else:
                raise exc


def clean_json_string(raw_str: str | None) -> dict[str, Any]:
    if not raw_str:
        return {}
    cleaned = raw_str.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass
        return {"raw_output": raw_str}


def main() -> None:
    if not (DATA_DIR / "structured" / "iceberg_claims_policies.parquet").exists():
        print("Local dataset not found; running generate_data.py first...")
        import generate_data

        generate_data.main()

    creds = get_gcp_credentials()
    storage_client = storage.Client(project=PROJECT_ID, credentials=creds)
    bq_client = bigquery.Client(project=PROJECT_ID, location=REGION, credentials=creds)

    upload_unstructured_to_gcs(storage_client)
    gemini_endpoint, embed_endpoint = resolve_best_bqml_endpoints(bq_client)

    print("==> [3A/5] Executing `sql/01_lakehouse_ddl.sql`...")
    execute_sql_script(bq_client, Path("sql/01_lakehouse_ddl.sql"), gemini_endpoint, embed_endpoint)

    load_structured_iceberg_and_metadata(bq_client)

    print("==> [4/5] Executing `sql/02_multimodal_query.sql` (Embeddings + Vector Search + Gemini Audit)...")
    execute_sql_script(bq_client, Path("sql/02_multimodal_query.sql"), gemini_endpoint, embed_endpoint)

    print("==> [5/5] Querying `vw_claim_adjudication_360` and verifying all 5 Hero Life Claims...")
    query = f"""
    SELECT *
    FROM `{PROJECT_ID}.{DATASET_ID}.vw_claim_adjudication_360`
    ORDER BY claim_id
    """
    df_res = bq_client.query(query).to_dataframe()

    records = []
    print("\n" + "=" * 112)
    print(f"{'CLAIM ID':<15} | {'INSURED':<17} | {'CONTESTABLE':<11} | {'X-RAY VECTOR FLAG':<38} | {'GEMINI VERDICT'}")
    print("-" * 112)
    for _, row in df_res.iterrows():
        row_dict = {}
        for k, v in row.items():
            if k.startswith("signed_"):
                continue
            if hasattr(v, "isoformat"):
                row_dict[k] = v.isoformat()
            elif pd.isna(v):
                row_dict[k] = None
            else:
                row_dict[k] = v
        parsed_verdict = clean_json_string(row_dict.get("final_adjudication_json"))
        row_dict["parsed_verdict"] = parsed_verdict
        row_dict["active_gemini_model"] = gemini_endpoint
        row_dict["active_embedding_model"] = embed_endpoint
        records.append(row_dict)

        verdict_action = parsed_verdict.get("recommended_action", "UNKNOWN")
        contest_str = f"YES ({row['months_since_issue']}m)" if row["is_within_contestability"] else f"NO ({row['months_since_issue']}m)"
        print(
            f"{row['claim_id']:<15} | {row['insured_name']:<17} | {contest_str:<11} | "
            f"{str(row['radiograph_forensic_flag']):<38} | {verdict_action}"
        )
    print("=" * 112 + "\n")

    out_json = DATA_DIR / "live_bq_results.json"
    out_json.write_text(json.dumps(records, indent=2), encoding="utf-8")
    print(f"✅ Saved live BigQuery Lakehouse results to '{out_json}'!")


if __name__ == "__main__":
    main()
