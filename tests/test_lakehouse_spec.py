"""
Spec-Driven Verification Tests (`tests/test_lakehouse_spec.py`)
Validates:
  1. Multi-modal Life Insurance dataset generation (Parquet/CSV, PNGs, PDFs, WAVs)
  2. Biometric/Radiograph duplicate fraud case hash perturbation (CLM-CI-1004 vs PRIOR_CLM_8821_FRAUD)
  3. SQL DDL & Multimodal query template integrity (Managed Iceberg, Object Tables, ObjectRef, OBJ.GET_ACCESS_URL)
"""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


DATA_DIR = Path("data")


def test_structured_life_insurance_tables() -> None:
    claims_pq = DATA_DIR / "structured" / "iceberg_claims_policies.parquet"
    atlas_pq = DATA_DIR / "structured" / "dim_ref_medical_atlas_meta.parquet"
    assert claims_pq.exists(), "iceberg_claims_policies.parquet missing"
    assert atlas_pq.exists(), "dim_ref_medical_atlas_meta.parquet missing"

    df_claims = pd.read_parquet(claims_pq)
    df_atlas = pd.read_parquet(atlas_pq)

    assert len(df_claims) == 5, "Expected exactly 5 Hero Life Insurance Claims"
    assert len(df_atlas) == 6, "Expected exactly 6 Reference Medical Atlas entries"

    expected_ids = {"CLM-LIFE-1001", "CLM-LIFE-1002", "CLM-ADD-1003", "CLM-CI-1004", "CLM-CI-1005"}
    assert set(df_claims["claim_id"]) == expected_ids

    # Check 24-month Contestability Window flags
    row_1001 = df_claims[df_claims["claim_id"] == "CLM-LIFE-1001"].iloc[0]
    row_1002 = df_claims[df_claims["claim_id"] == "CLM-LIFE-1002"].iloc[0]
    assert not bool(row_1001["is_within_contestability"]) and row_1001["months_since_issue"] == 68
    assert bool(row_1002["is_within_contestability"]) and row_1002["months_since_issue"] == 7


def test_radiograph_images_and_fraud_perturbation() -> None:
    ref_dir = DATA_DIR / "unstructured_evidence" / "ref_medical_atlas"
    claim_dir = DATA_DIR / "unstructured_evidence" / "claim_radiographs"

    ref_pngs = list(ref_dir.glob("*.png"))
    claim_pngs = list(claim_dir.glob("*.png"))
    assert len(ref_pngs) == 6
    assert len(claim_pngs) == 5

    # Verify CLM-CI-1004_xray.png vs PRIOR_CLM_8821_FRAUD.png:
    # Different SHA256 (defeating exact hash checks), but >0.999 pixel cosine similarity!
    fraud_claim_path = claim_dir / "CLM-CI-1004_xray.png"
    prior_claim_path = ref_dir / "PRIOR_CLM_8821_FRAUD.png"
    sha_claim = hashlib.sha256(fraud_claim_path.read_bytes()).hexdigest()
    sha_prior = hashlib.sha256(prior_claim_path.read_bytes()).hexdigest()
    assert sha_claim != sha_prior, "Perturbed duplicate must have different SHA256"

    arr_a = np.asarray(Image.open(fraud_claim_path), dtype=np.float64).ravel()
    arr_b = np.asarray(Image.open(prior_claim_path), dtype=np.float64).ravel()
    cos_sim = float(np.dot(arr_a, arr_b) / (np.linalg.norm(arr_a) * np.linalg.norm(arr_b)))
    assert cos_sim > 0.9999, f"Expected >0.9999 image similarity, got {cos_sim}"


def test_pdfs_and_wav_audio_files() -> None:
    pdf_dir = DATA_DIR / "unstructured_evidence" / "policy_and_aps_pdfs"
    audio_dir = DATA_DIR / "unstructured_evidence" / "beneficiary_fnol_audio"

    pdfs = list(pdf_dir.glob("*.pdf"))
    wavs = list(audio_dir.glob("*.wav"))
    assert len(pdfs) == 9, "Expected 4 Master Policy PDFs + 5 APS PDFs"
    assert len(wavs) == 5, "Expected 5 Beneficiary FNOL WAV audio files"

    for p in pdfs:
        header = p.read_bytes()[:4]
        assert header == b"%PDF", f"Invalid PDF header in {p}"

    for w in wavs:
        with wave.open(str(w), "rb") as wf:
            assert wf.getnchannels() == 1
            assert wf.getnframes() > 16000


def test_sql_templates_conform_to_lakehouse_spec() -> None:
    ddl_sql = Path("sql/01_lakehouse_ddl.sql").read_text(encoding="utf-8")
    query_sql = Path("sql/02_multimodal_query.sql").read_text(encoding="utf-8")

    # Verify Managed Apache Iceberg DDL syntax
    assert "table_format = 'ICEBERG'" in ddl_sql
    assert "file_format = 'PARQUET'" in ddl_sql
    assert "WITH CONNECTION" in ddl_sql
    assert "object_metadata = 'SIMPLE'" in ddl_sql

    # Verify Multimodal Vector Search + ObjectRef + Signed URLs
    assert "ML.GENERATE_EMBEDDING" in query_sql
    assert "VECTOR_SEARCH" in query_sql
    assert "OBJ.MAKE_REF" in query_sql
    assert "OBJ.FETCH_METADATA" in query_sql
    assert "OBJ.GET_ACCESS_URL" in query_sql


def test_streamlit_app_data_loader() -> None:
    import app

    records, label = app.load_claim_records(prefer_live_bq=False)
    assert len(records) == 5
    required_ui_keys = {
        "claim_id",
        "policy_id",
        "policy_code",
        "insured_name",
        "dob",
        "gender",
        "uw_health_class",
        "declared_smoker",
        "declared_medical_history",
        "beneficiary_name",
        "beneficiary_relationship",
        "reported_cause_of_event",
        "months_since_issue",
        "is_within_contestability",
        "base_sum_assured",
        "add_rider_sum_assured",
        "ci_rider_sum_assured",
        "claimed_benefit_amount",
        "fnol_transcript_reference",
        "aps_text_reference",
        "policy_text_reference",
        "parsed_verdict",
    }
    for r in records:
        missing = required_ui_keys - set(r.keys())
        assert not missing, f"Missing UI keys in record {r.get('claim_id')}: {missing}"
        assert r["dob"] is not None and r["gender"] is not None
        assert r["parsed_verdict"].get("recommended_action") == r["expected_action"]

    # Full headless Streamlit UI execution test across all 5 claims
    import io
    import logging
    from streamlit.testing.v1 import AppTest

    log_capture = io.StringIO()
    handler = logging.StreamHandler(log_capture)
    st_logger = logging.getLogger("streamlit")
    st_logger.addHandler(handler)
    try:
        app_path = Path(__file__).resolve().parent.parent / "app.py"
        at = AppTest.from_file(str(app_path), default_timeout=30)
        at.run()
        assert not at.exception, f"Streamlit UI exception on initial load: {at.exception}"
        for idx in range(5):
            at.selectbox[0].select_index(idx).run()
            assert not at.exception, f"Streamlit UI exception on claim index {idx}: {at.exception}"
        logs = log_capture.getvalue()
        assert "use_container_width" not in logs, f"Deprecated Streamlit parameter warning detected: {logs}"
    finally:
        st_logger.removeHandler(handler)



