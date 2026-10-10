#!/usr/bin/env python3
"""
OmniClaim Life 360 — Minimalist Educational Workbench (`app.py`)

A clean, story-driven interactive guide showing how Google Cloud's Multi-Modal Lakehouse
solves complex Life & Living Benefits Insurance claims across 4 modalities in a single SQL namespace:
  • Step 1: Structured Policy & 24-Month Contestability (`BigLake Managed Apache Iceberg`)
  • Step 2: Clinical Radiograph Forensics (`BigLake Object Tables` + `gemini-embedding-2-preview` + `VECTOR_SEARCH`)
  • Step 3: Physician APS & Policy Contract Audit (`ObjectRef` + `Gemini 3.8 Flash` — Zero OCR ETL)
  • Step 4: Recorded Beneficiary Call & Final Verdict (`ObjectRef` + `Gemini 3.8 Flash` — Native Audio SQL)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
REGION = os.getenv("GCP_REGION", "us-central1")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", f"{PROJECT_ID}-bucket")
DATASET_ID = os.getenv("BQ_DATASET_ID", "omniclaim_lakehouse")

DATA_DIR = Path("data")

st.set_page_config(
    page_title="OmniClaim Life 360 | Google Cloud Multi-Modal Lakehouse",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================================
# DATA LOADING (LIVE BIGQUERY, CACHED BIGQUERY SNAPSHOT, OR LOCAL PREVIEW)
# ============================================================================

@st.cache_data(ttl=120)
def fetch_live_bigquery_results() -> list[dict[str, Any]] | None:
    """Queries `vw_claim_adjudication_360` live from Google Cloud BigQuery."""
    try:
        from google.cloud import bigquery
        from run_lakehouse import clean_json_string, get_gcp_credentials

        client = bigquery.Client(project=PROJECT_ID, location=REGION, credentials=get_gcp_credentials())
        query = f"""
        SELECT *
        FROM `{PROJECT_ID}.{DATASET_ID}.vw_claim_adjudication_360`
        ORDER BY claim_id
        """
        df = client.query(query).to_dataframe()
        records = []
        for _, row in df.iterrows():
            d = {k: (v.isoformat() if hasattr(v, "isoformat") else (None if pd.isna(v) else v)) for k, v in row.items()}
            d["parsed_verdict"] = clean_json_string(d.get("final_adjudication_json"))
            d["source_mode"] = "LIVE_BIGQUERY_QUERY"
            records.append(d)
        return records
    except Exception:
        return None


def load_claim_records(prefer_live_bq: bool = True) -> tuple[list[dict[str, Any]], str]:
    """Returns (list of 5 enriched claim dicts, active_source_label)."""
    df_local = pd.read_parquet(DATA_DIR / "structured" / "iceberg_claims_policies.parquet")
    local_lookup = {row["claim_id"]: row.to_dict() for _, row in df_local.iterrows()}

    # 1. Try Live BigQuery query if requested
    if prefer_live_bq:
        live_rows = fetch_live_bigquery_results()
        if live_rows:
            for r in live_rows:
                cid = r["claim_id"]
                if cid in local_lookup:
                    for k, v in local_lookup[cid].items():
                        if r.get(k) is None:
                            r[k] = v
            return live_rows, f"⚡ Live BigQuery (`{DATASET_ID}.vw_claim_adjudication_360`)"

    # 2. Check cached live BigQuery execution results (`data/live_bq_results.json`)
    live_cache_path = DATA_DIR / "live_bq_results.json"
    if live_cache_path.exists():
        cached_rows = json.loads(live_cache_path.read_text(encoding="utf-8"))
        for r in cached_rows:
            cid = r["claim_id"]
            if cid in local_lookup:
                for k, v in local_lookup[cid].items():
                    if r.get(k) is None:
                        r[k] = v
        return cached_rows, f"☁️ Verified BigQuery Snapshot (`{DATASET_ID}.vw_claim_adjudication_360`)"

    # 3. Local Preview Fallback
    fallback_matches = {
        "CLM-LIFE-1001": ("REF_PNEUMONIA_001", 0.0124, 0.9876, "ACUTE_LOBAR_PNEUMONIA_RLL", None, "CONSISTENT_ACUTE_OR_TRAUMA_FINDING"),
        "CLM-LIFE-1002": ("REF_CARDIOMEGALY_001", 0.0645, 0.9355, "SEVERE_CARDIOMEGALY_WITH_PACEMAKER", None, "CHRONIC_PRE_EXISTING_CARDIAC_INDICATOR"),
        "CLM-ADD-1003": ("REF_TRAUMA_001", 0.0453, 0.9547, "HIGH_ENERGY_CLAVICULAR_THORACIC_FRACTURE", None, "CONSISTENT_ACUTE_OR_TRAUMA_FINDING"),
        "CLM-CI-1004": ("PRIOR_CLM_8821_FRAUD", 0.0038, 0.9962, "PRIOR_CLAIM_LUL_MASS_ARTIFACT_8821", "PRIOR-CLM-8821", "CRITICAL_DUPLICATE_IMAGE_FRAUD"),
        "CLM-CI-1005": ("REF_PULM_MASS_001", 0.0332, 0.9668, "MALIGNANT_RUL_PULMONARY_NEOPLASM_28MM", None, "CONSISTENT_ACUTE_OR_TRAUMA_FINDING"),
    }
    fallback_verdicts = {
        "CLM-LIFE-1001": {
            "audio_transcript_summary": "Daughter Clara Vance reports sudden acute right-lower-lobe pneumonia hospitalization 68 months after policy issuance.",
            "audio_discrepancies_detected": "NONE — Caller testimony matches Unity Hospital APS and chest radiograph.",
            "contestability_alert": False,
            "rider_exclusion_triggered": "NONE",
            "duplicate_image_fraud": False,
            "recommended_action": "AUTO_APPROVE_PAYOUT",
            "approved_payout_amount": 500000,
            "executive_rationale": "Policy POL-10001 has been in force for 68 months (past the 24-month Incontestability Clause). BigQuery Vector Search confirms acute RLL lobar pneumonia consistent with the APS PDF and beneficiary FNOL call.",
        },
        "CLM-LIFE-1002": {
            "audio_transcript_summary": "Spouse Victoria Sterling states insured collapsed at home and was treated by Dr. Alan Grant, his cardiologist of 4 years who monitored his pacemaker.",
            "audio_discrepancies_detected": "CRITICAL CONTRADICTION — Spouse admits 4-year cardiology history & pacemaker, contradicting the Preferred Plus application declaration ('Zero cardiac history').",
            "contestability_alert": True,
            "rider_exclusion_triggered": "Section 1.2 & Section 1.4 Material Misrepresentation within 24-Month Contestability Window",
            "duplicate_image_fraud": False,
            "recommended_action": "SIU_RESCISSION_HOLD",
            "approved_payout_amount": 0,
            "executive_rationale": "Claim occurred 7 months after policy issuance (inside the 24-month Contestability Window). Radiograph vector search matches Severe Stage III Cardiomegaly with CRT Pacemaker, and both the APS PDF and spouse FNOL audio confirm material misrepresentation at underwriting.",
        },
        "CLM-ADD-1003": {
            "audio_transcript_summary": "Brother David Thorne admits insured Marcus Thorne was performing an unroped free-solo ice climb at 4,250m on Mount Rainier when he fell.",
            "audio_discrepancies_detected": "Paper claim form stated 'recreational walking trail', whereas brother's audio and Coroner APS confirm unroped free-solo alpine ice climbing at 4,250m.",
            "contestability_alert": False,
            "rider_exclusion_triggered": "Section 4.2(a) AD&D Exclusion — Free-solo / unroped technical mountaineering above 4,000 meters",
            "duplicate_image_fraud": False,
            "recommended_action": "PARTIAL_APPROVE_BASE_DENY_ADD_RIDER",
            "approved_payout_amount": 1000000,
            "executive_rationale": "Because Policy POL-10003 has been in force for 38 months (>24m Incontestability), the $1,000,000 Base Life benefit is payable. However, the $1,000,000 AD&D Double-Indemnity Rider is denied under Section 4.2(a) due to unroped ice climbing above 4,000m.",
        },
        "CLM-CI-1004": {
            "audio_transcript_summary": "Claimant Victor Krum demands immediate $350,000 wire transfer for a critical illness claim while refusing to disclose the treating hospital or pathology lab.",
            "audio_discrepancies_detected": "Caller refuses to identify hospital or biopsy accession number; clinic license #UNV-0091 is unverified.",
            "contestability_alert": True,
            "rider_exclusion_triggered": "Section 5.3 Biometric & Diagnostic Anti-Fraud Clause (Syndicated Radiograph Reuse)",
            "duplicate_image_fraud": True,
            "recommended_action": "SIU_FRAUD_RING_REJECT",
            "approved_payout_amount": 0,
            "executive_rationale": "BigQuery VECTOR_SEARCH detected a 0.9962 cosine similarity between the submitted radiograph and historical Claim PRIOR-CLM-8821 submitted 9 months ago under another policy, despite a modified file hash.",
        },
        "CLM-CI-1005": {
            "audio_transcript_summary": "Insured Diana Chen reports confirmed 30x20mm Stage IIA right-upper-lobe lung adenocarcinoma diagnosed via biopsy.",
            "audio_discrepancies_detected": "NONE — Caller statement matches Oncology APS biopsy report and RUL radiograph.",
            "contestability_alert": False,
            "rider_exclusion_triggered": "NONE — Meets Section 2.1 Critical Illness Severity Threshold (>= 2.0cm malignant pulmonary neoplasm)",
            "duplicate_image_fraud": False,
            "recommended_action": "APPROVE_LIVING_BENEFIT",
            "approved_payout_amount": 250000,
            "executive_rationale": "Policy POL-10005 has been in force for 29 months. Radiograph vector search confirms a solitary RUL pulmonary neoplasm with zero prior-claim duplication, corroborated by histopathological biopsy in the APS PDF.",
        },
    }

    rows = []
    for cid, d in local_lookup.items():
        ref_id, dist, sim, path_lbl, prior_id, flag = fallback_matches[cid]
        d = dict(d)
        d["matched_ref_image_id"] = ref_id
        d["cosine_distance"] = dist
        d["cosine_similarity"] = sim
        d["detected_radiograph_pathology"] = path_lbl
        d["matched_prior_claim_id"] = prior_id
        d["radiograph_forensic_flag"] = flag
        d["pdf_clinical_contract_audit"] = (
            f"• **APS Clinical Extraction**: {d['aps_text_reference']}\n"
            f"• **Policy Wording Cross-Check**: Evaluated against `{d['policy_code']}` Incontestability & Rider clauses."
        )
        d["parsed_verdict"] = fallback_verdicts[cid]
        d["final_adjudication_json"] = json.dumps(fallback_verdicts[cid], indent=2)
        rows.append(d)
    return rows, "📦 Local Preview Dataset (`data/`)"


# ============================================================================
# STORYLINES & EDUCATIONAL METADATA
# ============================================================================

VERDICT_STYLES = {
    "AUTO_APPROVE_PAYOUT": ("#0f5132", "#d1e7dd", "✅ Fast-Track Approved"),
    "APPROVE_LIVING_BENEFIT": ("#0f5132", "#d1e7dd", "✅ Living Benefit Approved"),
    "PARTIAL_APPROVE_BASE_DENY_ADD_RIDER": ("#664d03", "#fff3cd", "⚠️ Partial Payout (Base Paid / AD&D Denied)"),
    "SIU_RESCISSION_HOLD": ("#842029", "#f8d7da", "🚨 Rescission Hold (24m Contestability Fraud)"),
    "SIU_FRAUD_RING_REJECT": ("#842029", "#f8d7da", "🛑 Rejected (Recycled X-Ray Fraud Ring)"),
}

CUSTOMER_STORYLINES: dict[str, dict[str, str]] = {
    "CLM-LIFE-1001": {
        "short_label": "Case 1: Eleanor Vance — Clean 68m Term Life ($500K Fast-Track)",
        "mystery": "A $500,000 natural death claim arrives after 68 months in force. Can the Lakehouse verify the hospital X-ray, physician PDF, and daughter's call in one pass for zero-touch payout?",
        "nih_citation": "NIH PubMed Central `PMC10405870` (PMID: 37554481) — Acute Right Lower Lobe bacterial pneumonia",
        "key_lesson": "When structured Iceberg policy age (>24m incontestable), radiograph vector search, APS PDF, and beneficiary audio all align, BigQuery auto-adjudicates the claim without human bottlenecks.",
    },
    "CLM-LIFE-1002": {
        "short_label": "Case 2: Robert Sterling — Concealed Pacemaker in 24m Window ($1.5M Fraud)",
        "mystery": "Insured signed a 'Preferred Plus' application claiming ZERO heart disease, then died just 7 months later. What is hiding inside his emergency chest X-ray and spouse's phone call?",
        "nih_citation": "NIH PubMed Central `PMC7033380` (PMID: 32095464) — Severe Stage III Cardiomegaly & Biventricular CRT Pacemaker (Panel A vs. Panel B)",
        "key_lesson": "A tabular database only sees an ICD-10 text code. BigQuery Vector Search + Gemini 3.8 Flash catch the implanted pacemaker on the X-ray and the spouse's verbal admission of a 4-year cardiology history.",
    },
    "CLM-ADD-1003": {
        "short_label": "Case 3: Marcus Thorne — $2M AD&D Mountaineering Exclusion Conflict",
        "mystery": "The paper claim form says 'slip on a walking trail' to claim a $1M Base + $1M Accidental Death double-indemnity rider. Why do the trauma X-ray, coroner PDF, and brother's voice call disagree?",
        "nih_citation": "NIH PubMed Central `PMC5702695` (PMID: 29181345) — High-energy displaced mid-shaft clavicular fracture",
        "key_lesson": "The 38-month policy is incontestable for the $1M Base Life benefit, but Gemini 3.8 Flash cross-checks Policy Section 4.2(a) against the coroner PDF and recorded audio (unroped climbing at 4,250m) to deny the $1M AD&D Rider.",
    },
    "CLM-CI-1004": {
        "short_label": "Case 4: Victor Krum — Stolen & Recycled X-Ray Fraud Ring ($350K CI)",
        "mystery": "A fraudster modifies a corner pixel of a stolen lung cancer X-ray so its MD5/SHA-256 hash looks brand new. Can BigQuery `VECTOR_SEARCH` catch the duplicate across historical claims?",
        "nih_citation": "NIH MedPix `MPX1219` (`synpic14404`) — Left Upper Lobe Non-Small Cell Lung Cancer mass",
        "key_lesson": "Cryptographic file hashes (`SHA-256`) break when a single pixel changes. Dense multimodal embeddings (`gemini-embedding-2-preview`) catch the `0.9962` cosine similarity against prior claim `PRIOR-CLM-8821`.",
    },
    "CLM-CI-1005": {
        "short_label": "Case 5: Diana Chen — Stage IIA Lung Cancer Living Benefit ($250K CI)",
        "mystery": "A non-smoking policyholder files a $250,000 Critical Illness Living Benefit claim to fund immediate cancer treatment. Does her tumor meet the Policy Section 2.1 (>= 2.0 cm) severity threshold?",
        "nih_citation": "NIH PubMed Central `PMC7529567` (PMID: 32869499) — Solitary 30x20mm Right Upper Lobe lung adenocarcinoma",
        "key_lesson": "BigQuery verifies zero prior-claim image reuse via `VECTOR_SEARCH` while Gemini 3.8 Flash reads the pathology biopsy PDF (`3.0 cm`, Stage IIA) via `ObjectRef` to approve her living benefit.",
    },
}


def extract_signed_read_url(val: Any) -> str | None:
    """Extracts the HTTPS read_url from BigQuery OBJ.GET_ACCESS_URL JSON output."""
    if not val:
        return None
    if isinstance(val, dict):
        return val.get("access_urls", {}).get("read_url")
    if isinstance(val, str):
        s = val.strip()
        if s.startswith("{"):
            try:
                parsed = json.loads(s)
                return parsed.get("access_urls", {}).get("read_url")
            except Exception:
                return None
        if s.startswith("http"):
            return s
    return None


# ============================================================================
# MAIN STREAMLIT WORKBENCH UI
# ============================================================================

def main() -> None:
    # Minimalist Sidebar: Case Selector + Data Source Toggle + Architecture Cheat Sheet
    with st.sidebar:
        st.markdown("## 🛡️ OmniClaim Life 360")
        st.caption("Google Cloud Multi-Modal Lakehouse POC")

        records, source_label = load_claim_records(prefer_live_bq=False)

        claim_options = {
            CUSTOMER_STORYLINES[r["claim_id"]]["short_label"]: r["claim_id"]
            for r in records
        }
        selected_label = st.selectbox(
            "1️⃣ Choose a Customer Storyline:",
            list(claim_options.keys()),
            index=1,  # Default to Case 2 (Robert Sterling Pacemaker Contestability)
        )
        selected_claim_id = claim_options[selected_label]

        st.divider()
        query_live = st.toggle("Query Live BigQuery View", value=False)
        if query_live:
            records, source_label = load_claim_records(prefer_live_bq=True)
        st.caption(f"**Data Source:** {source_label}")

        st.divider()
        st.markdown("#### ☁️ GCP Lakehouse Building Blocks")
        st.markdown(
            """
            1. **Apache Iceberg (`PARQUET`)**
               Structured policy & contestability rules on GCS.
            2. **BigLake Object Tables**
               Governs raw `.png`, `.pdf`, and `.wav` files in GCS via `ObjectRef`.
            3. **`VECTOR_SEARCH` (`COSINE`)**
               3072-D `gemini-embedding-2-preview` visual forensics.
            4. **`ML.GENERATE_TEXT`**
               `gemini-3.8-flash` reads PDFs & listens to audio directly in SQL.
            """
        )

    claim = next(r for r in records if r["claim_id"] == selected_claim_id)
    verdict = claim.get("parsed_verdict") or {}
    story = CUSTOMER_STORYLINES[selected_claim_id]

    action_code = verdict.get("recommended_action", claim.get("expected_action", "UNKNOWN"))
    fg_color, bg_color, short_verdict = VERDICT_STYLES.get(
        action_code, ("#084298", "#cfe2ff", action_code)
    )
    approved_amt = float(verdict.get("approved_payout_amount", claim.get("expected_payout", 0.0)))
    claimed_amt = float(claim["claimed_benefit_amount"])

    # ========================================================================
    # CLEAN TOP HERO CARD: CUSTOMER STORYLINE & UNIFIED OUTCOME
    # ========================================================================
    st.markdown(
        f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-left: 6px solid {fg_color}; border-radius: 10px; padding: 18px 22px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap; gap: 12px;">
                <div style="flex: 1; min-width: 300px;">
                    <div style="font-size: 0.82rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: #475569;">
                        {claim['claim_id']} • {claim['product_type']} • Policy {claim['policy_id']}
                    </div>
                    <h3 style="margin: 4px 0 8px 0; color: #0f172a;">{story['short_label']}</h3>
                    <p style="margin: 0; color: #334155; font-size: 0.98rem;">
                        <b>🕵️ The Insurance Investigation Question:</b> {story['mystery']}
                    </p>
                </div>
                <div style="background: {bg_color}; color: {fg_color}; padding: 12px 18px; border-radius: 8px; text-align: right; min-width: 220px;">
                    <div style="font-size: 0.85rem; font-weight: 700;">{short_verdict}</div>
                    <div style="font-size: 1.45rem; font-weight: 800; margin-top: 2px;">
                        ${approved_amt:,.0f} <span style="font-size: 0.9rem; font-weight: 500;">/ ${claimed_amt:,.0f} claimed</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ========================================================================
    # STEP-BY-STEP EDUCATIONAL TABS (ONE MODALITY AT A TIME)
    # ========================================================================
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "1️⃣ Step 1: Policy & Contestability (Iceberg)",
            "2️⃣ Step 2: X-Ray Vector Search (Embeddings)",
            "3️⃣ Step 3: Physician & Policy PDFs (Gemini)",
            "4️⃣ Step 4: FNOL Audio & Final Verdict (Gemini)",
            "📊 All 5 Cases & Lakehouse Architecture",
        ]
    )

    # ------------------------------------------------------------------------
    # TAB 1: STRUCTURED APACHE ICEBERG
    # ------------------------------------------------------------------------
    with tab1:
        st.markdown("#### 🧊 Capability 1: Open Table Format (`BigLake Managed Apache Iceberg`)")
        st.caption(
            "In Life Insurance, every claim begins by checking whether the event occurred inside the **24-Month Contestability Window** "
            "and comparing the claim against the policyholder's **Signed Underwriting Declaration**."
        )

        months = int(claim["months_since_issue"])
        is_contestable = bool(claim["is_within_contestability"])

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Insured Customer", claim["insured_name"], f"DOB: {claim['dob']} ({claim['gender']})")
        m2.metric("Time Since Policy Issue", f"{months} Months", "Contestable (<24m)" if is_contestable else "Incontestable (>24m)")
        m3.metric("Base + Rider Coverage", f"${float(claim['base_sum_assured']):,.0f} Base", f"AD&D: ${float(claim['add_rider_sum_assured']):,.0f} | CI: ${float(claim['ci_rider_sum_assured']):,.0f}")
        m4.metric("Underwriting Tier", claim["uw_health_class"], f"Smoker: {claim['declared_smoker']}")

        col_uw, col_clm = st.columns(2, gap="large")
        with col_uw:
            st.info(
                f"📝 **What {claim['insured_name']} Declared at Policy Issue (`{claim['issue_date']}`)**\n\n"
                f"*\"{claim['declared_medical_history']}\"*"
            )
        with col_clm:
            if is_contestable:
                st.error(
                    f"🚨 **What Was Filed on the Claim Form (`{claim['event_date']}` — {months} Months Later)**\n\n"
                    f"**Reported Cause:** {claim['reported_cause_of_event']}\n\n"
                    f"**Beneficiary:** {claim['beneficiary_name']} ({claim['beneficiary_relationship']})\n\n"
                    f"⚠️ *Policy is within the 24-month Contestability Window — requires multi-modal medical & audio verification.*"
                )
            else:
                st.success(
                    f"✅ **What Was Filed on the Claim Form (`{claim['event_date']}` — {months} Months Later)**\n\n"
                    f"**Reported Cause:** {claim['reported_cause_of_event']}\n\n"
                    f"**Beneficiary:** {claim['beneficiary_name']} ({claim['beneficiary_relationship']})\n\n"
                    f"*Base policy is past the 24-month Contestability Window (`Incontestable`).*"
                )

        with st.expander("💡 How Google Cloud Lakehouse Does This in SQL (`BigLake Managed Apache Iceberg`)"):
            st.code(
                f"""-- Stored as open Parquet + Iceberg manifests in gs://{BUCKET_NAME}/iceberg_warehouse/
CREATE TABLE IF NOT EXISTS `{PROJECT_ID}.{DATASET_ID}.iceberg_claims_policies` (
    claim_id STRING NOT NULL,
    policy_id STRING NOT NULL,
    insured_name STRING,
    uw_health_class STRING,
    declared_medical_history STRING,
    issue_date DATE,
    event_date DATE,
    months_since_issue INT64,
    is_within_contestability BOOL,
    claimed_benefit_amount FLOAT64
)
CLUSTER BY claim_id, policy_id
WITH CONNECTION `{PROJECT_ID}.{REGION}.lakehouse-omniclaim-conn`
OPTIONS (
    file_format = 'PARQUET',
    table_format = 'ICEBERG',
    storage_uri = 'gs://{BUCKET_NAME}/iceberg_warehouse/iceberg_claims_policies'
);""",
                language="sql",
            )

    # ------------------------------------------------------------------------
    # TAB 2: LONGITUDINAL X-RAY VECTOR SEARCH
    # ------------------------------------------------------------------------
    with tab2:
        st.markdown("#### 🩻 Capability 2: Unstructured Image Forensics (`BigLake Object Tables` + `VECTOR_SEARCH`)")
        st.caption(
            f"BigQuery catalogs raw chest/trauma radiographs directly in GCS (`obj_claim_radiographs`), embeds them with **`gemini-embedding-2-preview`** (3072-D), "
            f"and compares the customer's **Underwriting Baseline X-Ray** against the **Claim Event X-Ray** and our **Reference Pathology / Prior-Claims Atlas**."
        )

        flag = claim.get("radiograph_forensic_flag", "")
        sim = float(claim.get("cosine_similarity", 0.0))
        dist = float(claim.get("cosine_distance", 0.0))
        matched_ref = claim.get("matched_ref_image_id", "")
        pathology = claim.get("detected_radiograph_pathology", "")

        if flag == "CRITICAL_DUPLICATE_IMAGE_FRAUD":
            st.error(
                f"🛑 **DUPLICATE X-RAY FRAUD CAUGHT BY VECTOR SEARCH**: The submitted radiograph matches historical claim "
                f"`{claim.get('matched_prior_claim_id')}` (`{matched_ref}`) with **`{sim:.4f}` Cosine Similarity** (`distance = {dist:.4f}`), "
                f"even though the attacker modified a pixel so the file's SHA-256 hash looked brand new!"
            )
        elif flag == "CHRONIC_PRE_EXISTING_CARDIAC_INDICATOR":
            st.warning(
                f"🚨 **UNDISCLOSED PRE-EXISTING PATHOLOGY DETECTED**: Compared against the healthy underwriting baseline, the emergency X-ray "
                f"matches `{pathology}` (**`{sim:.4f}` Cosine Similarity**)—visibly revealing severe heart enlargement and an implanted pacemaker generator with leads!"
            )
        else:
            st.success(
                f"✅ **CONSISTENT CLINICAL RADIOGRAPH**: BigQuery `VECTOR_SEARCH` matches `{matched_ref}` (`{pathology}`) "
                f"with **`{sim:.4f}` Cosine Similarity** (`distance = {dist:.4f}`) and zero prior-claim duplication."
            )

        img_col0, img_col1, img_col2 = st.columns(3, gap="medium")
        local_norm_xray = DATA_DIR / "unstructured_evidence" / "ref_medical_atlas" / "REF_NORM_001.png"
        local_claim_xray = DATA_DIR / "unstructured_evidence" / "claim_radiographs" / f"{claim['claim_id']}_xray.png"
        local_ref_xray = DATA_DIR / "unstructured_evidence" / "ref_medical_atlas" / f"{matched_ref}.png"

        signed_claim_xray = extract_signed_read_url(claim.get("signed_claim_xray_url"))
        signed_ref_xray = extract_signed_read_url(claim.get("signed_matched_ref_xray_url"))

        with img_col0:
            st.markdown(f"**1. Underwriting Baseline (`{claim['issue_date']}`)**")
            if local_norm_xray.exists():
                st.image(str(local_norm_xray), width="stretch")
            st.caption("Clean Baseline (`REF_NORM_001` • NIH `PMC12127709`)")

        with img_col1:
            st.markdown(f"**2. Claim Event X-Ray (`{claim['event_date']}`)**")
            if local_claim_xray.exists():
                st.image(str(local_claim_xray), width="stretch")
            elif signed_claim_xray:
                st.image(signed_claim_xray, width="stretch")
            st.caption(f"Submitted: `{claim['claim_id']}_xray.png` ({story['nih_citation']})")

        with img_col2:
            st.markdown(f"**3. Top-1 BigQuery Vector Match (`{matched_ref}`)**")
            if local_ref_xray.exists():
                st.image(str(local_ref_xray), width="stretch")
            elif signed_ref_xray:
                st.image(signed_ref_xray, width="stretch")
            st.caption(f"Cosine Similarity: **`{sim:.4f}`** (`distance = {dist:.4f}`)")

        with st.expander("💡 How Google Cloud Lakehouse Does This in SQL (`ML.GENERATE_EMBEDDING` + `VECTOR_SEARCH`)"):
            st.code(
                f"""-- 1. Generate 3072-D multimodal embeddings directly over the GCS Object Table
CREATE OR REPLACE TABLE `{PROJECT_ID}.{DATASET_ID}.claims_image_embeddings` AS
SELECT uri AS image_gcs_uri, ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
    MODEL `{PROJECT_ID}.{DATASET_ID}.model_multimodal_embedding`, -- gemini-embedding-2-preview
    TABLE `{PROJECT_ID}.{DATASET_ID}.obj_claim_radiographs`
);

-- 2. In-Engine Cosine Vector Search against Reference Medical & Prior-Claims Atlas
SELECT
    query.claim_id,
    base.ref_image_id AS matched_ref_image_id,
    ROUND(1.0 - distance, 4) AS cosine_similarity
FROM VECTOR_SEARCH(
    TABLE `{PROJECT_ID}.{DATASET_ID}.ref_medical_atlas_embeddings`, 'embedding',
    TABLE `{PROJECT_ID}.{DATASET_ID}.claims_image_embeddings`, 'embedding',
    top_k => 1, distance_type => 'COSINE', options => '{{"use_brute_force": true}}'
);""",
                language="sql",
            )

    # ------------------------------------------------------------------------
    # TAB 3: PHYSICIAN APS & POLICY CONTRACT PDFs
    # ------------------------------------------------------------------------
    with tab3:
        st.markdown("#### 📄 Capability 3: Zero-OCR Clinical & Contract PDF Audit (`ObjectRef` + `Gemini 3.8 Flash`)")
        st.caption(
            "Instead of maintaining an external OCR pipeline, BigQuery constructs an **`ObjectRef`** pointer to the Attending Physician Statement (APS) PDF "
            "and Master Policy Wording PDF in GCS and passes both documents directly into **`gemini-3.8-flash`** inside `ML.GENERATE_TEXT`."
        )

        col_audit, col_docs = st.columns([1.05, 1.15], gap="large")
        with col_audit:
            st.markdown("##### 🤖 Gemini 3.8 Flash Clinical & Contract Audit Output")
            st.info(claim.get("pdf_clinical_contract_audit", "_No PDF audit output available._"))

        with col_docs:
            st.markdown("##### 📂 Inspect the Underlying PDF Documents in GCS")
            doc_subtab1, doc_subtab2 = st.tabs(
                [
                    f"🩺 Physician APS ({claim['claim_id']}_aps.pdf)",
                    f"📜 Policy Contract ({claim['policy_code']}.pdf)",
                ]
            )
            local_aps_pdf = DATA_DIR / "unstructured_evidence" / "policy_and_aps_pdfs" / f"{claim['claim_id']}_aps.pdf"
            local_pol_pdf = DATA_DIR / "unstructured_evidence" / "policy_and_aps_pdfs" / f"{claim['policy_code']}.pdf"
            signed_aps_pdf = extract_signed_read_url(claim.get("signed_aps_pdf_url"))
            signed_pol_pdf = extract_signed_read_url(claim.get("signed_policy_pdf_url"))

            with doc_subtab1:
                st.code(claim.get("aps_text_reference", ""), language="text")
                if local_aps_pdf.exists():
                    st.download_button(
                        label=f"⬇️ Download {claim['claim_id']}_aps.pdf",
                        data=local_aps_pdf.read_bytes(),
                        file_name=f"{claim['claim_id']}_aps.pdf",
                        mime="application/pdf",
                    )
                if signed_aps_pdf:
                    st.markdown(f"🔗 [Open Time-Limited Signed GCS URL (`OBJ.GET_ACCESS_URL`)]({signed_aps_pdf})")

            with doc_subtab2:
                st.code(claim.get("policy_text_reference", ""), language="text")
                if local_pol_pdf.exists():
                    st.download_button(
                        label=f"⬇️ Download {claim['policy_code']}.pdf",
                        data=local_pol_pdf.read_bytes(),
                        file_name=f"{claim['policy_code']}.pdf",
                        mime="application/pdf",
                    )
                if signed_pol_pdf:
                    st.markdown(f"🔗 [Open Time-Limited Signed GCS URL (`OBJ.GET_ACCESS_URL`)]({signed_pol_pdf})")

        with st.expander("💡 How Google Cloud Lakehouse Does This in SQL (`OBJ.MAKE_REF` + `ML.GENERATE_TEXT`)"):
            st.code(
                f"""-- Pass raw PDF files on GCS directly to Gemini 3.8 Flash via ObjectRef (Zero external OCR!)
SELECT
    claim_id,
    ml_generate_text_llm_result AS pdf_clinical_contract_audit
FROM ML.GENERATE_TEXT(
    MODEL `{PROJECT_ID}.{DATASET_ID}.model_gemini_auditor`, -- gemini-3.8-flash
    (
        SELECT
            c.claim_id,
            STRUCT(
                CONCAT('Audit the Policy PDF and Attending Physician Statement PDF for ', c.claim_id) AS prompt,
                [
                    OBJ.FETCH_METADATA(OBJ.MAKE_REF(c.policy_pdf_gcs_uri, '{PROJECT_ID}.{REGION}.lakehouse-omniclaim-conn')),
                    OBJ.FETCH_METADATA(OBJ.MAKE_REF(c.aps_pdf_gcs_uri, '{PROJECT_ID}.{REGION}.lakehouse-omniclaim-conn'))
                ] AS object_ref_list
            ) AS prompt
        FROM `{PROJECT_ID}.{DATASET_ID}.iceberg_claims_policies` c
    ),
    STRUCT(0.0 AS temperature, 1024 AS max_output_tokens, TRUE AS flatten_json_output)
);""",
                language="sql",
            )

    # ------------------------------------------------------------------------
    # TAB 4: BENEFICIARY FNOL AUDIO & FINAL VERDICT
    # ------------------------------------------------------------------------
    with tab4:
        st.markdown("#### 🎧 Capability 4: Native Audio Cross-Examination & Final Adjudication (`ObjectRef` + `Gemini 3.8 Flash`)")
        st.caption(
            "Finally, BigQuery passes the beneficiary's recorded First Notice of Loss (`.wav`, synthesized with **Google Cloud `Chirp 3: HD`**) "
            "as an `ObjectRef` into **`gemini-3.8-flash`**, combining all 4 modalities to render a structured adjudication verdict."
        )

        col_call, col_verdict = st.columns([1.05, 1.15], gap="large")

        with col_call:
            st.markdown(f"##### 🔊 Listen to Beneficiary Call (`{claim['claim_id']}_fnol.wav`)")
            st.caption(f"Caller: **{claim['beneficiary_name']}** ({claim['beneficiary_relationship']})")

            local_audio_wav = DATA_DIR / "unstructured_evidence" / "beneficiary_fnol_audio" / f"{claim['claim_id']}_fnol.wav"
            signed_fnol_audio = extract_signed_read_url(claim.get("signed_fnol_audio_url"))
            if local_audio_wav.exists():
                st.audio(local_audio_wav.read_bytes(), format="audio/wav")
            elif signed_fnol_audio:
                st.audio(signed_fnol_audio, format="audio/wav")

            st.markdown("**Reference Call Transcript:**")
            st.info(f'🗣️ *"{claim.get("fnol_transcript_reference", "")}"*')

        with col_verdict:
            st.markdown("##### ⚖️ Gemini 3.8 Flash Cross-Modal Adjudication Synthesis")
            st.markdown(
                f"""
                - **Recommended Action:** `{action_code}`
                - **Approved Payout:** **`${approved_amt:,.0f}`** *(of `${claimed_amt:,.0f}` claimed)*
                - **Audio Summary:** {verdict.get('audio_transcript_summary', '')}
                - **Cross-Modal Discrepancies Detected:** {verdict.get('audio_discrepancies_detected', 'None')}
                - **Policy Clause / Exclusion Triggered:** `{verdict.get('rider_exclusion_triggered', 'NONE')}`
                - **Executive Rationale:** {verdict.get('executive_rationale', '')}
                """
            )
            st.success(f"🎓 **Key Lakehouse Takeaway:** {story['key_lesson']}")

            with st.expander("🔍 View Raw JSON Emitted by `ML.GENERATE_TEXT` in BigQuery"):
                st.json(verdict)

    # ------------------------------------------------------------------------
    # TAB 5: ALL 5 CASES & ARCHITECTURE OVERVIEW
    # ------------------------------------------------------------------------
    with tab5:
        st.markdown("#### 📊 All 5 Hero Life Insurance Scenarios in One View")
        summary_rows = []
        for r in records:
            pv = r.get("parsed_verdict") or {}
            summary_rows.append(
                {
                    "Claim ID": r["claim_id"],
                    "Insured": r["insured_name"],
                    "Policy / Rider": r["product_type"],
                    "Contestability": f"{r['months_since_issue']}m ({'Contestable <24m' if r['is_within_contestability'] else 'Incontestable'})",
                    "Claimed ($)": f"${float(r['claimed_benefit_amount']):,.0f}",
                    "Top-1 X-Ray Match": f"{r.get('matched_ref_image_id')} ({float(r.get('cosine_similarity', 0)):.4f})",
                    "Forensic Flag": r.get("radiograph_forensic_flag"),
                    "Gemini 3.8 Verdict": pv.get("recommended_action", r.get("expected_action")),
                    "Approved ($)": f"${float(pv.get('approved_payout_amount', r.get('expected_payout', 0))):,.0f}",
                }
            )
        st.dataframe(pd.DataFrame(summary_rows), width="stretch", hide_index=True)

        with st.expander("📜 View Full BigQuery Lakehouse SQL (`sql/01_lakehouse_ddl.sql` & `sql/02_multimodal_query.sql`)"):
            s1, s2 = st.columns(2)
            with s1:
                st.markdown("**`sql/01_lakehouse_ddl.sql`**")
                st.code(Path("sql/01_lakehouse_ddl.sql").read_text(encoding="utf-8"), language="sql")
            with s2:
                st.markdown("**`sql/02_multimodal_query.sql`**")
                st.code(Path("sql/02_multimodal_query.sql").read_text(encoding="utf-8"), language="sql")


if __name__ == "__main__":
    main()
