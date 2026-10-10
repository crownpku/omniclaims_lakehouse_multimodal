#!/usr/bin/env python3
"""
OmniClaim Life Lakehouse — Multi-Modal Life Insurance Dataset Builder (`generate_data.py`)

Fetches REAL open-access de-identified clinical radiographs and peer-reviewed case report findings
from the US National Institutes of Health (NIH) National Library of Medicine Open-i / PubMed Central
(`https://openi.nlm.nih.gov`) and synthesizes realistic multi-modal insurance claim evidence in `data/`:
  1. Structured Policy & Claim Core (CSV & Parquet for BigQuery Managed Apache Iceberg)
  2. Real Clinical Thoracic & Trauma Radiographs from NIH Open-i / PubMed Central / MedPix (PNG)
  3. Master Policy Contracts & NIH PubMed-Grounded Attending Physician Statements (PDF)
  4. Natural Conversational Beneficiary FNOL Call Recordings via Google Cloud Chirp 3: HD Audio (WAV)
"""

from __future__ import annotations

import hashlib
import io
import os
import subprocess
import urllib.request
import wave
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fpdf import FPDF
from PIL import Image, ImageEnhance

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT_ID", "your-gcp-project-id")
BUCKET_NAME = os.getenv("GCS_BUCKET_NAME", f"{PROJECT_ID}-bucket")
DATA_DIR = Path("data")


def get_gcp_credentials():
    """Returns Google Cloud credentials using ADC or gcloud CLI access token fallback."""
    import google.auth
    from google.oauth2 import credentials as oauth2_credentials

    try:
        creds, _ = google.auth.default(quota_project_id=PROJECT_ID)
        return creds
    except Exception:
        token = subprocess.check_output(["gcloud", "auth", "print-access-token", "--quiet"], text=True).strip()
        return oauth2_credentials.Credentials(token=token, quota_project_id=PROJECT_ID)


# ============================================================================
# 1. REAL NIH NLM OPEN-I / PUBMED CENTRAL CLINICAL RADIOGRAPHS & CASE REPORTS
# ============================================================================

REF_MEDICAL_ATLAS: list[dict[str, Any]] = [
    {
        "ref_image_id": "REF_NORM_001",
        "reference_category": "VERIFIED_PATHOLOGY_ATLAS",
        "pathology_label": "NORMAL_THORACIC_RADIOGRAPH",
        "prior_claim_id": None,
        "prior_policyholder_id": None,
        "nih_source_id": "PMC12127709 (PMID: 40458334)",
        "nih_image_url": "https://openi.nlm.nih.gov/imgs/512/266/12127709/PMC12127709_cureus-0017-00000083326-i02.png",
        "clinical_notes": (
            "NIH PubMed Central PMC12127709 (SUNY Downstate Medical Center): Frontal chest radiograph "
            "showing normal cardiac silhouette, clear bilateral lung fields, and no consolidation or atelectasis."
        ),
    },
    {
        "ref_image_id": "REF_PNEUMONIA_001",
        "reference_category": "VERIFIED_PATHOLOGY_ATLAS",
        "pathology_label": "ACUTE_LOBAR_PNEUMONIA_RLL",
        "prior_claim_id": None,
        "prior_policyholder_id": None,
        "nih_source_id": "PMC10405870 (PMID: 37554481)",
        "nih_image_url": "https://openi.nlm.nih.gov/imgs/512/321/10405870/PMC10405870_3940_Fig1.png",
        "clinical_notes": (
            "NIH PubMed Central PMC10405870 (Unity Hospital / Rochester Regional Health, NY): Frontal chest "
            "radiograph demonstrating dense right lower lobe (RLL) airspace consolidation consistent with acute lobar pneumonia."
        ),
    },
    {
        "ref_image_id": "REF_CARDIOMEGALY_001",
        "reference_category": "VERIFIED_PATHOLOGY_ATLAS",
        "pathology_label": "SEVERE_CARDIOMEGALY_WITH_PACEMAKER",
        "prior_claim_id": None,
        "prior_policyholder_id": None,
        "nih_source_id": "PMC7033380 (PMID: 32095464)",
        "nih_image_url": "https://openi.nlm.nih.gov/imgs/512/241/7033380/PMC7033380_tcp-25-138-g003.png",
        "clinical_notes": (
            "NIH PubMed Central PMC7033380 (Chosun University Hospital Division of Cardiology): Chest radiograph "
            "demonstrating marked Stage III cardiomegaly with implanted permanent pacemaker generator and transvenous pacing leads."
        ),
    },
    {
        "ref_image_id": "REF_TRAUMA_001",
        "reference_category": "VERIFIED_PATHOLOGY_ATLAS",
        "pathology_label": "HIGH_ENERGY_CLAVICULAR_THORACIC_FRACTURE",
        "prior_claim_id": None,
        "prior_policyholder_id": None,
        "nih_source_id": "PMC5702695 (PMID: 29181345)",
        "nih_image_url": "https://openi.nlm.nih.gov/imgs/512/74/5702695/PMC5702695_JOCR-7-17-g001.png",
        "clinical_notes": (
            "NIH PubMed Central PMC5702695 (Cooper Medical School of Rowan University Trauma Center): Anteroposterior "
            "radiograph depicting severely displaced right clavicular fracture with inferior fragment displacement and partial pneumothorax."
        ),
    },
    {
        "ref_image_id": "REF_PULM_MASS_001",
        "reference_category": "VERIFIED_PATHOLOGY_ATLAS",
        "pathology_label": "MALIGNANT_RUL_PULMONARY_NEOPLASM_28MM",
        "prior_claim_id": None,
        "prior_policyholder_id": None,
        "nih_source_id": "PMC7529567 (PMID: 32869499)",
        "nih_image_url": "https://openi.nlm.nih.gov/imgs/512/391/7529567/PMC7529567_TCA-11-3038-g001.png",
        "clinical_notes": (
            "NIH PubMed Central PMC7529567 (Thoracic Cancer / Mie Chuo Medical Center): Posteroanterior chest radiograph "
            "showing a 30 mm right upper lobe (RUL) pulmonary mass confirmed as invasive lung adenocarcinoma by transbronchial biopsy."
        ),
    },
    {
        "ref_image_id": "PRIOR_CLM_8821_FRAUD",
        "reference_category": "HISTORICAL_PRIOR_CLAIM",
        "pathology_label": "PRIOR_CLAIM_LUL_MASS_ARTIFACT_8821",
        "prior_claim_id": "PRIOR-CLM-8821",
        "prior_policyholder_id": "PH-99102 (Arthur Pendelton)",
        "nih_source_id": "NIH MedPix MPX1219 (synpic14404)",
        "nih_image_url": "https://openi.nlm.nih.gov/imgs/512/16/1219/MPX1219_synpic14404.png",
        "clinical_notes": (
            "NIH NLM MedPix MPX1219 (Uniformed Services University): Historical radiograph submitted 9 months ago under "
            "Policy POL-UL-99102 (Claim PRIOR-CLM-8821) showing a large destructive left upper lobe apical mass."
        ),
    },
]


MASTER_POLICIES: dict[str, dict[str, str]] = {
    "POL-TERM-STD-2025": {
        "title": "OMNILIFE 20-YEAR LEVEL TERM LIFE INSURANCE POLICY",
        "subtitle": "Form POL-TERM-STD-2025 | Standard & Preferred Non-Smoker Contract",
        "body": (
            "SECTION 1.1 -- DEATH BENEFIT COVERAGE\n"
            "OmniLife Insurance Company agrees to pay the Base Sum Assured to the designated Beneficiary upon receipt "
            "of due proof of the Insured's death occurring while this Policy is in force.\n\n"
            "SECTION 1.4 -- 24-MONTH INCONTESTABILITY CLAUSE\n"
            "Except for non-payment of premiums, this Policy shall be incontestable after it has been in force during "
            "the lifetime of the Insured for a period of twenty-four (24) months from its Issue Date. Once the 24-month "
            "Contestability Window has elapsed, natural death claims substantiated by an Attending Physician Statement (APS) "
            "and consistent diagnostic imaging qualify for Fast-Track Payout.\n\n"
            "SECTION 3.1 -- GENERAL EXCLUSIONS\n"
            "Death resulting from suicide within twenty-four (24) months of the Issue Date limits liability to a refund of premiums paid."
        ),
    },
    "POL-WL-PREF-2025": {
        "title": "OMNILIFE LEGACY WHOLE LIFE PREFERRED PLUS CONTRACT",
        "subtitle": "Form POL-WL-PREF-2025 | Preferred Plus Non-Smoker Tier",
        "body": (
            "SECTION 1.2 -- PREFERRED PLUS UNDERWRITING REPRESENTATIONS\n"
            "Issuance under the Preferred Plus Non-Smoker rate class relies materially upon the Applicant's signed health "
            "declarations confirming zero history of cardiomyopathy, congestive heart failure, coronary artery disease, "
            "cardiac arrhythmia, or implanted cardiac pacemakers/defibrillators prior to the Application Date.\n\n"
            "SECTION 1.4 -- 24-MONTH CONTESTABILITY & MATERIAL MISREPRESENTATION RESCISSION\n"
            "If a claim event occurs within twenty-four (24) months of the Policy Issue Date, OmniLife Special Investigations "
            "Unit (SIU) shall audit all pre-existing clinical records, Attending Physician Statements (APS), and thoracic "
            "radiographs. Any undisclosed chronic cardiac pathology or implanted cardiac device predating the Application Date "
            "constitutes Material Misrepresentation, voiding coverage and triggering an immediate SIU_RESCISSION_HOLD ($0 payout)."
        ),
    },
    "POL-ADD-EXEC-2025": {
        "title": "OMNILIFE EXECUTIVE TERM + AD&D DOUBLE-INDEMNITY RIDER",
        "subtitle": "Form POL-ADD-EXEC-2025 | Base Life + 100% Accidental Death & Dismemberment Rider",
        "body": (
            "SECTION 1.4 -- BASE LIFE INCONTESTABILITY\n"
            "After twenty-four (24) months from the Issue Date, the Base Life Sum Assured ($1,000,000) is incontestable "
            "and payable upon proof of death from any cause.\n\n"
            "SECTION 4.1 -- ACCIDENTAL DEATH & DISMEMBERMENT (AD&D) RIDER BENEFIT\n"
            "An additional 100% AD&D Rider Benefit ($1,000,000 Double Indemnity) is payable ONLY if death results directly "
            "and independently from external accidental bodily injury and is NOT excluded under Section 4.2.\n\n"
            "SECTION 4.2 -- AD&D RIDER HAZARDOUS ACTIVITY & INTOXICATION EXCLUSIONS\n"
            "No AD&D Rider Benefit shall be paid if the accidental injury or death occurs while the Insured is participating in:\n"
            "  (a) Free-solo climbing, unroped technical mountaineering, or alpine ice climbing above 4,000 meters elevation;\n"
            "  (b) Piloting or crew operation of any non-commercial private or experimental aircraft; or\n"
            "  (c) Operating any vehicle or apparatus with a Blood Alcohol Concentration (BAC) of 0.08% or higher.\n"
            "When Section 4.2 is triggered on an incontestable policy, Adjudication Action MUST be PARTIAL_APPROVE_BASE_DENY_ADD_RIDER "
            "(approving the Base Life Sum Assured while denying the AD&D Rider)."
        ),
    },
    "POL-UL-CI-2025": {
        "title": "OMNILIFE UNIVERSAL LIFE + CRITICAL ILLNESS LIVING BENEFIT RIDER",
        "subtitle": "Form POL-UL-CI-2025 | Accelerated Living Benefit Endorsement",
        "body": (
            "SECTION 2.1 -- QUALIFYING CRITICAL ILLNESS (CI) SEVERITY THRESHOLDS\n"
            "The Critical Illness Living Benefit Rider pays a lump-sum accelerated benefit upon definitive diagnosis of "
            "Invasive Malignant Pulmonary Neoplasm, defined strictly as a primary lung tumor measuring >= 2.0 cm on thoracic "
            "radiograph/CT accompanied by histopathological biopsy confirmation (ICD-10 C34.x) after the 90-day waiting period.\n\n"
            "SECTION 5.3 -- BIOMETRIC & DIAGNOSTIC IMAGE ANTI-FRAUD CLAUSE\n"
            "All diagnostic radiographs submitted for Living Benefit claims are indexed via multimodal vector search against "
            "OmniLife's historical claims atlas. Submission of a duplicate, recycled, or digitally altered radiograph previously "
            "submitted under another policy or claimant constitutes Syndicated Insurance Fraud, requiring immediate "
            "SIU_FRAUD_RING_REJECT ($0 payout) and referral to the National Insurance Crime Bureau (NICB)."
        ),
    },
}


HERO_CLAIMS: list[dict[str, Any]] = [
    {
        "claim_id": "CLM-LIFE-1001",
        "policy_id": "POL-10001",
        "policyholder_id": "PH-5001",
        "policy_code": "POL-TERM-STD-2025",
        "product_type": "TERM_20_LIFE",
        "insured_name": "Eleanor Vance",
        "dob": "1968-04-14",
        "gender": "F",
        "uw_health_class": "PREFERRED_NON_SMOKER",
        "declared_smoker": False,
        "declared_medical_history": "No chronic cardiac, pulmonary, or neoplastic disease. Routine annual physicals normal.",
        "issue_date": "2020-06-15",
        "event_date": "2026-02-18",
        "months_since_issue": 68,
        "is_within_contestability": False,
        "claim_type": "DEATH_NATURAL",
        "base_sum_assured": 500000.0,
        "has_add_rider": False,
        "add_rider_sum_assured": 0.0,
        "has_ci_rider": False,
        "ci_rider_sum_assured": 0.0,
        "claimed_benefit_amount": 500000.0,
        "beneficiary_name": "Clara Vance",
        "beneficiary_relationship": "Daughter",
        "reported_cause_of_event": "Acute bacterial lobar pneumonia (Right Lower Lobe) with septic respiratory failure",
        "nih_source_id": "PMC10405870 (PMID: 37554481)",
        "nih_image_url": "https://openi.nlm.nih.gov/imgs/512/321/10405870/PMC10405870_3940_Fig1.png",
        "image_transform": "clinical_acquisition_crop",
        "aps_title": "ATTENDING PHYSICIAN STATEMENT (APS) -- UNITY HOSPITAL ROCHESTER REGIONAL HEALTH",
        "aps_physician": "Dr. Sarah Jenkins, MD (Department of Pulmonary & Critical Care Medicine, License #NY-88412)",
        "aps_body": (
            "Patient Name: Eleanor Vance | Claim ID: CLM-LIFE-1001 | NIH Clinical Case Reference: PMC10405870 (PMID: 37554481)\n"
            "Primary Diagnosis: ICD-10 J18.1 -- Acute Lobar Pneumonia, Right Lower Lobe, complicated by acute respiratory failure.\n"
            "Date of First Symptom Onset: February 12, 2026 (68 months after policy issuance).\n"
            "Clinical Presentation & Vitals: Patient presented febrile (38.1 C), tachycardic (110 bpm), tachypneic (RR 30), "
            "with oxygen saturation 92% on room air and coarse rhonchi over the right lower lung zone.\n"
            "Radiograph Findings: Posteroanterior chest radiograph (PMC10405870 Fig. 1) demonstrates dense right lower lobe "
            "consolidation concerning for acute bacterial lobar pneumonia with normal cardiomediastinal silhouette.\n"
            "Prior Medical History Audit: Review of 10-year primary care records confirms ZERO prior pulmonary or cardiac disease prior to February 2026."
        ),
        "fnol_transcript": (
            "Hi, um, my name is Clara Vance. I'm calling to file a death benefit claim on policy P-O-L-10001 for my mother, Eleanor Vance. "
            "Mom was completely healthy and active until she came down with a sudden, severe chest infection on February twelfth. "
            "We rushed her to Unity Hospital where the emergency room chest X-ray showed acute right lower lobe pneumonia, "
            "and sadly she passed away four days later. Her policy has been in force since June of twenty twenty."
        ),
        "tts_voice": "en-US-Chirp3-HD-Aoede",
        "expected_action": "AUTO_APPROVE_PAYOUT",
        "expected_payout": 500000.0,
    },
    {
        "claim_id": "CLM-LIFE-1002",
        "policy_id": "POL-10002",
        "policyholder_id": "PH-5002",
        "policy_code": "POL-WL-PREF-2025",
        "product_type": "WHOLE_LIFE_PREFERRED_PLUS",
        "insured_name": "Robert Sterling",
        "dob": "1964-09-22",
        "gender": "M",
        "uw_health_class": "PREFERRED_PLUS_NON_SMOKER",
        "declared_smoker": False,
        "declared_medical_history": "Signed Application Declaration (July 2025): Zero history of cardiac disease, cardiomyopathy, heart failure, or implanted pacemaker.",
        "issue_date": "2025-07-10",
        "event_date": "2026-02-14",
        "months_since_issue": 7,
        "is_within_contestability": True,
        "claim_type": "DEATH_NATURAL_CONTESTABLE",
        "base_sum_assured": 1500000.0,
        "has_add_rider": False,
        "add_rider_sum_assured": 0.0,
        "has_ci_rider": False,
        "ci_rider_sum_assured": 0.0,
        "claimed_benefit_amount": 1500000.0,
        "beneficiary_name": "Victoria Sterling",
        "beneficiary_relationship": "Spouse",
        "reported_cause_of_event": "Sudden unexpected cardiac arrest",
        "nih_source_id": "PMC7033380 (PMID: 32095464)",
        "nih_image_url": "https://openi.nlm.nih.gov/imgs/512/241/7033380/PMC7033380_tcp-25-138-g003.png",
        "image_transform": "clinical_acquisition_crop",
        "aps_title": "ATTENDING PHYSICIAN STATEMENT (APS) -- DIVISION OF CARDIOLOGY, UNIVERSITY HOSPITAL",
        "aps_physician": "Dr. Alan Grant, MD, FACC (Chief of Electrophysiology & Cardiology, License #NY-44910)",
        "aps_body": (
            "Patient Name: Robert Sterling | Claim ID: CLM-LIFE-1002 | NIH Clinical Case Reference: PMC7033380 (PMID: 32095464)\n"
            "Primary Diagnosis: ICD-10 I42.0 / I48.91 -- Chronic Dilated Cardiomyopathy, Atrial Fibrillation, and End-Stage Heart Failure.\n"
            "Date of Initial Diagnosis: March 18, 2021 (More than 4 years PRIOR to policy application in July 2025).\n"
            "Surgical & Implanted Device History: Permanent dual-chamber transvenous cardiac pacemaker implanted in November 2022 "
            "(pacing threshold 0.5V -> 1.5V at 40ms; sensing threshold 2.8mV; lead impedance 410 ohms).\n"
            "Radiograph & Lab Findings: Emergency chest radiograph (PMC7033380 Fig. 3) demonstrates severe cardiomegaly with "
            "bilateral pleural effusions and clearly visible permanent pacemaker generator and right atrial/ventricular pacing leads in place since 2022. "
            "Labs: WBC 14,200/mm3, Hb 11.7 g/dL, D-dimer 6.04 mg/mL, CRP 10.3 mg/dL, Creatinine 1.9 mg/dL."
        ),
        "fnol_transcript": (
            "Hello, this is Victoria Sterling calling to file a life insurance claim on my husband, Robert Sterling, under policy P-O-L-10002. "
            "Robert collapsed at home on February fourteenth with severe shortness of breath. We immediately called his long-time cardiologist, "
            "Doctor Alan Grant, who has been treating Robert's enlarged heart and checking his implanted pacemaker for the past four years, "
            "and had him rushed to the emergency department."
        ),
        "tts_voice": "en-US-Chirp3-HD-Kore",
        "expected_action": "SIU_RESCISSION_HOLD",
        "expected_payout": 0.0,
    },
    {
        "claim_id": "CLM-ADD-1003",
        "policy_id": "POL-10003",
        "policyholder_id": "PH-5003",
        "policy_code": "POL-ADD-EXEC-2025",
        "product_type": "EXEC_TERM_WITH_ADD_RIDER",
        "insured_name": "Marcus Thorne",
        "dob": "1979-11-03",
        "gender": "M",
        "uw_health_class": "STANDARD_NON_SMOKER",
        "declared_smoker": False,
        "declared_medical_history": "Healthy recreational runner; no chronic medical conditions.",
        "issue_date": "2022-12-01",
        "event_date": "2026-02-05",
        "months_since_issue": 38,
        "is_within_contestability": False,
        "claim_type": "DEATH_ACCIDENTAL_ADD",
        "base_sum_assured": 1000000.0,
        "has_add_rider": True,
        "add_rider_sum_assured": 1000000.0,
        "has_ci_rider": False,
        "ci_rider_sum_assured": 0.0,
        "claimed_benefit_amount": 2000000.0,
        "beneficiary_name": "David Thorne",
        "beneficiary_relationship": "Brother",
        "reported_cause_of_event": "Claim form states: Accidental slip on recreational walking trail",
        "nih_source_id": "PMC5702695 (PMID: 29181345)",
        "nih_image_url": "https://openi.nlm.nih.gov/imgs/512/74/5702695/PMC5702695_JOCR-7-17-g001.png",
        "image_transform": "clinical_acquisition_crop",
        "aps_title": "ATTENDING PHYSICIAN & TRAUMA REPORT -- REGIONAL TRAUMA CENTER & MEDICAL EXAMINER",
        "aps_physician": "Dr. Raymond Chen, MD (Orthopaedic & Forensic Trauma Surgery, License #WA-33921)",
        "aps_body": (
            "Decedent: Marcus Thorne | Claim ID: CLM-ADD-1003 | NIH Clinical Case Reference: PMC5702695 (PMID: 29181345)\n"
            "Diagnosis & Trauma Mechanism: ICD-10 S42.001A / W15 -- High-energy vertical fall trauma resulting in gross inferior "
            "displacement of a comminuted right clavicular fracture penetrating the pleural cavity with associated traumatic pneumothorax.\n"
            "Radiograph Findings: Anteroposterior trauma bay chest radiograph (PMC5702695 Fig. 1) confirms gross displacement of the "
            "right proximal/mid-third clavicle fracture with intrapleural penetration and partial right lung collapse.\n"
            "Incident Investigation Notes: National Park Mountain Rescue recovered decedent at 4,250 meters elevation on "
            "Liberty Ridge, Mount Rainier. Decedent was performing unroped free-solo technical alpine ice climbing without belay "
            "protection when an ice shelf sheared. Toxicology negative (BAC 0.00%)."
        ),
        "fnol_transcript": (
            "Hi, I'm David Thorne, the beneficiary for my brother Marcus Thorne on policy P-O-L-10003. "
            "We're submitting a claim for both the one million dollar base life benefit and the one million dollar accidental death double indemnity rider. "
            "Look, even though the paper form mentioned a walking trail, I want to be completely honest with you: Marcus was doing an unroped free-solo ice climb "
            "at four thousand two hundred and fifty meters on Mount Rainier when the ice shelf gave way and he fell."
        ),
        "tts_voice": "en-US-Chirp3-HD-Fenrir",
        "expected_action": "PARTIAL_APPROVE_BASE_DENY_ADD_RIDER",
        "expected_payout": 1000000.0,
    },
    {
        "claim_id": "CLM-CI-1004",
        "policy_id": "POL-10004",
        "policyholder_id": "PH-5004",
        "policy_code": "POL-UL-CI-2025",
        "product_type": "UNIVERSAL_LIFE_CI_RIDER",
        "insured_name": "Victor Krum",
        "dob": "1982-02-19",
        "gender": "M",
        "uw_health_class": "STANDARD_NON_SMOKER",
        "declared_smoker": False,
        "declared_medical_history": "No prior medical history reported.",
        "issue_date": "2024-12-10",
        "event_date": "2026-02-15",
        "months_since_issue": 14,
        "is_within_contestability": True,
        "claim_type": "CRITICAL_ILLNESS_LIVING_BENEFIT",
        "base_sum_assured": 500000.0,
        "has_add_rider": False,
        "add_rider_sum_assured": 0.0,
        "has_ci_rider": True,
        "ci_rider_sum_assured": 350000.0,
        "claimed_benefit_amount": 350000.0,
        "beneficiary_name": "Victor Krum (Living Benefit Claimant)",
        "beneficiary_relationship": "Self (Insured)",
        "reported_cause_of_event": "Accelerated Critical Illness claim for newly diagnosed pulmonary mass",
        "nih_source_id": "NIH MedPix MPX1219 (synpic14404 - Perturbed Recycled Copy)",
        "nih_image_url": "https://openi.nlm.nih.gov/imgs/512/16/1219/MPX1219_synpic14404.png",
        "image_transform": "perturbed_duplicate",
        "aps_title": "ATTENDING PHYSICIAN STATEMENT (APS) -- OFFSHORE RAPID DIAGNOSTICS CLINIC",
        "aps_physician": "Dr. V. Petrov (Unverified Clinic Registration #UNV-0091)",
        "aps_body": (
            "Patient Name: Victor Krum | Claim ID: CLM-CI-1004\n"
            "Reported Diagnosis: Pulmonary Mass, Left Upper Lobe (Requesting immediate $350,000 wire transfer).\n"
            "Clinical Documentation Audit: Single-page form lacking histopathological biopsy accession number, oncology staging, "
            "or hospital admission record. Submitted radiograph is a recycled digital copy of NIH MedPix MPX1219 (previously submitted "
            "under Claim PRIOR-CLM-8821 for Arthur Pendelton) with a 1-intensity corner pixel modification to evade MD5/SHA256 hash checks."
        ),
        "fnol_transcript": (
            "Yeah, this is Victor Krum calling about my policy P-O-L-10004. I uploaded the chest X-ray showing a lung mass and I need the "
            "three hundred and fifty thousand dollar critical illness payout wired to my account immediately. "
            "No, I don't want to give you the name of the hospital or the biopsy lab, just look at the X-ray I uploaded and process the wire today."
        ),
        "tts_voice": "en-US-Chirp3-HD-Charon",
        "expected_action": "SIU_FRAUD_RING_REJECT",
        "expected_payout": 0.0,
    },
    {
        "claim_id": "CLM-CI-1005",
        "policy_id": "POL-10005",
        "policyholder_id": "PH-5005",
        "policy_code": "POL-UL-CI-2025",
        "product_type": "UNIVERSAL_LIFE_CI_RIDER",
        "insured_name": "Diana Chen",
        "dob": "1974-07-08",
        "gender": "F",
        "uw_health_class": "PREFERRED_NON_SMOKER",
        "declared_smoker": False,
        "declared_medical_history": "Non-smoker, normal baseline chest radiograph and physical exam at issue in September 2023.",
        "issue_date": "2023-09-10",
        "event_date": "2026-02-16",
        "months_since_issue": 29,
        "is_within_contestability": False,
        "claim_type": "CRITICAL_ILLNESS_LIVING_BENEFIT",
        "base_sum_assured": 500000.0,
        "has_add_rider": False,
        "add_rider_sum_assured": 0.0,
        "has_ci_rider": True,
        "ci_rider_sum_assured": 250000.0,
        "claimed_benefit_amount": 250000.0,
        "beneficiary_name": "Diana Chen (Living Benefit Claimant)",
        "beneficiary_relationship": "Self (Insured)",
        "reported_cause_of_event": "Invasive Right Upper Lobe Lung Adenocarcinoma (2.8cm Stage IIA) — Living Benefit Claim",
        "nih_source_id": "PMC7529567 (PMID: 32869499)",
        "nih_image_url": "https://openi.nlm.nih.gov/imgs/512/391/7529567/PMC7529567_TCA-11-3038-g001.png",
        "image_transform": "clinical_acquisition_crop",
        "aps_title": "ATTENDING PHYSICIAN STATEMENT (APS) -- DEPARTMENT OF THORACIC ONCOLOGY",
        "aps_physician": "Dr. Emily Nakamura, MD, PhD (Thoracic Oncology, License #CA-77204)",
        "aps_body": (
            "Patient Name: Diana Chen | Claim ID: CLM-CI-1005 | NIH Clinical Case Reference: PMC7529567 (PMID: 32869499)\n"
            "Primary Diagnosis: ICD-10 C34.11 -- Invasive Malignant Neoplasm of Right Upper Lobe (RUL) Bronchus and Lung, Stage IIA.\n"
            "Date of Diagnosis: February 10, 2026 (29 months after policy issuance; well past 90-day CI waiting period and 24-month contestability).\n"
            "Radiograph & Histopathology Findings: Posteroanterior chest radiograph (PMC7529567 Fig. 1) and thoracic CT demonstrate a "
            "well-defined 2.8-3.0 cm mass in the right upper lung field (>= 2.0 cm Policy Section 2.1 threshold) with FDG-PET SUVmax 6.16. "
            "Transbronchial biopsy (Pathology Accession #STN-2026-8841) confirms invasive acinar pulmonary adenocarcinoma."
        ),
        "fnol_transcript": (
            "Hello, this is Diana Chen calling regarding my universal life policy P-O-L-10005 to request the two hundred and fifty thousand dollar "
            "critical illness living benefit. Last week at Thoracic Oncology, Doctor Emily Nakamura confirmed a two point eight centimeter "
            "stage two-A lung adenocarcinoma in my right upper lobe via biopsy. We have uploaded the full pathology report and my chest radiograph."
        ),
        "tts_voice": "en-US-Chirp3-HD-Leda",
        "expected_action": "APPROVE_LIVING_BENEFIT",
        "expected_payout": 250000.0,
    },
]


# ============================================================================
# 2. REAL NIH OPEN-I CLINICAL RADIOGRAPH FETCHER & PROCESSOR
# ============================================================================

_IMAGE_CACHE: dict[str, bytes] = {}


def fetch_real_nih_radiograph(
    output_path: Path,
    nih_image_url: str,
    transform_mode: str = "reference",
) -> str:
    """
    Downloads a real open-access clinical radiograph PNG from the US National Library of Medicine
    (NIH Open-i / PubMed Central / MedPix) without adding any fake cartoon drawings or synthetic banners.
    - `reference`: saves the authentic NIH radiograph as a single frontal radiograph panel.
    - `clinical_acquisition_crop`: applies a subtle clinical PACS border crop & windowing calibration,
      preserving high visual/semantic similarity in Vertex AI `gemini-embedding-2-preview`.
    - `perturbed_duplicate`: modifies a single corner pixel by +1 intensity so the SHA-256 hash
      is completely different while the image is a 99.9%+ duplicate of `PRIOR_CLM_8821_FRAUD`.
    """
    if nih_image_url not in _IMAGE_CACHE:
        req = urllib.request.Request(
            nih_image_url,
            headers={
                "User-Agent": "Mozilla/5.0 (OmniClaim-Medical-Lakehouse-Research/1.0; +https://openi.nlm.nih.gov)"
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            _IMAGE_CACHE[nih_image_url] = resp.read()

    raw_bytes = _IMAGE_CACHE[nih_image_url]
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    w, h = img.size

    # PMC7033380_tcp-25-138-g003.png contains two stacked radiographs of the same pacemaker patient (A: emergency admission, B: 3 months prior).
    # Extract the clean single-panel frontal radiograph (trimming the bottom-left A/B letter).
    if "PMC7033380_tcp-25-138-g003.png" in nih_image_url:
        if transform_mode == "reference":
            # Bottom panel (3 months prior baseline with pacemaker & cardiomegaly), trimming left edge 'B'
            img = img.crop((48, h // 2 + 16, w - 8, h - 6))
        else:
            # Top panel (emergency admission with worsening cardiomegaly & pacemaker), trimming left edge 'A'
            img = img.crop((48, 6, w - 8, h // 2 - 22))
        w, h = img.size
    elif "PMC7529567_TCA-11-3038-g001.png" in nih_image_url:
        # Left panel is the posteroanterior (PA) frontal chest radiograph showing the 30mm RUL adenocarcinoma mass
        img = img.crop((4, 4, int(w * 0.535), h - 4))
        w, h = img.size

    if transform_mode == "clinical_acquisition_crop":
        # Subtle 4% clinical PACS margin trim + slight contrast/brightness windowing
        margin_x = max(4, int(w * 0.04))
        margin_y = max(4, int(h * 0.04))
        img = img.crop((margin_x, margin_y, w - margin_x, h - margin_y)).resize((w, h), Image.Resampling.BICUBIC)
        img = ImageEnhance.Contrast(img).enhance(1.06)
        img = ImageEnhance.Brightness(img).enhance(0.96)
    elif transform_mode == "perturbed_duplicate":
        # Modify bottom-right corner pixel by +1 intensity so SHA256/MD5 hash is completely different
        # while visual & vector embedding similarity remains >0.999!
        arr = np.array(img)
        arr[-1, -1, 0] = np.uint8((int(arr[-1, -1, 0]) + 1) % 256)
        img = Image.fromarray(arr)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path, format="PNG")
    return hashlib.sha256(output_path.read_bytes()).hexdigest()


# ============================================================================
# 3. POLICY CONTRACT & ATTENDING PHYSICIAN STATEMENT PDF GENERATOR (FPDF2)
# ============================================================================

def _clean_latin1(text: str) -> str:
    return (
        text.replace("\u2014", " -- ")
        .replace("\u2013", "-")
        .replace("\u2018", "'")
        .replace("\u2019", "'")
        .replace("\u201c", '"')
        .replace("\u201d", '"')
    )


def render_pdf_document(output_path: Path, title: str, subtitle: str, body_text: str, footer_note: str) -> None:
    title = _clean_latin1(title)
    subtitle = _clean_latin1(subtitle)
    body_text = _clean_latin1(body_text)
    footer_note = _clean_latin1(footer_note)

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # Header Banner
    pdf.set_fill_color(19, 41, 75)
    pdf.rect(0, 0, 210, 28, style="F")
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 12.5)
    pdf.set_xy(10, 7)
    pdf.cell(190, 8, title, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9.5)
    pdf.set_x(10)
    pdf.cell(190, 6, subtitle, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(12)
    pdf.set_text_color(25, 25, 25)
    pdf.set_font("Helvetica", "", 11)
    pdf.multi_cell(0, 6.5, body_text)

    pdf.ln(10)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(4)
    pdf.set_font("Helvetica", "I", 9)
    pdf.set_text_color(90, 90, 90)
    pdf.multi_cell(0, 5, footer_note)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(output_path))


# ============================================================================
# 4. BENEFICIARY FNOL AUDIO GENERATOR (GOOGLE CLOUD CHIRP 3: HD NATURAL VOICE)
# ============================================================================

def synthesize_fnol_audio(output_path: Path, transcript: str, voice_name: str, seed: int) -> str:
    """
    Uses Google Cloud Text-to-Speech `Chirp3-HD` generative human voices (authenticated via
    ADC or `gcloud auth print-access-token`) to produce natural conversational call-center
    beneficiary recordings.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from google.cloud import texttospeech

        creds = get_gcp_credentials()
        client = texttospeech.TextToSpeechClient(credentials=creds)
        synthesis_input = texttospeech.SynthesisInput(text=transcript)
        fallback_journey = "en-US-Journey-F" if voice_name.endswith(("Aoede", "Kore", "Leda")) else "en-US-Journey-D"

        for candidate_voice in (voice_name, fallback_journey):
            try:
                voice = texttospeech.VoiceSelectionParams(language_code="en-US", name=candidate_voice)
                audio_config = texttospeech.AudioConfig(
                    audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                    sample_rate_hertz=24000,
                )
                response = client.synthesize_speech(
                    input=synthesis_input, voice=voice, audio_config=audio_config
                )
                output_path.write_bytes(response.audio_content)
                return f"GOOGLE_CLOUD_TTS ({candidate_voice})"
            except Exception as inner_e:
                print(f"    [!] Voice {candidate_voice} warning: {inner_e}")
                continue
    except Exception as e:
        print(f"    [!] Google Cloud TTS warning: {e}")

    # Offline fallback if no GCP credentials exist
    try:
        subprocess.run(
            ["say", "-o", str(output_path), "--data-format=LEI16@22050", transcript],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if output_path.exists() and output_path.stat().st_size > 1000:
            return "LOCAL_SPOKEN_TTS_WAV"
    except Exception:
        pass

    raise RuntimeError(f"Failed to synthesize audio for {output_path}")


# ============================================================================
# 5. MAIN MULTI-MODAL DATA GENERATION PIPELINE
# ============================================================================

def main() -> None:
    print(f"==> Building Real Open-Source NIH + Cloud TTS Multi-Modal Dataset into '{DATA_DIR}/'...")
    structured_dir = DATA_DIR / "structured"
    ref_atlas_dir = DATA_DIR / "unstructured_evidence" / "ref_medical_atlas"
    claim_xray_dir = DATA_DIR / "unstructured_evidence" / "claim_radiographs"
    pdf_dir = DATA_DIR / "unstructured_evidence" / "policy_and_aps_pdfs"
    audio_dir = DATA_DIR / "unstructured_evidence" / "beneficiary_fnol_audio"

    for d in (structured_dir, ref_atlas_dir, claim_xray_dir, pdf_dir, audio_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Fetch 6 Real NIH Open-i Reference Medical Atlas Radiographs + Metadata Table
    atlas_rows = []
    hashes: dict[str, str] = {}
    for item in REF_MEDICAL_ATLAS:
        ref_id = item["ref_image_id"]
        filename = f"{ref_id}.png"
        out_path = ref_atlas_dir / filename
        sha = fetch_real_nih_radiograph(
            output_path=out_path,
            nih_image_url=item["nih_image_url"],
            transform_mode="reference",
        )
        hashes[ref_id] = sha
        print(f"  [✓] Fetched real NIH radiograph {filename} ({item['nih_source_id']})")
        atlas_rows.append(
            {
                "ref_image_id": ref_id,
                "image_gcs_uri": f"gs://{BUCKET_NAME}/unstructured_evidence/ref_medical_atlas/{filename}",
                "reference_category": item["reference_category"],
                "pathology_label": item["pathology_label"],
                "prior_claim_id": item["prior_claim_id"],
                "prior_policyholder_id": item["prior_policyholder_id"],
                "clinical_notes": item["clinical_notes"],
                "local_path": str(out_path),
            }
        )
    df_atlas = pd.DataFrame(atlas_rows)
    df_atlas.to_csv(structured_dir / "dim_ref_medical_atlas_meta.csv", index=False)
    df_atlas.to_parquet(structured_dir / "dim_ref_medical_atlas_meta.parquet", index=False)
    print(f"  [✓] Saved {len(df_atlas)} real NIH Open-i Reference Medical Atlas radiographs & metadata.")

    # 2. Generate 4 Master Life Insurance Policy Contract PDFs
    for policy_code, pol_doc in MASTER_POLICIES.items():
        pdf_path = pdf_dir / f"{policy_code}.pdf"
        render_pdf_document(
            output_path=pdf_path,
            title=pol_doc["title"],
            subtitle=pol_doc["subtitle"],
            body_text=pol_doc["body"],
            footer_note=f"OmniLife Insurance Company -- Policy Contract Form {policy_code} -- Governed by State Life & Living Benefits Code.",
        )
    print(f"  [✓] Generated {len(MASTER_POLICIES)} Master Life Insurance Policy Contract PDFs.")

    # 3. Build 5 Hero Life Insurance Claims (Real NIH X-Rays, PubMed-Grounded APS PDFs, Chirp3-HD Audio WAVs, Iceberg Rows)
    claim_rows = []
    tts_modes = []
    for idx, claim in enumerate(HERO_CLAIMS, start=1):
        cid = claim["claim_id"]
        pol_code = claim["policy_code"]

        # 3a. Real Clinical Radiograph PNG from NIH NLM Open-i
        xray_filename = f"{cid}_xray.png"
        xray_path = claim_xray_dir / xray_filename
        claim_sha = fetch_real_nih_radiograph(
            output_path=xray_path,
            nih_image_url=claim["nih_image_url"],
            transform_mode=claim["image_transform"],
        )
        hashes[cid] = claim_sha

        # 3b. Attending Physician Statement (APS) PDF grounded in real NIH PubMed Central case
        aps_filename = f"{cid}_aps.pdf"
        aps_path = pdf_dir / aps_filename
        render_pdf_document(
            output_path=aps_path,
            title=claim["aps_title"],
            subtitle=f"Attending Physician: {claim['aps_physician']}",
            body_text=claim["aps_body"],
            footer_note=f"Certified Medical & Coroner Record submitted for Claim {cid} | NIH Open-i Case Source: {claim['nih_source_id']}.",
        )

        # 3c. Beneficiary FNOL Recorded Audio WAV via Google Cloud Chirp 3: HD Voices
        audio_filename = f"{cid}_fnol.wav"
        audio_path = audio_dir / audio_filename
        tts_mode = synthesize_fnol_audio(
            output_path=audio_path,
            transcript=claim["fnol_transcript"],
            voice_name=claim["tts_voice"],
            seed=700 + idx,
        )
        tts_modes.append(tts_mode)
        print(f"  [✓] Built Claim {cid}: Real NIH X-Ray ({claim['nih_source_id']}) + APS PDF + Audio [{tts_mode}]")

        claim_rows.append(
            {
                "claim_id": cid,
                "policy_id": claim["policy_id"],
                "policyholder_id": claim["policyholder_id"],
                "policy_code": pol_code,
                "product_type": claim["product_type"],
                "insured_name": claim["insured_name"],
                "dob": claim["dob"],
                "gender": claim["gender"],
                "uw_health_class": claim["uw_health_class"],
                "declared_smoker": claim["declared_smoker"],
                "declared_medical_history": claim["declared_medical_history"],
                "issue_date": claim["issue_date"],
                "event_date": claim["event_date"],
                "months_since_issue": claim["months_since_issue"],
                "is_within_contestability": claim["is_within_contestability"],
                "claim_type": claim["claim_type"],
                "base_sum_assured": claim["base_sum_assured"],
                "has_add_rider": claim["has_add_rider"],
                "add_rider_sum_assured": claim["add_rider_sum_assured"],
                "has_ci_rider": claim["has_ci_rider"],
                "ci_rider_sum_assured": claim["ci_rider_sum_assured"],
                "claimed_benefit_amount": claim["claimed_benefit_amount"],
                "beneficiary_name": claim["beneficiary_name"],
                "beneficiary_relationship": claim["beneficiary_relationship"],
                "reported_cause_of_event": claim["reported_cause_of_event"],
                "image_gcs_uri": f"gs://{BUCKET_NAME}/unstructured_evidence/claim_radiographs/{xray_filename}",
                "aps_pdf_gcs_uri": f"gs://{BUCKET_NAME}/unstructured_evidence/policy_and_aps_pdfs/{aps_filename}",
                "policy_pdf_gcs_uri": f"gs://{BUCKET_NAME}/unstructured_evidence/policy_and_aps_pdfs/{pol_code}.pdf",
                "fnol_audio_gcs_uri": f"gs://{BUCKET_NAME}/unstructured_evidence/beneficiary_fnol_audio/{audio_filename}",
                "fnol_transcript_reference": claim["fnol_transcript"],
                "aps_text_reference": claim["aps_body"],
                "policy_text_reference": MASTER_POLICIES[pol_code]["body"],
                "expected_action": claim["expected_action"],
                "expected_payout": claim["expected_payout"],
            }
        )

    df_claims = pd.DataFrame(claim_rows)
    df_claims.to_csv(structured_dir / "iceberg_claims_policies.csv", index=False)
    df_claims.to_parquet(structured_dir / "iceberg_claims_policies.parquet", index=False)

    # Verify anti-fraud hash difference for CLM-CI-1004 vs PRIOR_CLM_8821_FRAUD
    assert hashes["CLM-CI-1004"] != hashes["PRIOR_CLM_8821_FRAUD"], "Fraud duplicate must have distinct SHA256!"
    print(f"  [✓] Generated 5 Hero Life Insurance Claims with REAL NIH radiographs & {tts_modes[0]} voices.")
    print(f"  [✓] Verified Fraud Case CLM-CI-1004 SHA256 ({hashes['CLM-CI-1004'][:10]}...) != Prior Claim SHA256 ({hashes['PRIOR_CLM_8821_FRAUD'][:10]}...).")
    print("==> Real Multi-Modal Dataset Build Complete!")


if __name__ == "__main__":
    main()
