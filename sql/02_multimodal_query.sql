-- ============================================================================
-- OmniClaim Life Lakehouse — Multi-Modal SQL Pipeline (`sql/02_multimodal_query.sql`)
-- 1. Multimodal Image Embeddings over BigLake Object Tables
-- 2. In-Engine BigQuery VECTOR_SEARCH (Pathology & Duplicate Fraud Detection)
-- 3. Gemini Native Multimodal Audit over PDF & Audio ObjectRefs
-- 4. Unified Adjudication 360 View with OBJ.GET_ACCESS_URL Signed URLs
-- ============================================================================

-- STEP 1A: Embed the Reference Medical & Prior-Claim Radiograph Atlas
CREATE OR REPLACE TABLE `__PROJECT_ID__.__DATASET_ID__.ref_medical_atlas_embeddings` AS
SELECT
    REGEXP_EXTRACT(uri, r'([^/]+)\.png$') AS ref_image_id,
    uri AS image_gcs_uri,
    ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
    MODEL `__PROJECT_ID__.__DATASET_ID__.model_multimodal_embed`,
    TABLE `__PROJECT_ID__.__DATASET_ID__.obj_ref_medical_atlas`
)
WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0;

-- STEP 1B: Embed Submitted Claim Radiographs
CREATE OR REPLACE TABLE `__PROJECT_ID__.__DATASET_ID__.claims_image_embeddings` AS
SELECT
    REGEXP_EXTRACT(uri, r'(CLM-[A-Z]+-[0-9]+)') AS claim_id,
    uri AS image_gcs_uri,
    content_type,
    updated AS uploaded_at,
    ml_generate_embedding_result AS embedding
FROM ML.GENERATE_EMBEDDING(
    MODEL `__PROJECT_ID__.__DATASET_ID__.model_multimodal_embed`,
    TABLE `__PROJECT_ID__.__DATASET_ID__.obj_claim_radiographs`
)
WHERE ARRAY_LENGTH(ml_generate_embedding_result) > 0;

-- ============================================================================
-- STEP 2: In-Engine BigQuery VECTOR_SEARCH (Cosine Similarity Forensics)
-- ============================================================================
CREATE OR REPLACE TABLE `__PROJECT_ID__.__DATASET_ID__.stg_radiograph_vector_forensics` AS
WITH VectorMatches AS (
    SELECT
        query.claim_id,
        query.image_gcs_uri AS submitted_image_uri,
        base.ref_image_id AS matched_ref_image_id,
        base.image_gcs_uri AS matched_ref_image_uri,
        distance AS cosine_distance,
        ROUND(1.0 - distance, 4) AS cosine_similarity
    FROM VECTOR_SEARCH(
        TABLE `__PROJECT_ID__.__DATASET_ID__.ref_medical_atlas_embeddings`,
        'embedding',
        TABLE `__PROJECT_ID__.__DATASET_ID__.claims_image_embeddings`,
        'embedding',
        top_k => 1,
        distance_type => 'COSINE',
        options => '{"use_brute_force": true}'
    )
)
SELECT
    vm.claim_id,
    vm.submitted_image_uri,
    vm.matched_ref_image_id,
    vm.matched_ref_image_uri,
    vm.cosine_distance,
    vm.cosine_similarity,
    meta.reference_category,
    meta.pathology_label AS detected_radiograph_pathology,
    meta.prior_claim_id,
    meta.prior_policyholder_id,
    meta.clinical_notes AS atlas_clinical_notes,
    CASE
        WHEN meta.reference_category = 'HISTORICAL_PRIOR_CLAIM' AND vm.cosine_distance < 0.05
            THEN 'CRITICAL_DUPLICATE_IMAGE_FRAUD'
        WHEN meta.pathology_label LIKE '%CARDIOMEGALY%'
            THEN 'CHRONIC_PRE_EXISTING_CARDIAC_INDICATOR'
        ELSE 'CONSISTENT_ACUTE_OR_TRAUMA_FINDING'
    END AS radiograph_forensic_flag
FROM VectorMatches vm
JOIN `__PROJECT_ID__.__DATASET_ID__.dim_ref_medical_atlas_meta` meta
    ON vm.matched_ref_image_id = meta.ref_image_id;

-- ============================================================================
-- STEP 3A: Gemini Multimodal Audit over Policy Contract PDF & APS PDF ObjectRefs
-- ============================================================================
CREATE OR REPLACE TABLE `__PROJECT_ID__.__DATASET_ID__.stg_document_clinical_audit` AS
SELECT
    claim_id,
    ml_generate_text_llm_result AS pdf_clinical_contract_audit
FROM ML.GENERATE_TEXT(
    MODEL `__PROJECT_ID__.__DATASET_ID__.model_gemini_auditor`,
    (
        SELECT
            c.claim_id,
            STRUCT(
                CONCAT(
                    'You are a Senior Life Insurance Claims Medical Underwriter and Legal Auditor. ',
                    'Examine the attached Policy Wording PDF and Attending Physician Statement (APS) PDF for Claim ', c.claim_id, '.\n',
                    'Structured Apache Iceberg Context:\n',
                    '- Insured: ', c.insured_name, ' | Product: ', c.product_type, ' (', c.policy_code, ')\n',
                    '- Claim Type: ', c.claim_type, ' | Claimed Benefit: $', CAST(c.claimed_benefit_amount AS STRING), '\n',
                    '- Months Since Policy Issue: ', CAST(c.months_since_issue AS STRING), ' (Within 24m Contestability Window: ', CAST(c.is_within_contestability AS STRING), ')\n',
                    '- Underwriting Class: ', c.uw_health_class, ' | Signed Health Declaration: "', c.declared_medical_history, '"\n',
                    '- Radiograph Vector Forensic Finding: ', vf.detected_radiograph_pathology, ' (Flag: ', vf.radiograph_forensic_flag, ')\n\n',
                    'Tasks:\n',
                    '1. Extract the primary diagnosis, ICD-10 code, and initial diagnosis/symptom onset date from the APS PDF.\n',
                    '2. Determine if the APS reveals undisclosed pre-existing conditions during the 24-month Contestability Window.\n',
                    '3. Identify any Policy Contract or Rider Exclusion clauses triggered (e.g., Section 1.4 Rescission, Section 4.2 AD&D Mountaineering >4,000m Exclusion, Section 2.1 Critical Illness threshold, or Section 5.3 Duplicate Radiograph Fraud).\n',
                    'Provide 3-4 concise bullet points.'
                ) AS prompt,
                [
                    OBJ.FETCH_METADATA(OBJ.MAKE_REF(c.policy_pdf_gcs_uri, '__PROJECT_ID__.__REGION__.__CONNECTION_ID__')),
                    OBJ.FETCH_METADATA(OBJ.MAKE_REF(c.aps_pdf_gcs_uri, '__PROJECT_ID__.__REGION__.__CONNECTION_ID__'))
                ] AS object_ref_list
            ) AS prompt
        FROM `__PROJECT_ID__.__DATASET_ID__.iceberg_claims_policies` c
        JOIN `__PROJECT_ID__.__DATASET_ID__.stg_radiograph_vector_forensics` vf ON c.claim_id = vf.claim_id
    ),
    STRUCT(
        0.0 AS temperature,
        1024 AS max_output_tokens,
        TRUE AS flatten_json_output
    )
);

-- ============================================================================
-- STEP 3B: Gemini Multimodal Audit over Beneficiary FNOL Audio (.WAV) ObjectRef
-- ============================================================================
CREATE OR REPLACE TABLE `__PROJECT_ID__.__DATASET_ID__.stg_audio_and_final_verdict` AS
SELECT
    claim_id,
    ml_generate_text_llm_result AS final_adjudication_json
FROM ML.GENERATE_TEXT(
    MODEL `__PROJECT_ID__.__DATASET_ID__.model_gemini_auditor`,
    (
        SELECT
            c.claim_id,
            STRUCT(
                CONCAT(
                    'You are the Chief Life Insurance Claims Adjudicator & SIU Director. ',
                    'Listen to the attached Beneficiary First Notice of Loss (FNOL) recorded statement audio file (.wav) and combine it with our Apache Iceberg policy data, Radiograph Vector Search forensics, and PDF Contract/APS audit.\n\n',
                    '=== CASE SUMMARY ===\n',
                    '- Claim ID: ', c.claim_id, ' | Insured: ', c.insured_name, ' | Policy Code: ', c.policy_code, '\n',
                    '- Claim Type: ', c.claim_type, ' | Claimed Benefit: $', CAST(c.claimed_benefit_amount AS STRING), '\n',
                    '- Base Sum Assured: $', CAST(c.base_sum_assured AS STRING), ' | AD&D Rider: $', CAST(c.add_rider_sum_assured AS STRING), ' | CI Rider: $', CAST(c.ci_rider_sum_assured AS STRING), '\n',
                    '- Months Since Issue: ', CAST(c.months_since_issue AS STRING), ' (Within 24m Contestability Window: ', CAST(c.is_within_contestability AS STRING), ')\n',
                    '- Radiograph Vector Forensic Flag: ', vf.radiograph_forensic_flag, ' (Matched Ref: ', vf.matched_ref_image_id, ' | Pathology: ', vf.detected_radiograph_pathology, ' | Cosine Similarity: ', CAST(vf.cosine_similarity AS STRING), ' | Prior Claim Match: ', IFNULL(vf.prior_claim_id, 'NONE'), ')\n',
                    '- PDF Contract & APS Audit Findings:\n', doc.pdf_clinical_contract_audit, '\n\n',
                    '=== DECISION RULES ===\n',
                    '- Use AUTO_APPROVE_PAYOUT when a natural death claim is past the 24-month contestability window and all evidence is consistent.\n',
                    '- Use SIU_RESCISSION_HOLD ($0 payout) when a claim is within the 24-month contestability window and the X-ray/APS/Audio reveal undisclosed pre-existing cardiac disease or pacemaker.\n',
                    '- Use PARTIAL_APPROVE_BASE_DENY_ADD_RIDER ($1,000,000 payout) when Base Life is incontestable (>24m) but the AD&D Rider is excluded under Section 4.2 (e.g., unroped free-solo climbing >4,000m).\n',
                    '- Use SIU_FRAUD_RING_REJECT ($0 payout) when Vector Search detects a duplicate/recycled radiograph from a prior claim (CRITICAL_DUPLICATE_IMAGE_FRAUD).\n',
                    '- Use APPROVE_LIVING_BENEFIT ($250,000 payout) when a Critical Illness claim meets Section 2.1 clinical criteria (>=2.0cm pulmonary neoplasm with biopsy) and has no fraud flags.\n\n',
                    'Return ONLY a valid JSON object (no markdown code fences) with these exact keys:\n',
                    '{\n',
                    '  "audio_transcript_summary": "1-2 sentence summary of what the caller stated on the audio recording",\n',
                    '  "audio_discrepancies_detected": "Any admission, contradiction, or red flag heard on the call (or NONE)",\n',
                    '  "contestability_alert": true or false,\n',
                    '  "rider_exclusion_triggered": "NONE or specific policy section triggered",\n',
                    '  "duplicate_image_fraud": true or false,\n',
                    '  "recommended_action": "AUTO_APPROVE_PAYOUT | SIU_RESCISSION_HOLD | PARTIAL_APPROVE_BASE_DENY_ADD_RIDER | SIU_FRAUD_RING_REJECT | APPROVE_LIVING_BENEFIT",\n',
                    '  "approved_payout_amount": numeric dollar amount approved,\n',
                    '  "executive_rationale": "3-sentence synthesis citing the Iceberg policy data, X-ray vector similarity, PDF clause, and Audio statement."\n',
                    '}'
                ) AS prompt,
                [
                    OBJ.FETCH_METADATA(OBJ.MAKE_REF(c.fnol_audio_gcs_uri, '__PROJECT_ID__.__REGION__.__CONNECTION_ID__'))
                ] AS object_ref_list
            ) AS prompt
        FROM `__PROJECT_ID__.__DATASET_ID__.iceberg_claims_policies` c
        JOIN `__PROJECT_ID__.__DATASET_ID__.stg_radiograph_vector_forensics` vf ON c.claim_id = vf.claim_id
        JOIN `__PROJECT_ID__.__DATASET_ID__.stg_document_clinical_audit` doc ON c.claim_id = doc.claim_id
    ),
    STRUCT(
        0.0 AS temperature,
        1024 AS max_output_tokens,
        TRUE AS flatten_json_output
    )
);

-- ============================================================================
-- STEP 4: Unified OmniClaim-360 Adjudication View with Signed GCS URLs
-- ============================================================================
CREATE OR REPLACE VIEW `__PROJECT_ID__.__DATASET_ID__.vw_claim_adjudication_360` AS
SELECT
    c.claim_id,
    c.policy_id,
    c.policyholder_id,
    c.policy_code,
    c.product_type,
    c.insured_name,
    c.dob,
    c.gender,
    c.uw_health_class,
    c.declared_smoker,
    c.declared_medical_history,
    c.beneficiary_name,
    c.beneficiary_relationship,
    c.claim_type,
    c.issue_date,
    c.event_date,
    c.months_since_issue,
    c.is_within_contestability,
    c.base_sum_assured,
    c.add_rider_sum_assured,
    c.ci_rider_sum_assured,
    c.claimed_benefit_amount,
    c.reported_cause_of_event,
    c.expected_action,
    c.expected_payout,
    -- Vector Search Forensics
    vf.matched_ref_image_id,
    vf.cosine_distance,
    vf.cosine_similarity,
    vf.detected_radiograph_pathology,
    vf.prior_claim_id AS matched_prior_claim_id,
    vf.prior_policyholder_id AS matched_prior_policyholder,
    vf.radiograph_forensic_flag,
    vf.atlas_clinical_notes,
    -- Gemini 3.8 Flash Multimodal Audits
    doc.pdf_clinical_contract_audit,
    aud.final_adjudication_json,
    -- Zero-Egress Signed URLs for Examiner UI (OBJ.GET_ACCESS_URL)
    OBJ.GET_ACCESS_URL(OBJ.MAKE_REF(c.image_gcs_uri, '__PROJECT_ID__.__REGION__.__CONNECTION_ID__'), 'r') AS signed_claim_xray_url,
    OBJ.GET_ACCESS_URL(OBJ.MAKE_REF(vf.matched_ref_image_uri, '__PROJECT_ID__.__REGION__.__CONNECTION_ID__'), 'r') AS signed_matched_ref_xray_url,
    OBJ.GET_ACCESS_URL(OBJ.MAKE_REF(c.aps_pdf_gcs_uri, '__PROJECT_ID__.__REGION__.__CONNECTION_ID__'), 'r') AS signed_aps_pdf_url,
    OBJ.GET_ACCESS_URL(OBJ.MAKE_REF(c.policy_pdf_gcs_uri, '__PROJECT_ID__.__REGION__.__CONNECTION_ID__'), 'r') AS signed_policy_pdf_url,
    OBJ.GET_ACCESS_URL(OBJ.MAKE_REF(c.fnol_audio_gcs_uri, '__PROJECT_ID__.__REGION__.__CONNECTION_ID__'), 'r') AS signed_fnol_audio_url
FROM `__PROJECT_ID__.__DATASET_ID__.iceberg_claims_policies` c
LEFT JOIN `__PROJECT_ID__.__DATASET_ID__.stg_radiograph_vector_forensics` vf ON c.claim_id = vf.claim_id
LEFT JOIN `__PROJECT_ID__.__DATASET_ID__.stg_document_clinical_audit` doc ON c.claim_id = doc.claim_id
LEFT JOIN `__PROJECT_ID__.__DATASET_ID__.stg_audio_and_final_verdict` aud ON c.claim_id = aud.claim_id;
