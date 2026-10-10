#!/usr/bin/env bash
# ============================================================================
# OmniClaim Life Lakehouse — Google Cloud Infrastructure Provisioning (`setup_gcp.sh`)
# ============================================================================
set -euo pipefail

export CLOUDSDK_CORE_DISABLE_PROMPTS=1

if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

export PROJECT_ID="${GCP_PROJECT_ID:-your-gcp-project-id}"
export REGION="${GCP_REGION:-us-central1}"
export BUCKET_NAME="${GCS_BUCKET_NAME:-${PROJECT_ID}-bucket}"
export DATASET_ID="${BQ_DATASET_ID:-omniclaim_lakehouse}"
export CONNECTION_ID="${BQ_CONNECTION_ID:-lakehouse-omniclaim-conn}"

echo "============================================================================"
echo "🛡️  OmniClaim Life Lakehouse — GCP Setup"
echo "   Project ID    : ${PROJECT_ID}"
echo "   Region        : ${REGION}"
echo "   GCS Bucket    : gs://${BUCKET_NAME}"
echo "   BQ Dataset    : ${DATASET_ID}"
echo "   BQ Connection : ${CONNECTION_ID}"
echo "============================================================================"

# 0. Check if project exists; if not, create it and link active billing account
if ! gcloud projects describe "${PROJECT_ID}" --quiet &>/dev/null; then
    echo "==> [0/5] Creating GCP Project '${PROJECT_ID}'..."
    gcloud projects create "${PROJECT_ID}" --name="OmniClaim Lakehouse Multimodal" --quiet
fi

gcloud config set project "${PROJECT_ID}" --quiet

BILLING_ENABLED=$(gcloud billing projects describe "${PROJECT_ID}" --format="value(billingEnabled)" --quiet 2>/dev/null || echo "False")
if [ "${BILLING_ENABLED}" != "True" ]; then
    BILLING_ACCOUNT=$(gcloud billing accounts list --filter="open=true" --format="value(ACCOUNT_ID)" --limit=1 --quiet 2>/dev/null || true)
    if [ -n "${BILLING_ACCOUNT}" ]; then
        echo "==> Linking active billing account ${BILLING_ACCOUNT} to ${PROJECT_ID}..."
        gcloud billing projects link "${PROJECT_ID}" --billing-account="${BILLING_ACCOUNT}" --quiet
    else
        echo "⚠️  WARNING: No open billing account found automatically. Please ensure billing is linked to ${PROJECT_ID}."
    fi
else
    echo "    Billing is active on ${PROJECT_ID}."
fi

echo "==> [1/5] Enabling Google Cloud Lakehouse & Vertex AI APIs..."
gcloud services enable \
    bigquery.googleapis.com \
    biglake.googleapis.com \
    bigqueryconnection.googleapis.com \
    storage.googleapis.com \
    aiplatform.googleapis.com \
    texttospeech.googleapis.com \
    --project="${PROJECT_ID}" \
    --quiet

echo "==> [2/5] Creating secured regional GCS bucket gs://${BUCKET_NAME} in ${REGION}..."
if ! gcloud storage buckets describe "gs://${BUCKET_NAME}" --project="${PROJECT_ID}" --quiet &>/dev/null; then
    gcloud storage buckets create "gs://${BUCKET_NAME}" \
        --project="${PROJECT_ID}" \
        --location="${REGION}" \
        --uniform-bucket-level-access \
        --public-access-prevention \
        --quiet
else
    echo "    Bucket gs://${BUCKET_NAME} already exists."
fi

echo "==> [3/5] Creating BigQuery Dataset ${PROJECT_ID}:${DATASET_ID} in ${REGION}..."
if ! bq --project_id="${PROJECT_ID}" show --dataset "${PROJECT_ID}:${DATASET_ID}" &>/dev/null; then
    bq --location="${REGION}" --project_id="${PROJECT_ID}" mk --dataset \
        --description="OmniClaim Life Insurance Multi-Modal Lakehouse" \
        "${PROJECT_ID}:${DATASET_ID}"
else
    echo "    Dataset ${PROJECT_ID}:${DATASET_ID} already exists."
fi

echo "==> [4/5] Creating BigQuery Cloud Resource Connection ${CONNECTION_ID}..."
if ! bq --project_id="${PROJECT_ID}" show --connection "${PROJECT_ID}.${REGION}.${CONNECTION_ID}" &>/dev/null; then
    bq mk --connection \
        --location="${REGION}" \
        --project_id="${PROJECT_ID}" \
        --connection_type=CLOUD_RESOURCE \
        "${CONNECTION_ID}"
else
    echo "    Connection ${PROJECT_ID}.${REGION}.${CONNECTION_ID} already exists."
fi

export CONNECTION_SA=$(bq --project_id="${PROJECT_ID}" show --format=json --connection "${PROJECT_ID}.${REGION}.${CONNECTION_ID}" \
    | python3 -c "import sys, json; print(json.load(sys.stdin)['cloudResource']['serviceAccountId'])")
echo "    Connection Service Account: ${CONNECTION_SA}"

echo "==> [5/5] Granting IAM Roles (Storage Object Admin for Managed Iceberg + Vertex AI User)..."
gcloud storage buckets add-iam-policy-binding "gs://${BUCKET_NAME}" \
    --member="serviceAccount:${CONNECTION_SA}" \
    --role="roles/storage.objectAdmin" \
    --quiet >/dev/null

gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${CONNECTION_SA}" \
    --role="roles/aiplatform.user" \
    --condition=None \
    --quiet >/dev/null

echo "✅ GCP Lakehouse Infrastructure Ready!"
