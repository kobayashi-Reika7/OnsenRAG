#!/bin/bash
# ============================================================
# OnsenRAG デプロイスクリプト
# ============================================================
# 使い方:
#   ./deploy.sh              # Backend (Cloud Run) + Frontend (Firebase Hosting)
#   ./deploy.sh backend      # Backend のみ
#   ./deploy.sh frontend     # Frontend のみ
# ============================================================

set -euo pipefail

PROJECT_ID="onsen-c5fec"
REGION="asia-northeast1"
SERVICE_NAME="onsen-rag"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# 色付き出力
info()  { echo -e "\033[1;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[1;32m[OK]\033[0m $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

deploy_backend() {
    info "=== Backend デプロイ (Cloud Run) ==="

    # Docker イメージをビルド & プッシュ
    info "Docker イメージをビルド中..."
    gcloud builds submit --tag "${IMAGE}" --project "${PROJECT_ID}" .

    # Cloud Run にデプロイ
    info "Cloud Run にデプロイ中..."
    gcloud run deploy "${SERVICE_NAME}" \
        --image "${IMAGE}" \
        --region "${REGION}" \
        --project "${PROJECT_ID}" \
        --platform managed \
        --allow-unauthenticated \
        --memory 2Gi \
        --cpu 2 \
        --timeout 120 \
        --min-instances 0 \
        --max-instances 3 \
        --set-env-vars "FIREBASE_PROJECT_ID=${PROJECT_ID}" \
        --set-env-vars "GOOGLE_API_KEY=$(grep GOOGLE_API_KEY .env | cut -d= -f2)" \
        --set-env-vars "GROQ_API_KEY=$(grep GROQ_API_KEY .env | cut -d= -f2)" \
        --set-env-vars "CORS_ORIGINS=https://${PROJECT_ID}.web.app,https://${PROJECT_ID}.firebaseapp.com" \
        --set-env-vars "ALLOW_FILE_ORIGIN=false"

    SERVICE_URL=$(gcloud run services describe "${SERVICE_NAME}" \
        --region "${REGION}" \
        --project "${PROJECT_ID}" \
        --format "value(status.url)")

    ok "Backend デプロイ完了: ${SERVICE_URL}"
}

deploy_frontend() {
    info "=== Frontend デプロイ (Firebase Hosting) ==="

    # Firebase Hosting にデプロイ
    firebase deploy --only hosting --project "${PROJECT_ID}"

    ok "Frontend デプロイ完了: https://${PROJECT_ID}.web.app"
}

deploy_firestore_rules() {
    info "=== Firestore ルール & インデックスをデプロイ ==="
    firebase deploy --only firestore --project "${PROJECT_ID}"
    ok "Firestore ルールデプロイ完了"
}

# メイン
case "${1:-all}" in
    backend)
        deploy_backend
        ;;
    frontend)
        deploy_frontend
        ;;
    firestore)
        deploy_firestore_rules
        ;;
    all)
        deploy_backend
        deploy_firestore_rules
        deploy_frontend
        echo ""
        ok "=== 全デプロイ完了 ==="
        echo "  Backend:  Cloud Run (${REGION})"
        echo "  Frontend: https://${PROJECT_ID}.web.app"
        echo "  Firestore: onsen-c5fec"
        ;;
    *)
        echo "使い方: $0 [backend|frontend|firestore|all]"
        exit 1
        ;;
esac
