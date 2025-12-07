#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-sora2api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
AWS_REGION="${AWS_REGION:-ap-southeast-2}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-350073489756}"
ECR_REPO="${ECR_REPO:-sora2api}"

REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REMOTE_IMAGE="${REGISTRY}/${ECR_REPO}:${IMAGE_TAG}"

echo "Logging in to ${REGISTRY}..."
aws ecr get-login-password --region "${AWS_REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}"

echo "Building ${IMAGE_NAME}:${IMAGE_TAG}..."
IMAGE_NAME="${IMAGE_NAME}" IMAGE_TAG="${IMAGE_TAG}" bash ./build.sh

echo "Tagging ${IMAGE_NAME}:${IMAGE_TAG} -> ${REMOTE_IMAGE}..."
docker tag "${IMAGE_NAME}:${IMAGE_TAG}" "${REMOTE_IMAGE}"

echo "Pushing ${REMOTE_IMAGE}..."
docker push "${REMOTE_IMAGE}"

echo "Image pushed successfully."
