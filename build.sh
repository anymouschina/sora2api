#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-sora2api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

USE_BUILDX_CACHE="${USE_BUILDX_CACHE:-0}"
CACHE_DIR="${CACHE_DIR:-.buildx-cache}"

BUILD_ARGS=(
  --build-arg HTTP_PROXY="${HTTP_PROXY:-}"
  --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}"
  --build-arg NO_PROXY="${NO_PROXY:-}"
  --build-arg http_proxy="${http_proxy:-}"
  --build-arg https_proxy="${https_proxy:-}"
  --build-arg no_proxy="${no_proxy:-}"
  --build-arg ALL_PROXY="${ALL_PROXY:-}"
  --build-arg all_proxy="${all_proxy:-}"
)

if [[ "${USE_BUILDX_CACHE}" == "1" ]]; then
  docker buildx build \
    --load \
    --cache-from "type=local,src=${CACHE_DIR}" \
    --cache-to "type=local,dest=${CACHE_DIR},mode=max" \
    "${BUILD_ARGS[@]}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" .
else
  DOCKER_BUILDKIT=1 docker build \
    "${BUILD_ARGS[@]}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" .
fi
