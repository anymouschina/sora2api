#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-sora2api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

  docker build \
    --build-arg HTTP_PROXY="${HTTP_PROXY:-}" \
    --build-arg HTTPS_PROXY="${HTTPS_PROXY:-}" \
    --build-arg NO_PROXY="${NO_PROXY:-}" \
    --build-arg http_proxy="${http_proxy:-}" \
    --build-arg https_proxy="${https_proxy:-}" \
    --build-arg no_proxy="${no_proxy:-}" \
    --build-arg ALL_PROXY="${ALL_PROXY:-}" \
    --build-arg all_proxy="${all_proxy:-}" \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" .
