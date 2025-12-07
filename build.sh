#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="${IMAGE_NAME:-sora2api}"
IMAGE_TAG="${IMAGE_TAG:-latest}"

  docker build \
    --build-arg HTTP_PROXY= \
    --build-arg HTTPS_PROXY= \
    --build-arg http_proxy= \
    --build-arg https_proxy= \
    -t "${IMAGE_NAME}:${IMAGE_TAG}" .
