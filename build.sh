  docker build \
    --build-arg HTTP_PROXY= \
    --build-arg HTTPS_PROXY= \
    --build-arg http_proxy= \
    --build-arg https_proxy= \
    -t sora2api:latest .