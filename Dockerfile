# syntax=docker/dockerfile:1.7

ARG PYTHON_IMAGE=docker.1ms.run/python:3.11-slim
FROM ${PYTHON_IMAGE}

ARG APT_DEBIAN_URI=http://deb.debian.org/debian
ARG APT_SECURITY_URI=http://security.debian.org/debian-security

ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_TRUSTED_HOST=

# Ensure host/daemon proxy settings don't affect builds.
ENV HTTP_PROXY="" \
    HTTPS_PROXY="" \
    ALL_PROXY="" \
    NO_PROXY="" \
    http_proxy="" \
    https_proxy="" \
    all_proxy="" \
    no_proxy="" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_PYTHON_VERSION_WARNING=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_TRUSTED_HOST=${PIP_TRUSTED_HOST}

WORKDIR /app

# Install Playwright runtime dependencies (used by Gemini/Flow captcha in browser modes).
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt/lists,sharing=locked \
    printf 'Acquire::http::Proxy \"false\";\nAcquire::https::Proxy \"false\";\n' > /etc/apt/apt.conf.d/99disable-proxy \
    && if [[ -f /etc/apt/sources.list.d/debian.sources ]]; then \
         sed -i "s|http://deb.debian.org/debian|${APT_DEBIAN_URI}|g" /etc/apt/sources.list.d/debian.sources; \
         sed -i "s|http://security.debian.org/debian-security|${APT_SECURITY_URI}|g" /etc/apt/sources.list.d/debian.sources; \
       elif [[ -f /etc/apt/sources.list ]]; then \
         sed -i "s|http://deb.debian.org/debian|${APT_DEBIAN_URI}|g" /etc/apt/sources.list; \
         sed -i "s|http://security.debian.org/debian-security|${APT_SECURITY_URI}|g" /etc/apt/sources.list; \
       fi \
    && env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
    apt-get -o Acquire::Retries=5 -o Acquire::http::Timeout=30 -o Acquire::https::Timeout=30 update \
    && apt-get install -y \
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libpango-1.0-0 \
    libcairo2

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip,sharing=locked \
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
    pip install --retries 10 --timeout 60 --prefer-binary -r requirements.txt

# Install Playwright browsers (chromium). Safe no-op if playwright is not installed.
ENV PLAYWRIGHT_BROWSERS_PATH=/ms-playwright
RUN --mount=type=cache,target=/ms-playwright,sharing=locked \
    env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
    python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('playwright') else 1)" \
    && (playwright install chromium || true)

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
