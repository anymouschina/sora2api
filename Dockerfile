FROM 	docker.1ms.run/python:3.11-slim

# Ensure host/daemon proxy settings don't affect builds.
ENV HTTP_PROXY="" \
    HTTPS_PROXY="" \
    ALL_PROXY="" \
    NO_PROXY="" \
    http_proxy="" \
    https_proxy="" \
    all_proxy="" \
    no_proxy=""

WORKDIR /app

COPY requirements.txt .
RUN env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u NO_PROXY -u http_proxy -u https_proxy -u all_proxy -u no_proxy \
    pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
