FROM node:24.9.0-alpine AS frontend-build

WORKDIR /build/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

FROM python:3.12.12-slim-bookworm AS dependencies

ENV PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements.prod.lock .
RUN python -m pip download \
    --require-hashes \
    --only-binary=:all: \
    --dest /wheels \
    -r requirements.prod.lock

FROM python:3.12.12-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src

RUN groupadd --system --gid 10001 routellm \
    && useradd --system --uid 10001 --gid routellm --home-dir /nonexistent routellm

WORKDIR /app
COPY --from=dependencies /wheels /wheels
COPY requirements.prod.lock .
RUN python -m pip install \
    --no-cache-dir \
    --no-index \
    --find-links=/wheels \
    --require-hashes \
    -r requirements.prod.lock \
    && rm -rf /wheels

COPY --chown=routellm:routellm src ./src
COPY --chown=routellm:routellm migrations ./migrations
COPY --chown=routellm:routellm alembic.ini ./alembic.ini
COPY --chown=routellm:routellm deploy/router ./deploy/router

USER 10001:10001

CMD ["python", "-m", "uvicorn", "adaptive_llm_gateway.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers", "--forwarded-allow-ips", "172.30.0.2", "--no-access-log"]

FROM caddy:2.10.2-alpine AS edge

COPY --from=frontend-build /build/frontend/dist /srv
