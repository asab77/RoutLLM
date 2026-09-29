# RouteLLM single-EC2 deployment

This runbook deploys RouteLLM, PostgreSQL, Redis, and Caddy on one EC2 instance
with Docker Compose. It does not create AWS resources and does not run any
benchmark, DEV, FINAL, training, or evaluation workflow.

## Architecture and prerequisites

Internet traffic reaches Caddy on ports 80 and 443. Caddy terminates TLS and
proxies to one private Uvicorn worker. The app reaches PostgreSQL and ephemeral
Redis on a private Docker network and Vercel AI Gateway over outbound HTTPS.

The eventual AWS setup needs one EC2 instance, one EBS root volume, a security
group, a public address strategy, DNS, and an AWS Budget. Allow inbound 80/443;
restrict SSH to the operator's address or use a deliberate alternative. Never
open 8000, 5432, 6379, 9090, or 3000. Confirm current EC2, EBS, public IPv4,
snapshot, DNS, and data-transfer prices before creating anything.

Install Docker Engine, the Compose v2 plugin, Git, and enough disk space for
images and PostgreSQL. Enable Docker at boot. Point the chosen domain at the
instance before starting Caddy. A default EC2 public address may change after a
stop/start; update DNS or use a separately reviewed stable-address option.

## Prepare configuration and secrets

Check out the reviewed commit, then create the host-only environment file:

```sh
mkdir -p -m 0700 deploy
cp deploy/.env.production.example deploy/.env.production
chmod 0600 deploy/.env.production
```

The real file stays outside Git, is never copied into the image, and must be
owned by the deployment user. Replace every `REPLACE_...` value. Use URL-safe
random PostgreSQL passwords because the runtime password is embedded in a URL.
Generate a Caddy password hash without retaining the plaintext in the repository:

```sh
read -r -s -p 'Demo password: ' ROUTELLM_DEMO_PASSWORD
printf '%s' "$ROUTELLM_DEMO_PASSWORD" | docker run --rm -i caddy:2.10.2-alpine caddy hash-password
unset ROUTELLM_DEMO_PASSWORD
```

Put the resulting hash in single quotes in `deploy/.env.production` so `$`
characters remain literal. Never echo secrets into logs or place them in shell
command arguments. This small deployment intentionally uses a protected host file
rather than AWS Secrets Manager.

Set a monthly AWS Budget with actual-spend alerts at 50%, 80%, and 100%, plus a
forecast alert at 100%, before infrastructure creation.

## First deployment

All commands use the external environment file explicitly:

```sh
docker compose --env-file deploy/.env.production -f compose.prod.yaml config --quiet
docker compose --env-file deploy/.env.production -f compose.prod.yaml build --pull
docker compose --env-file deploy/.env.production -f compose.prod.yaml up -d postgres redis
docker compose --env-file deploy/.env.production -f compose.prod.yaml run --rm migrate
docker compose --env-file deploy/.env.production -f compose.prod.yaml up -d app caddy
docker compose --env-file deploy/.env.production -f compose.prod.yaml ps
```

The migration command uses the application image, waits for healthy PostgreSQL,
runs `alembic upgrade head`, and fails visibly. Migrations never run implicitly in
every app startup.

## Zero-provider verification

These checks make no paid LLM calls. Set `ROUTELLM_URL` to the HTTPS origin and
use the configured demo user for the fake request.

```sh
docker compose --env-file deploy/.env.production -f compose.prod.yaml ps
docker compose --env-file deploy/.env.production -f compose.prod.yaml run --rm migrate python -m alembic current
curl -fsS "$ROUTELLM_URL/health"
curl -fsS "$ROUTELLM_URL/ready"
curl -fsS "$ROUTELLM_URL/v1/models"
curl -o /dev/null -sS -w '%{http_code}\n' "$ROUTELLM_URL/metrics"
curl -u "$DEMO_AUTH_USER" -H 'Content-Type: application/json' \
  -d '{"model_id":"fake-small","prompt":"deployment smoke","max_output_tokens":2}' \
  "$ROUTELLM_URL/v1/inference"
```

The public metrics request must return 404. Inspect authorized metrics and the
database-backed summary only inside the app container/network:

```sh
docker compose --env-file deploy/.env.production -f compose.prod.yaml exec app \
  python -c "import os,urllib.request; r=urllib.request.Request('http://127.0.0.1:8000/metrics',headers={'Authorization':'Bearer '+os.environ['METRICS_BEARER_TOKEN']}); print(urllib.request.urlopen(r).status)"
docker compose --env-file deploy/.env.production -f compose.prod.yaml exec app \
  python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/v1/metrics/summary').read().decode())"
```

Confirm the fake inference appears in the telemetry summary. A rate-limit test may
temporarily lower the client quota and use only `fake-small`; restore the configured
quota afterward. Review `docker compose ... logs app` for JSON application events
and absence of prompts, credentials, or provider bodies. Verify only 80/443 are
public using the host's listening-port tool and an external scan.

## Optional one-provider smoke test

Only after all zero-provider checks pass, make at most one explicit call with no
validation or escalation. Reconfirm the cheapest available approved model and its
current price first. Use the prompt `Reply exactly: OK`, temperature `0`, and at
most two output tokens. Stop after that one provider attempt regardless of outcome.

## Routine start, stop, updates, and rollback

With `restart: unless-stopped` and Docker enabled at boot, starting EC2 restores
the containers automatically. Wait for `/ready`. To turn the portfolio off, stop
the EC2 instance. Do not manually stop every Compose service first because Docker
can remember that manual stopped state.

For an update, fetch the repository, check out the reviewed commit, rebuild, run
the explicit migration, and recreate the long-running services. For rollback,
check out the previously reviewed Git commit and repeat the build/up procedure.
Never downgrade a populated database unless a separately reviewed migration plan
explicitly permits it.

## Persistence, logs, and recovery limits

PostgreSQL uses `routellm-prod_postgres_data`; it survives container recreation
and EC2 stop/start because Docker stores it on EBS. Redis has no volume and its
rate-limit counters reset on restart. Caddy's ACME certificates and state use
named volumes. `docker compose down` preserves these volumes; never add `--volumes`
unless permanent data deletion is deliberate.

Use Compose logs for operations. No CloudWatch agent, logging vendor,
Prometheus, or Grafana runs in this production stack. Monitor free disk space and
Docker log growth. One root EBS volume is not a backup; termination, volume
deletion, corruption, or operator error can destroy telemetry. Add a separately
cost-reviewed snapshot/backup policy if that risk is unacceptable.

Ordinary startup only loads the frozen router, initializes local dependencies,
and creates an idle provider HTTP client. It does not call providers, Astra, DEV,
or FINAL, and it does not retrain or run benchmark datasets.

To shut down containers without deleting state:

```sh
docker compose --env-file deploy/.env.production -f compose.prod.yaml down
```

Do not append `--volumes`.
