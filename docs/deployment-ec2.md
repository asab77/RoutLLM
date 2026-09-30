# RouteLLM single-EC2 deployment

This runbook deploys RouteLLM, PostgreSQL, Redis, and Caddy on one EC2 instance
with Docker Compose. It creates no AWS resources and runs no benchmark, DEV,
FINAL, training, or evaluation workflow.

## Architecture and prerequisites

Internet traffic reaches Caddy. The Caddy image contains the reproducibly built
React static assets and serves the dashboard itself; same-origin `/v1/*`,
`/health`, and `/ready` requests are proxied to one private Uvicorn worker. The
app reaches PostgreSQL and ephemeral Redis on a private Docker network and Vercel
AI Gateway over outbound HTTPS. Never expose ports 8000, 5432, 6379, 9090, or
3000. Only Caddy publishes host ports. Node is used only in the image build and
is not a production service or runtime dependency.

The current domainless path needs the existing EC2 instance, EBS root volume,
security group, and public address. It adds no Elastic IP, load balancer, Route
53, NAT Gateway, RDS, ElastiCache, domain, or third-party dependency. Confirm
current EC2, EBS, public IPv4, snapshot, and data-transfer prices and keep an AWS
Budget enabled with actual-spend alerts at 50%, 80%, and 100% plus a forecast
alert at 100%.

Install Docker Engine, the Compose plugin, Git, and enough disk space for images
and PostgreSQL. Enable Docker at boot.

## Prepare shared configuration and secrets

Create the host-only environment file from the documented template:

```sh
mkdir -p -m 0700 deploy
cp deploy/.env.production.example deploy/.env.production
chmod 0600 deploy/.env.production
```

The real file stays outside Git, is never copied into the image, and must be
owned by the deployment user. Replace every `REPLACE_...` value. Common settings,
including demo auth, gateway, database, Redis/rate-limit, metrics, router, and
input-limit settings, are required in both modes. Set
`ROUTELLM_DEFAULT_MODEL_ID` deliberately to an enabled registry model; startup
fails safely if it is missing or invalid. `ROUTELLM_DOMAIN` and
`ACME_EMAIL` are Mode B only and may be deleted from the Mode A host file. Mode A
has no mode-specific environment variable.

Use URL-safe random PostgreSQL passwords. Generate the demo password hash
without retaining the plaintext in the repository:

```sh
read -r -s -p 'Demo password: ' ROUTELLM_DEMO_PASSWORD
printf '%s' "$ROUTELLM_DEMO_PASSWORD" | docker run --rm -i caddy:2.10.2-alpine caddy hash-password
unset ROUTELLM_DEMO_PASSWORD
```

Put the resulting hash in single quotes in `deploy/.env.production` so `$`
characters remain literal. Never echo secrets into logs or command arguments.

## Mode A — domainless EC2 portfolio demo (current path)

Mode A costs nothing beyond the existing EC2/EBS deployment. It serves HTTP on
the instance's current public IPv4 address or public EC2 hostname. It is intended
for temporary, attended portfolio demonstrations: HTTP does not encrypt Basic
Auth credentials or inference traffic in transit. Do not use it on an untrusted
network or for sensitive prompts. Stop the instance when the demo is not needed.
The dashboard and all `/v1/*` API routes require the existing Basic Auth
credentials. `/health` and `/ready` remain unauthenticated for operational
probes. Public metrics, benchmark, schema, and documentation routes return 404;
authorized Prometheus access remains available only inside the app container.

Permit inbound TCP port 80 in the EC2 security group. Port 443 is not needed for
Mode A. Do not add any other public application or data-store port.

Use both Compose files for every Mode A operation:

```sh
docker compose --env-file deploy/.env.production -f compose.prod.yaml -f compose.demo.yaml config --quiet
docker compose --env-file deploy/.env.production -f compose.prod.yaml -f compose.demo.yaml build --pull
docker compose --env-file deploy/.env.production -f compose.prod.yaml -f compose.demo.yaml up -d postgres redis
docker compose --env-file deploy/.env.production -f compose.prod.yaml -f compose.demo.yaml run --rm migrate
docker compose --env-file deploy/.env.production -f compose.prod.yaml -f compose.demo.yaml up -d app caddy
docker compose --env-file deploy/.env.production -f compose.prod.yaml -f compose.demo.yaml ps
```

Set the public URL using the current EC2 address:

```sh
ROUTELLM_URL=http://EC2_PUBLIC_IPV4
```

The default EC2 public address can change after stop/start. Read the new address
from EC2 and update `ROUTELLM_URL`; no repository or Compose change is required.

## Mode B — domain plus automatic HTTPS (future stable path)

Mode B preserves the original Caddy automatic-TLS deployment. Point the chosen
domain at the instance, set `ROUTELLM_DOMAIN` and `ACME_EMAIL`, and permit inbound
TCP 80/443 plus UDP 443 as desired. Use only the production Compose file:

```sh
docker compose --env-file deploy/.env.production -f compose.prod.yaml config --quiet
docker compose --env-file deploy/.env.production -f compose.prod.yaml build --pull
docker compose --env-file deploy/.env.production -f compose.prod.yaml up -d postgres redis
docker compose --env-file deploy/.env.production -f compose.prod.yaml run --rm migrate
docker compose --env-file deploy/.env.production -f compose.prod.yaml up -d app caddy
```

Caddy obtains and renews certificates and redirects HTTP to HTTPS. Mode B is the
intended path for a stable public deployment. It uses the same auth scope as
Mode A, with credentials and inference traffic protected by TLS.

## Zero-provider verification

These checks make no paid LLM calls. They apply to either mode after setting
`ROUTELLM_URL` appropriately. The fake request still requires the demo gate.

```sh
docker compose --env-file deploy/.env.production -f compose.prod.yaml -f compose.demo.yaml ps
docker compose --env-file deploy/.env.production -f compose.prod.yaml -f compose.demo.yaml run --rm migrate python -m alembic current
curl -fsS "$ROUTELLM_URL/health"
curl -fsS "$ROUTELLM_URL/ready"
curl -o /dev/null -sS -w '%{http_code}\n' "$ROUTELLM_URL/"
curl -u "$DEMO_AUTH_USER" -fsS "$ROUTELLM_URL/" | grep -F '<div id="root"></div>'
curl -u "$DEMO_AUTH_USER" -fsS "$ROUTELLM_URL/v1/models"
curl -o /dev/null -sS -w '%{http_code}\n' "$ROUTELLM_URL/metrics"
curl -u "$DEMO_AUTH_USER" -H 'Content-Type: application/json' \
  -d '{"prompt":"deployment smoke","routing_mode":"auto","max_output_tokens":2}' \
  "$ROUTELLM_URL/v1/chat"
curl -u "$DEMO_AUTH_USER" -fsS "$ROUTELLM_URL/v1/activity?limit=5"
```

The public metrics request must return 404. Inspect authorized metrics and the
database-backed summary only inside the app container, using the same Compose
file selection as the active mode:

```sh
docker compose --env-file deploy/.env.production -f compose.prod.yaml -f compose.demo.yaml exec app \
  python -c "import os,urllib.request; r=urllib.request.Request('http://127.0.0.1:8000/metrics',headers={'Authorization':'Bearer '+os.environ['METRICS_BEARER_TOKEN']}); print(urllib.request.urlopen(r).status)"
docker compose --env-file deploy/.env.production -f compose.prod.yaml -f compose.demo.yaml exec app \
  python -c "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8000/v1/metrics/summary').read().decode())"
```

The unauthenticated dashboard request must return 401 and the public metrics
request must return 404. The browser and API share one origin, so CORS is neither
configured nor required. Unknown `/v1/*` routes are always sent to FastAPI and
can never fall back to the SPA. Non-API client-side routes fall back to
`index.html` after authentication.

The normal migration command upgrades an existing `0002` database through the
request-activity `0003` migration before the application is recreated. Confirm
the offline chat request appears in `/v1/activity` and low-level telemetry.
Review Compose logs for structured
JSON events and absence of prompts, credentials, or provider bodies. Verify the
host exposes only port 80 in Mode A, or 80/443 in Mode B. A rate-limit test may
temporarily lower the client quota and use only `fake-small`; restore it afterward.

## Optional one-provider smoke test

Only after zero-provider checks pass, make at most one explicit call with no
validation or escalation. Reconfirm the cheapest approved model and current
price first. Use `Reply exactly: OK`, temperature 0, and at most two output
tokens. Stop after that one provider attempt regardless of outcome.

## Routine operation, persistence, and recovery limits

The migration service uses the application image, waits for healthy PostgreSQL,
runs `alembic upgrade head`, and fails visibly. Migrations do not run during app
startup. For updates, check out the reviewed commit, rebuild, migrate, and
recreate long-running services. For rollback, check out the previously reviewed
commit and repeat; never downgrade a populated database without a reviewed plan.

With `restart: unless-stopped` and Docker enabled at boot, starting EC2 restores
containers automatically. To turn the portfolio off, stop the EC2 instance. Do
not manually stop every Compose service first because Docker can remember that
state.

PostgreSQL uses the persistent `routellm-prod_postgres_data` volume. Redis has no
volume and its counters reset on restart. Caddy state remains in named volumes;
Mode B certificates survive container replacement. Compose logs are sufficient;
no CloudWatch agent, logging vendor, Prometheus, or Grafana runs in this stack.
Monitor disk space and Docker log growth. One EBS volume is not a backup.

Ordinary startup loads the frozen router and initializes idle clients. It does
not call providers, Astra, DEV, or FINAL and does not run benchmark datasets.

To shut down Mode A without deleting state:

```sh
docker compose --env-file deploy/.env.production -f compose.prod.yaml -f compose.demo.yaml down
```

For Mode B, omit `-f compose.demo.yaml`. Never append `--volumes` unless permanent
data destruction is deliberate.
