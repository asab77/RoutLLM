# RoutLLM frontend

React and TypeScript conversation interface for the same-origin RoutLLM API.
Production composition posts JSON to `POST /v1/chat`; it contains no provider,
database, site-authentication, hostname, or deployment credentials.

## Local workflow

Use Node 24 and npm from this directory:

```sh
npm ci
npm run dev
npm run typecheck
npm test
npm run build
npm run preview
```

Vite development and preview bind to loopback. The production CSP permits only
same-origin connections with `connect-src 'self'`; there is no proxy or configurable
external API origin.

## Request behavior

Auto mode sends `routing_mode: "auto"` without a category, validation contract, or
quality threshold. The server executes its configured direct model. The frontend
does not infer, detect, or fabricate a category.

Manual mode requires the user to select one of the seven canonical categories and
sends `routing_mode: "manual"`. The server-owned adaptive router remains responsible
for selection at its fixed 0.80 threshold. Validation stays visibly Off until its
separate integration phase and no validation contract is synthesized.

Responses are discriminated by `execution_mode`. Direct traces show only direct
execution and the returned model. Adaptive traces show the manual category, initial
routed model, routing fallback, and returned model. Validation escalation appears
only when real execution metadata is present. Exact decimal cost strings are
displayed without rounding small nonzero values to zero.

HTTP errors are normalized from status and allowlisted backend codes. Backend JSON,
empty bodies, malformed bodies, and non-JSON Caddy 401 responses are supported;
raw provider messages, HTML, and exception text are never displayed. Requests are
not automatically retried and overlapping submissions are blocked.

## Test fixtures and privacy

Deterministic fixtures remain available through an explicitly injected
`InferenceClient` for unit/component tests. The production `App` default is the real
relative HTTP client, and no Settings or environment toggle enables fixture results.

Prompts and responses remain in React memory only. Refresh or New chat clears the
conversation. The frontend uses no browser persistence, analytics, remote fonts,
or remote assets. Activity reads bounded request-level operational summaries from
`GET /v1/activity`; those summaries contain no prompt, system prompt, or response
text and are never synthesized from local chat state.

The composer retains Enter-to-send, Shift+Enter newline handling, automatic growth,
fixed-size send control, and duplicate-submit protection. Theme and advanced-control
preferences remain memory-only.
