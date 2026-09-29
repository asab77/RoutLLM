# RoutLLM frontend — Phase 12.2A.1

An offline React + TypeScript conversation interface. All routing decisions, response text,
model/provider labels, usage, and costs are deterministic fixtures. Neither the
frozen router nor a provider executes. Activity intentionally contains no metrics.

## Local development

Use Node 24 (validated on 24.13.0) and npm. From this directory:

```sh
npm ci
npm run dev
```

Open the loopback URL printed by Vite (normally http://127.0.0.1:5173).
No environment file, backend, database, Redis, Docker, AWS, or credentials are needed.
Package installation accesses the npm registry; inference itself is entirely local.

```sh
npm run typecheck
npm test
npm run build
npm run preview
```

## Safety and contracts

`api/types.ts` describes the existing request/response contract. `InferenceClient`
is an asynchronous boundary supporting AbortSignal, suitable for a future relative
HTTP implementation. This phase includes **only** `createSimulatedClient`: a timer
and cloned fixtures, with no fetch, HTTP adapter, API base URL, proxy, credential,
or live-mode switch. Production HTML also sets `connect-src 'none'`. Vite's local
development asset loading/HMR is development tooling, not inference traffic.

Settings → Demo scenarios changes predefined fixtures only: threshold met, routing
fallback, escalation-shaped success, and a 503 application error. It does not
execute validation or change the policy. Prompt/category/token edits do not alter
fixture content. Mock request IDs and usage are repeatable, not production records.

Costs remain decimal strings and display every digit. Initial selected model and
returned model are distinct. A changed model never implies routing fallback.
The threshold stays fixed at 0.80, with frontend token bounds 1–512 and a 12,000
character prompt limit. Temperature defaults to 1.0 and accepts 0–2. No validation
or system-prompt controls are implemented; validation is read-only Off.

## Conversation and settings

The bottom composer supports Enter to send, Shift+Enter for a newline, and IME
composition without premature submission. Turns stay in memory until New chat or
refresh. New chat clears messages/draft, not theme/generation preferences. It is
disabled while a response is pending. Recent items are labeled demo prompt
examples, not durable history, and only populate a draft (no automatic send).

The gear opens a small non-modal Settings popover, dismissible with Escape or an
outside pointer interaction. Advanced controls defaults Off. Turning it on opens
the optional drawer; the switch, X, and Escape inside the drawer all close it
consistently. On small screens the controls stack above chat, avoiding an overlay
that would obscure the composer. Invalid settings block sending even with the
drawer closed and provide a link to reopen it.

Auto-detect is a fixed QA presentation fixture, explicitly labeled in the drawer
and route. No classifier executes. Manual override exposes the seven supported
categories. These values feed the unchanged typed request boundary; replies remain
deterministic and do not depend on prompt/category/generation values.

Routes use native keyboard-accessible details/summary: Request → Router → initial
selected model → Response. The escalation fixture additionally identifies the
returned model without implying fallback or inventing per-attempt validation
outcomes. Exact costs and cumulative metrics retain the original semantics.

Theme defaults to System, resolving prefers-color-scheme through matchMedia and
listening for changes. Dark/Light overrides and all preferences are memory-only.
Both palettes use semantic CSS tokens; only active routes receive a subtle glow.
No decorative animation is used; reduced-motion styling is included.

Prompts and responses live only in React memory. No browser persistence, analytics,
remote fonts, or remote assets are used. Refresh clears the session. Inference is
submitted only by the form, cannot overlap another request, and never auto-retries.
Tests fail on network/storage attempts even if caught, and exercise success/error
paths. Static checks also reject network/storage APIs and external assets in app
source. These are regression safeguards, not a substitute for reviewing future
changes to the offline boundary.

## Manual visual QA

Phase 12.2A.1 uses TypeScript, Vitest/RTL/jsdom, static checks, and a Vite build only.
No browser automation, browser E2E, live browser agents, or automated screenshots
are used. Open the preview yourself and review desktop/narrow-screen layout,
Dark/Light/System, keyboard focus, the optional drawer, and all four route/error
scenarios. Refresh an existing preview tab after rebuilding.

React/React DOM are the only runtime dependencies. Vite, TypeScript, Vitest,
Testing Library, type definitions, and jsdom are development dependencies. jsdom
29.1.1 is the newest major compatible with the installed Node 24.13 runtime;
jsdom 30 requires Node 24.15 or newer. Versions are pinned in package-lock.json.

## Later integration

Phase 12.2B can add a separately reviewed relative-HTTP adapter, authentication
handling, and a bounded telemetry DTO. That work must intentionally update the
offline CSP and integration tests. Do not insert secrets into browser bundles or
`VITE_*` variables. Preserve existing Caddy operational protections and distinguish
recorded attempts, complete requests, and process-lifetime routing counters.
