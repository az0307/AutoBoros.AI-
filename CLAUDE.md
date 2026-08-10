# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**AutoBoros** is an AI automation OS — multi-agent orchestration with a human-in-the-loop approval ladder. This repository is the **product documentation and roadmap** for AutoBoros. The application source code lives elsewhere (see Related Repos); here you will find the architecture overview, the job-ladder specification, the security baseline, and testing notes.

This is a **docs-only repo** (`README.md` + `docs/`) — no application code, build system, or dependencies to install.

## Contents

| Path | Purpose |
|------|---------|
| `README.md` | Product overview: architecture diagram, the L0–L4 job ladder, quick-start pointers, security baseline. |
| `docs/TESTING.md` | Testing guidance for the AutoBoros stack. |

## The Job Ladder (L0–L4)

AutoBoros runs autonomous agents over structured job queues on a 5-level execution ladder; operators keep veto rights at every level:

| Level | Actor | Behaviour |
|-------|-------|-----------|
| L0 | Agent | Fully autonomous — no human touchpoint |
| L1 | Agent | Notifies operator; proceeds unless vetoed |
| L2 | Agent | Drafts action; waits for approval |
| L3 | Agent | Flags risk; requires explicit sign-off |
| L4 | Human | Hands off to operator — agent pauses |

## System Architecture (documented here, implemented in the code repos)

```text
React 19 Cockpit (Vite 6)
    ⇕ REST + WebSocket (JWT)
FastAPI :8000  ←  SQLAlchemy async + Alembic (Postgres / SQLite)
    ⇕ httpx webhook
n8n :5678  (workflow automation + external integrations)
    ⇕ HTTP callback
FastAPI  (resolves approvals, broadcasts WS events)
    ⇕ stdio
MCP server  (sandboxed tool access for LLM agents)
```

Security baseline (documented in the README): JWT login with brute-force lockout, MCP `shell_exec` via command allowlist + `shell=False`, mandatory WebSocket auth, explicit CORS origins (no wildcard in prod), and prod refusing to boot on default `SECRET_KEY` / `AB_PASSWORD`.

## Working On This Repo

- Edits are documentation changes. Keep the README's architecture diagram, ladder table, and security baseline consistent with the actual implementation in the code repos — when the code changes, update the docs here.
- No commands to run; validate by checking the Markdown renders.

## Related Repos (where the code lives)

- `Aurora-AI-Agency` → `autoboros/` — backend (FastAPI + n8n bridge + MCP, Python ≥3.11) and cockpit (React 19 / Vite 6). Canonical implementation.
- `autoborosai-dashboard` — Nexus ops dashboard (Next.js).
- `autoborosai.com` — public-facing website/app.
- `autoboros-backend` / `autoboros-cockpit` — split-out service repos.
