# AutoBoros

> AI automation OS — multi-agent orchestration with human-in-the-loop approval ladder.

AutoBoros runs autonomous AI agents across structured job queues with a **5-level execution ladder** (L0 fully autonomous → L4 human-delegated). Every job lives on the ladder; agents escalate when uncertain; operators retain veto rights at every level. A real-time React cockpit surfaces active jobs, audit feed, and approval actions.

## Current architecture

```
React 19 Cockpit (Vite 6)
    ⇕ REST + WebSocket (JWT)
FastAPI :8000  ←  SQLAlchemy async + Alembic (Postgres / SQLite)
    ⇕ httpx webhook
n8n  :5678  (workflow automation + external integrations)
    ⇕ HTTP callback
FastAPI  (resolves approvals, broadcasts WS events)
    ⇕ stdio
MCP server  (sandboxed tool access for LLM agents)
```

**Code:** this repo holds only this README and `docs/TESTING.md`. The AutoBoros code was **removed from
`Aurora-AI-Agency`** (only an orphaned lockfile remains there) and the intended homes,
the private repos `az0307/autoboros-backend` and `az0307/autoboros-cockpit`, currently contain **only a
placeholder CLAUDE.md** (Oct 2026). Until the code is restored there (it is in `Aurora-AI-Agency` git
history), the quick start below cannot be run.

- `backend/` — FastAPI + n8n bridge + MCP server (Python ≥3.11)
- `cockpit/` — React 19 operator dashboard (Vite 6)
- `docs/` — Deployment guide, audit changelog, security review

## The job ladder (L0 – L4)

| Level | Actor | Behaviour |
|-------|-------|----------|
| L0 | Agent | Fully autonomous — no human touchpoint |
| L1 | Agent | Notifies operator; proceeds unless vetoed |
| L2 | Agent | Drafts action; waits for approval |
| L3 | Agent | Flags risk; requires explicit sign-off |
| L4 | Human | Hands off to operator — agent pauses |

## Quick start

```bash
# Backend (az0307/autoboros-backend)
pip install -e ".[dev]"
DATABASE_URL="sqlite+aiosqlite:///./dev.db" SECRET_KEY=x AB_PASSWORD=autoboros \
  uvicorn app.main:app --reload

# Cockpit (az0307/autoboros-cockpit)
npm install && npm run dev
```

Tests: `DATABASE_URL="sqlite+aiosqlite:///./t.db" SECRET_KEY=x AB_PASSWORD=autoboros python -m pytest tests/ -q`

## Security baseline

- JWT login with 5-try / 5-min brute-force lockout
- MCP `shell_exec` runs through command allowlist + `shell=False` (no RCE)
- WebSocket auth mandatory (closes 4001 on missing token)
- CORS explicit origin list — no wildcard in production
- Prod refuses boot on default `SECRET_KEY` / `AB_PASSWORD`

See `docs/SECONDARY_REVIEW.md` in the backend repo for open P1 items (JWT revocation, Redis rate-limit, httpOnly cookies).

## Roadmap — Neural Network OS

The next evolution: a self-improving neural business OS using local LLMs, vector memory, and CrewAI multi-agent coordination. Full implementation blueprint in [`docs/TESTING.md`](./docs/TESTING.md) and the phased deployment guide previously in this README (available in git history).

Key v2 components: Ollama multi-model, Qdrant vector DB, PostgreSQL + Redis memory, CrewAI agent crew, n8n automation bridge, Prometheus + Grafana monitoring.

---

AutoBoros · [autoboros.ai](https://autoboros.ai) · Part of the Ouroboros ecosystem · Australia · 2026
