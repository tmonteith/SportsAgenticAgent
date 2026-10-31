---
name: sports-agent
description: Run, query, and test the SportsAgenticAgent FastAPI service that connects the SportsAgentic MCP server with llama-server and Gemini fallback. Use when starting the agent API, sending chat queries, checking upstream health, or running its test suite.
---

# SportsAgenticAgent Skill

This repository contains the FastAPI agent application (`app/main.py`) that bridges:
1. **SportsAgentic MCP Server** (`http://localhost:8080/sse`) for College Football tools (`get_cfb_scores`, `get_team_schedule`, `get_cfb_rankings`).
2. **`llama-server`** (`http://localhost:8081/v1`) as the primary local LLM inference backend.
3. **Google Gemini** (`gemini-2.5-flash`) as an automatic backup if `llama-server` times out or is unreachable, or as a selectable alternative.

## Running the FastAPI Service

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Docker / Docker Compose
```bash
docker compose up --build
```

### Helm (Kubernetes)
```bash
helm lint charts/sportsagentic-agent
helm upgrade --install sportsagentic-agent charts/sportsagentic-agent \
  --set config.mcpServerUrl="http://sportsagentic:8080/sse" \
  --set config.llamaServerUrl="http://llama-server:8081/v1"
```

## Querying the Agent API

### 1. Chat Completion (`POST /chat`)
```bash
curl -s http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is the schedule for Ole Miss?", "provider": "auto"}'
```
- `provider` accepts `"auto"` (default: `llama-server` with Gemini failover), `"llama"`, or `"gemini"`.

### 2. Check Upstream Health (`GET /health`)
```bash
curl -s http://localhost:8000/health
```

### 3. Inspect MCP Tools & Loaded Skills
```bash
curl -s http://localhost:8000/tools
curl -s http://localhost:8000/skills
```

## Running the Test Suite

```bash
pytest -v
```
