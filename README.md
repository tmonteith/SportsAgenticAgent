# SportsAgenticAgent

FastAPI College Football AI Agent application in Python that connects:
1. **SportsAgentic MCP Server** (`http://localhost:8080/sse`) for live College Football scores, team schedules, and national rankings.
2. **`llama-server`** (`http://localhost:8081/v1`) as the primary local LLM inference backend.
3. **Google Gemini** (`gemini-2.5-flash`) as an automatic backup when `llama-server` times out or is unreachable, or as a selectable alternative.

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install runtime and test dependencies
pip install -r requirements-dev.txt

# 3. Configure environment variables
cp .env.example .env
```

---

## Running the FastAPI Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Containerization (Docker & Docker Compose)

Build and run with Docker Compose:

```bash
cp .env.example .env
docker compose up --build
```

Or build and run the image directly:

```bash
docker build -t sportsagentic-agent:latest .
docker run --rm -p 8000:8000 --env-file .env sportsagentic-agent:latest
```

---

## Kubernetes Deployment (Helm)

A Helm chart is provided in [`charts/sportsagentic-agent`](./charts/sportsagentic-agent).

```bash
# Lint and preview rendered manifests
helm lint charts/sportsagentic-agent
helm template sportsagentic-agent charts/sportsagentic-agent

# Install or upgrade the release
helm upgrade --install sportsagentic-agent charts/sportsagentic-agent \
  --namespace sportsagentic --create-namespace \
  --set config.mcpServerUrl="http://sportsagentic:8080/sse" \
  --set config.llamaServerUrl="http://llama-server:8081/v1" \
  --set secrets.geminiApiKey="${GEMINI_API_KEY}"
```

To use an existing Kubernetes Secret for `GEMINI_API_KEY`:

```bash
helm upgrade --install sportsagentic-agent charts/sportsagentic-agent \
  --set secrets.existingSecret="my-gemini-secret" \
  --set secrets.existingSecretKey="GEMINI_API_KEY"
```

---

## API Endpoints

### `POST /chat`
Send a message to the agent. By default (`"provider": "auto"`), inference runs on `llama-server` and automatically fails over to Gemini if `llama-server` times out or is down.

```bash
curl -s http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What was the score of the Ole Miss game?",
    "provider": "auto"
  }'
```

### `GET /health`
Check reachability and status of `SportsAgentic` MCP, `llama-server`, and Gemini configuration.

```bash
curl -s http://localhost:8000/health
```

### `GET /tools`
List MCP tools discovered from `SportsAgentic` (`get_cfb_scores`, `get_team_schedule`, `get_cfb_rankings`).

### `GET /skills`
List loaded markdown skill definitions (`skills/cfb.md`, `.agents/skills/sports-agent/SKILL.md`).

---

## Running the Test Suite

```bash
pytest -v
```

