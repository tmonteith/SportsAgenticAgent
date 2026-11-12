"""Local benchmark: request latency and memory with telemetry off and on.

Not collected by pytest. Run from the repo root:

    python tests/bench_telemetry.py            # all modes, 3 rounds
    python tests/bench_telemetry.py --requests 2000 --rounds 5

Everything is in-process or on 127.0.0.1: a fake LLM router and a fake MCP client
answer instantly by default (the worst case for relative overhead, since real turns
take seconds; --llm-ms/--mcp-ms add simulated latency), and requests go through
httpx's ASGI transport. Modes:

  off          telemetry disabled (no KAFKA_BOOTSTRAP / TELEMETRY_SALT)
  refused      real aiokafka, bootstrap is a local port nothing listens on
  hung         real aiokafka, bootstrap accepts TCP but never answers
  reachable    real aiokafka against a minimal fake Kafka broker on 127.0.0.1

Each run happens in a fresh subprocess so RSS numbers are not shared, and the fake
brokers run in a process of their own, like a broker pod.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import os
from pathlib import Path
import socket
import statistics
import struct
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODES = ["off", "refused", "hung", "reachable"]


def rss_kb() -> int:
    """Current resident set size of this process in KiB."""
    statm = Path("/proc/self/statm")
    if statm.exists():
        return int(statm.read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE") // 1024
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True)
    return int(out.stdout.strip())


# --- fake Kafka broker ----------------------------------------------------------------------


class FakeKafkaBroker:
    """Answers ApiVersions, Metadata and Produce, enough for aiokafka to deliver messages."""

    def __init__(self, topic: str = "agent.events") -> None:
        self.topic = topic
        self.port = 0
        self.records = 0
        self._server: Optional[asyncio.base_events.Server] = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        from aiokafka.protocol.admin import ApiVersionResponse_v0
        from aiokafka.protocol.metadata import MetadataResponse_v1
        from aiokafka.protocol.produce import ProduceResponse_v3

        offset = 0
        try:
            while True:
                size = struct.unpack(">i", await reader.readexactly(4))[0]
                frame = await reader.readexactly(size)
                api_key, _api_version, correlation_id = struct.unpack(">hhi", frame[:8])
                if api_key == 18:  # ApiVersions
                    body = ApiVersionResponse_v0(0, [(0, 3, 3), (3, 1, 1), (18, 0, 0)]).encode()
                elif api_key == 3:  # Metadata
                    body = MetadataResponse_v1(
                        [(0, "127.0.0.1", self.port, None)],
                        0,
                        [(0, self.topic, False, [(0, 0, 0, [0], [0])])],
                    ).encode()
                elif api_key == 0:  # Produce: ack everything on partition 0
                    self.records += 1
                    body = ProduceResponse_v3([(self.topic, [(0, 0, offset, -1)])], 0).encode()
                    offset += 1
                else:
                    raise RuntimeError(f"unexpected api_key {api_key}")
                payload = struct.pack(">i", correlation_id) + body
                writer.write(struct.pack(">i", len(payload)) + payload)
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        finally:
            writer.close()


class HungBroker:
    """Accepts connections and never answers."""

    def __init__(self) -> None:
        self.port = 0
        self._server: Optional[asyncio.base_events.Server] = None

    async def start(self) -> None:
        async def hold(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            try:
                await reader.read()
            finally:
                writer.close()

        self._server = await asyncio.start_server(hold, "127.0.0.1", 0)
        self.port = self._server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self._server is not None:
            self._server.close()


def closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


async def serve_broker(kind: str) -> None:
    """Run a fake broker in this process (a child of the benchmark), printing its port."""
    broker: Any = FakeKafkaBroker() if kind == "reachable" else HungBroker()
    await broker.start()
    print(broker.port, flush=True)
    await asyncio.Event().wait()


class BrokerProcess:
    """A fake broker in its own process, so its CPU isn't charged to the app's event loop."""

    def __init__(self, kind: str) -> None:
        self.proc = subprocess.Popen(
            [sys.executable, __file__, "--broker", kind], stdout=subprocess.PIPE, text=True
        )
        assert self.proc.stdout is not None
        self.port = int(self.proc.stdout.readline().strip())

    def stop(self) -> None:
        self.proc.terminate()
        self.proc.wait(timeout=5)


# --- one run --------------------------------------------------------------------------------


async def run_mode(
    mode: str, requests: int, warmup: int, concurrency: int, llm_ms: float, mcp_ms: float
) -> Dict[str, Any]:
    import httpx

    from app.agent.orchestrator import AgentOrchestrator
    from app.config import Settings
    from app.llm.base import LLMResponse, ToolCallRequest
    from app.main import create_app

    class Router:
        async def chat(self, messages: Any, mcp_tools: Any = None, provider: Any = None) -> LLMResponse:
            if llm_ms:
                await asyncio.sleep(llm_ms / 1000.0)
            if messages[-1]["role"] == "user":
                return LLMResponse(
                    content="",
                    tool_calls=[ToolCallRequest(id="c1", name="get_team_schedule", arguments={"team": "Ole Miss"})],
                    input_tokens=900,
                    output_tokens=20,
                )
            return LLMResponse(content="Ole Miss is 10-2.", input_tokens=1400, output_tokens=12)

    class MCP:
        async def list_tools(self, fallback_to_defaults: bool = True) -> List[Dict[str, Any]]:
            return []

        async def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> str:
            if mcp_ms:
                await asyncio.sleep(mcp_ms / 1000.0)
            return '[{"week": 1, "home": "Ole Miss", "home_points": 41, "away_points": 10}]'

    broker: Optional[BrokerProcess] = None
    bootstrap: Optional[str] = None
    if mode == "refused":
        bootstrap = f"127.0.0.1:{closed_port()}"
    elif mode in ("hung", "reachable"):
        broker = BrokerProcess(mode)
        bootstrap = f"127.0.0.1:{broker.port}"

    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        skills_dir=str(ROOT / "skills"),
        rate_limit_enabled=False,
        log_level="WARNING",
        kafka_bootstrap=bootstrap if mode != "off" else None,
        telemetry_salt="bench-salt" if mode != "off" else None,
    )
    app = create_app(settings)
    app.state.orchestrator = AgentOrchestrator(mcp_client=MCP(), llm_router=Router(), skills_loader=app.state.skills_loader)

    headers = {
        "X-Real-IP": "203.0.113.7",
        "X-Session-Id": "3f2504e0-4f89-41d3-9a0c-0305e82c3301",
        "X-Client": "web",
        "X-App-Version": "a1b2c3d",
    }
    body = {"message": "How has Ole Miss done this season?"}
    gc.collect()
    rss_before_start = rss_kb()

    latencies: List[float] = []
    started = time.perf_counter()
    async with app.router.lifespan_context(app):
        startup_ms = (time.perf_counter() - started) * 1000.0
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://bench") as client:

            async def one() -> None:
                t0 = time.perf_counter()
                response = await client.post("/chat", json=body, headers=headers)
                latencies.append((time.perf_counter() - t0) * 1000.0)
                assert response.status_code == 200, response.text

            for _ in range(warmup):
                await one()
            latencies.clear()
            for _ in range(requests // concurrency):
                await asyncio.gather(*(one() for _ in range(concurrency)))
            await asyncio.sleep(0.3)  # let the last sends complete
            gc.collect()
            rss_after_traffic = rss_kb()
        telemetry = app.state.telemetry
        stats = telemetry.stats() if telemetry is not None else {}
    if broker is not None:
        broker.stop()

    latencies.sort()
    return {
        "mode": mode,
        "requests": len(latencies),
        "startup_ms": round(startup_ms, 2),
        "mean_ms": round(statistics.fmean(latencies), 3),
        "p50_ms": round(latencies[len(latencies) // 2], 3),
        "p95_ms": round(latencies[int(len(latencies) * 0.95) - 1], 3),
        "p99_ms": round(latencies[int(len(latencies) * 0.99) - 1], 3),
        "rss_before_start_kb": rss_before_start,
        "rss_after_traffic_kb": rss_after_traffic,
        "telemetry": stats,
        "python": sys.version.split()[0],
        "platform": sys.platform,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--requests", type=int, default=3000)
    parser.add_argument("--warmup", type=int, default=300)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--llm-ms", type=float, default=0.0, help="simulated latency per LLM call")
    parser.add_argument("--mcp-ms", type=float, default=0.0, help="simulated latency per MCP call")
    parser.add_argument("--child", help=argparse.SUPPRESS)
    parser.add_argument("--broker", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.broker:
        asyncio.run(serve_broker(args.broker))
        return
    if args.child:
        result = asyncio.run(
            run_mode(args.child, args.requests, args.warmup, args.concurrency, args.llm_ms, args.mcp_ms)
        )
        print(json.dumps(result))
        return

    modes = args.modes.split(",")
    results: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
    for round_number in range(args.rounds):
        for mode in modes:  # interleaved so drift affects every mode alike
            proc = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--child",
                    mode,
                    "--requests",
                    str(args.requests),
                    "--warmup",
                    str(args.warmup),
                    "--concurrency",
                    str(args.concurrency),
                    "--llm-ms",
                    str(args.llm_ms),
                    "--mcp-ms",
                    str(args.mcp_ms),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            result = json.loads(proc.stdout.strip().splitlines()[-1])
            results[mode].append(result)
            print(f"round {round_number + 1} {mode:9s} p50={result['p50_ms']}ms p95={result['p95_ms']}ms "
                  f"rss={result['rss_after_traffic_kb']}KiB telemetry={result['telemetry']}", file=sys.stderr)

    base = statistics.median(r["p95_ms"] for r in results[modes[0]])
    base_rss = statistics.median(r["rss_after_traffic_kb"] for r in results[modes[0]])
    print(f"\nconcurrency={args.concurrency} llm_ms={args.llm_ms} mcp_ms={args.mcp_ms} "
          f"requests/run={args.requests} rounds={args.rounds} "
          f"python={results[modes[0]][0]['python']} platform={results[modes[0]][0]['platform']}")
    print(f"{'mode':10s} {'p50 ms':>8s} {'p95 ms':>8s} {'p99 ms':>8s} {'p95 vs off':>11s} {'RSS MiB':>8s} {'RSS vs off':>11s} {'startup ms':>11s}")
    for mode in modes:
        runs = results[mode]
        p95 = statistics.median(r["p95_ms"] for r in runs)
        rss = statistics.median(r["rss_after_traffic_kb"] for r in runs)
        print(
            f"{mode:10s} {statistics.median(r['p50_ms'] for r in runs):8.3f} {p95:8.3f} "
            f"{statistics.median(r['p99_ms'] for r in runs):8.3f} {100.0 * (p95 - base) / base:+10.1f}% "
            f"{rss / 1024:8.1f} {(rss - base_rss) / 1024:+10.1f} {statistics.median(r['startup_ms'] for r in runs):11.2f}"
        )


if __name__ == "__main__":
    main()
