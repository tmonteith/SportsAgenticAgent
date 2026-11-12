"""Fire-and-forget delivery of telemetry events to Kafka.

Requests hand events to emit(), which only puts them on a bounded in-memory queue.
A background task connects to Kafka and sends them. Nothing here can fail or slow a
request: when the queue is full, Kafka is unreachable, or a send fails, the event is
dropped and counted, and drops are logged at most once per DROP_LOG_INTERVAL_SECONDS.
"""

from __future__ import annotations

import asyncio
from collections import Counter, deque
import json
import time
from typing import Any, Callable, Deque, Dict, Mapping, Optional

from app.logging_config import get_logger
from app.telemetry.events import AGENT_EVENTS_TOPIC

logger = get_logger("telemetry")

MAX_QUEUE_EVENTS = 1000  # about 0.5 MB of events at most
BATCH_WAIT_SECONDS = 0.1  # the sender collects events this long before sending them
SEND_CHUNK_EVENTS = 20  # then hands them over in chunks of this many, yielding between
CONNECT_TIMEOUT_SECONDS = 5.0
SEND_TIMEOUT_SECONDS = 5.0
STOP_TIMEOUT_SECONDS = 3.0
RETRY_INITIAL_SECONDS = 5.0
RETRY_MAX_SECONDS = 60.0
DROP_LOG_INTERVAL_SECONDS = 60.0
REQUEST_TIMEOUT_MS = 5000
CLIENT_ID = "sportsagentic-agent"

# Why an event was dropped
DROP_QUEUE_FULL = "queue_full"
DROP_KAFKA_UNAVAILABLE = "kafka_unavailable"
DROP_SEND_FAILED = "send_failed"
DROP_DELIVERY_FAILED = "delivery_failed"

# Builds an unstarted producer with aiokafka's AIOKafkaProducer interface.
ProducerFactory = Callable[[str], Any]


class KeyPartitioner:
    """Kafka's default key hashing, skipped when the topic has a single partition.

    The answer is the same either way; this only saves aiokafka's pure-Python
    murmur2 for agent.events, which has one partition.
    """

    def __init__(self, default: Callable[[Any, Any, Any], Any]) -> None:
        self._default = default

    def __call__(self, key: Optional[bytes], all_partitions: Any, available: Any) -> Any:
        if len(all_partitions) == 1:
            return all_partitions[0]
        return self._default(key, all_partitions, available)


def aiokafka_producer(bootstrap_servers: str) -> Any:
    """An aiokafka producer with short timeouts. Imported here so it costs nothing when telemetry is off."""
    from aiokafka import AIOKafkaProducer
    from aiokafka.partitioner import DefaultPartitioner

    return AIOKafkaProducer(
        bootstrap_servers=bootstrap_servers,
        client_id=CLIENT_ID,
        request_timeout_ms=REQUEST_TIMEOUT_MS,
        partitioner=KeyPartitioner(DefaultPartitioner()),
    )


class KafkaTelemetry:
    """Bounded queue plus a background sender for agent.events."""

    def __init__(
        self,
        bootstrap_servers: str,
        topic: str = AGENT_EVENTS_TOPIC,
        producer_factory: ProducerFactory = aiokafka_producer,
        max_queue: int = MAX_QUEUE_EVENTS,
        batch_wait: float = BATCH_WAIT_SECONDS,
        connect_timeout: float = CONNECT_TIMEOUT_SECONDS,
        send_timeout: float = SEND_TIMEOUT_SECONDS,
        retry_initial: float = RETRY_INITIAL_SECONDS,
        retry_max: float = RETRY_MAX_SECONDS,
        drop_log_interval: float = DROP_LOG_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.bootstrap_servers = bootstrap_servers
        self.topic = topic
        self._producer_factory = producer_factory
        self._max_queue = max_queue
        self._batch_wait = batch_wait
        self._connect_timeout = connect_timeout
        self._send_timeout = send_timeout
        self._retry_initial = retry_initial
        self._retry_max = retry_max
        self._drop_log_interval = drop_log_interval
        self._clock = clock

        self._queue: Optional["asyncio.Queue[Mapping[str, Any]]"] = None
        # Taken off the queue by the sender but not yet handed to the producer.
        self._pending: Deque[Mapping[str, Any]] = deque()
        self._producer: Any = None
        self._sender: Optional["asyncio.Task[None]"] = None
        self._reporter: Optional["asyncio.Task[None]"] = None

        self.sent = 0
        self.dropped = 0
        self.dropped_by_reason: Counter[str] = Counter()
        self._unlogged_drops = 0
        self._last_drop_log: Optional[float] = None

    @property
    def running(self) -> bool:
        return self._sender is not None

    @property
    def connected(self) -> bool:
        return self._producer is not None

    def stats(self) -> Dict[str, Any]:
        return {
            "sent": self.sent,
            "dropped": self.dropped,
            "dropped_by_reason": dict(self.dropped_by_reason),
            "queued": (self._queue.qsize() if self._queue is not None else 0) + len(self._pending),
            "connected": self.connected,
        }

    async def start(self) -> None:
        """Start the background sender. Returns at once; connecting happens in the background."""
        if self.running:
            return
        self._queue = asyncio.Queue(maxsize=self._max_queue)
        self._sender = asyncio.create_task(self._run(), name="telemetry-sender")
        self._reporter = asyncio.create_task(self._report_drops(), name="telemetry-drop-reporter")
        logger.info("Telemetry on: sending %s to Kafka at %s", self.topic, self.bootstrap_servers)

    def emit(self, event: Mapping[str, Any]) -> None:
        """Queue an event for delivery. Never blocks and never raises."""
        try:
            if self._queue is None or self._sender is None:
                self._record_drop(DROP_KAFKA_UNAVAILABLE)
                return
            self._queue.put_nowait(event)
        except asyncio.QueueFull:
            self._record_drop(DROP_QUEUE_FULL)
        except Exception:  # pragma: no cover - defensive, emit must never raise
            logger.debug("Could not queue a telemetry event", exc_info=True)

    async def stop(self, timeout: float = STOP_TIMEOUT_SECONDS) -> None:
        """Send what is queued, within timeout, then stop. Never raises."""
        if not self.running:
            return
        tasks = [t for t in (self._sender, self._reporter) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        self._sender = None
        self._reporter = None

        producer, self._producer = self._producer, None
        queue = self._queue
        if queue is not None:
            while not queue.empty():
                self._pending.append(queue.get_nowait())
        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout
        if producer is not None:
            try:
                while self._pending and loop.time() < deadline:
                    await self._send(producer, self._pending[0], deadline - loop.time())
                    self._pending.popleft()
                await asyncio.wait_for(producer.stop(), max(0.1, deadline - loop.time()))
                await asyncio.sleep(0)  # let delivery callbacks run before the summary
            except Exception:
                logger.debug("Telemetry producer did not stop cleanly", exc_info=True)
        self._drop_pending(DROP_KAFKA_UNAVAILABLE)
        self._queue = None
        if self.dropped:
            logger.warning(
                "Telemetry stopped: %d event(s) sent, %d dropped (%s)",
                self.sent,
                self.dropped,
                self._describe_reasons(),
            )
        self._unlogged_drops = 0

    async def _run(self) -> None:
        assert self._queue is not None
        backoff = self._retry_initial
        while True:
            if self._producer is None:
                self._producer = await self._connect()
                if self._producer is None:
                    self._drop_queued(DROP_KAFKA_UNAVAILABLE)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, self._retry_max)
                    continue
                backoff = self._retry_initial
                logger.info("Telemetry connected to Kafka at %s", self.bootstrap_servers)
            # Wake once per batch window rather than once per event, so a busy
            # event loop isn't interrupted for every event.
            self._pending.append(await self._queue.get())
            if self._batch_wait > 0:
                await asyncio.sleep(self._batch_wait)
            while not self._queue.empty():
                self._pending.append(self._queue.get_nowait())
            sent_in_chunk = 0
            while self._pending:
                await self._send(self._producer, self._pending[0], self._send_timeout)
                self._pending.popleft()
                sent_in_chunk += 1
                if sent_in_chunk == SEND_CHUNK_EVENTS:
                    sent_in_chunk = 0
                    await asyncio.sleep(0)  # let requests run between chunks

    async def _connect(self) -> Any:
        producer = None
        try:
            producer = self._producer_factory(self.bootstrap_servers)
            await asyncio.wait_for(producer.start(), self._connect_timeout)
            return producer
        except asyncio.CancelledError:
            if producer is not None:
                await self._close_quietly(producer)
            raise
        except Exception as exc:
            logger.debug("Telemetry could not connect to Kafka: %s", exc)
            if producer is not None:
                await self._close_quietly(producer)
            return None

    async def _close_quietly(self, producer: Any) -> None:
        try:
            await asyncio.wait_for(producer.stop(), self._connect_timeout)
        except Exception:
            logger.debug("Closing an unconnected telemetry producer failed", exc_info=True)

    async def _send(self, producer: Any, event: Mapping[str, Any], timeout: float) -> None:
        try:
            value = json.dumps(event, separators=(",", ":")).encode("utf-8")
            key = str(event.get("request_id") or "").encode("utf-8") or None
            delivery = await asyncio.wait_for(
                producer.send(self.topic, value=value, key=key), max(0.01, timeout)
            )
        except asyncio.CancelledError:
            # A cancelled send is not sent; the event stays pending for stop().
            raise
        except Exception as exc:
            logger.debug("Telemetry send failed: %r", exc)
            self._record_drop(DROP_SEND_FAILED)
            return
        delivery.add_done_callback(self._on_delivery)

    def _on_delivery(self, delivery: "asyncio.Future[Any]") -> None:
        if delivery.cancelled() or delivery.exception() is not None:
            self._record_drop(DROP_DELIVERY_FAILED)
        else:
            self.sent += 1

    def _drop_queued(self, reason: str) -> None:
        assert self._queue is not None
        while not self._queue.empty():
            self._queue.get_nowait()
            self._record_drop(reason)

    def _drop_pending(self, reason: str) -> None:
        while self._pending:
            self._pending.popleft()
            self._record_drop(reason)

    def _record_drop(self, reason: str) -> None:
        self.dropped += 1
        self.dropped_by_reason[reason] += 1
        self._unlogged_drops += 1
        self._maybe_log_drops()

    def _maybe_log_drops(self) -> None:
        if not self._unlogged_drops:
            return
        now = self._clock()
        if self._last_drop_log is not None and now - self._last_drop_log < self._drop_log_interval:
            return
        logger.warning(
            "Dropped %d telemetry event(s) since the last report; %d in total (%s)",
            self._unlogged_drops,
            self.dropped,
            self._describe_reasons(),
        )
        self._unlogged_drops = 0
        self._last_drop_log = now

    async def _report_drops(self) -> None:
        while True:
            await asyncio.sleep(self._drop_log_interval)
            self._maybe_log_drops()

    def _describe_reasons(self) -> str:
        return ", ".join(f"{reason}={count}" for reason, count in sorted(self.dropped_by_reason.items()))
