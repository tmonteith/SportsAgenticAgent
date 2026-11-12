"""Agent telemetry published to the Kafka topic agent.events."""

from __future__ import annotations

from typing import List, Optional

from fastapi import Request

from app.config import Settings
from app.rate_limit import client_key
from app.telemetry.events import (
    APP_VERSION_HEADER,
    CLIENT_HEADER,
    SESSION_ID_HEADER,
    clean_app_version,
    clean_client,
    clean_session_id,
    user_key,
)
from app.telemetry.producer import KafkaTelemetry, ProducerFactory, aiokafka_producer
from app.telemetry.request import RequestTelemetry, current_request

__all__ = [
    "KafkaTelemetry",
    "RequestTelemetry",
    "build_telemetry",
    "current_request",
    "missing_telemetry_settings",
    "request_telemetry",
]


def missing_telemetry_settings(settings: Settings) -> List[str]:
    """Names of the settings that keep telemetry off; empty when it is on."""
    missing = []
    if settings.telemetry_bootstrap is None:
        missing.append("KAFKA_BOOTSTRAP")
    if settings.telemetry_key is None:
        missing.append("TELEMETRY_SALT")
    return missing


def build_telemetry(
    settings: Settings,
    producer_factory: ProducerFactory = aiokafka_producer,
) -> Optional[KafkaTelemetry]:
    """A KafkaTelemetry when KAFKA_BOOTSTRAP and TELEMETRY_SALT are both set, else None."""
    bootstrap = settings.telemetry_bootstrap
    if bootstrap is None or settings.telemetry_key is None:
        return None
    return KafkaTelemetry(bootstrap, producer_factory=producer_factory)


def request_telemetry(
    request: Request,
    settings: Settings,
    telemetry: Optional[KafkaTelemetry],
    request_id: str,
    prompt: str,
) -> Optional[RequestTelemetry]:
    """Telemetry for one agent turn, or None when telemetry is off.

    Only the prompt's length, a keyed hash of the client IP and the validated UI
    headers are kept; the prompt and the IP themselves never leave this function.
    """
    salt = settings.telemetry_key
    if telemetry is None or salt is None:
        return None
    headers = request.headers
    return RequestTelemetry(
        sink=telemetry,
        request_id=request_id,
        user_key=user_key(salt, client_key(request, settings.rate_limit_client_ip_header)),
        session_id=clean_session_id(headers.get(SESSION_ID_HEADER)),
        client=clean_client(headers.get(CLIENT_HEADER)),
        app_version=clean_app_version(headers.get(APP_VERSION_HEADER)),
        prompt_chars=len(prompt),
        model_names={"llama": settings.llama_model, "gemini": settings.gemini_model},
    )
