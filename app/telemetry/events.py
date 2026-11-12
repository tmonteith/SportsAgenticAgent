"""The agent.events contract: event names, field values, and identity helpers.

Events never carry prompt text, client IPs or raw user identifiers. The only values
derived from the user are prompt_chars, user_key (an HMAC of the client IP) and the
three UI headers, each kept only when well-formed.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import re
from typing import Optional
import uuid

AGENT_EVENTS_TOPIC = "agent.events"

REQUEST_STARTED = "request_started"
TOOL_CALLED = "tool_called"
REQUEST_COMPLETED = "request_completed"
REQUEST_FAILED = "request_failed"

# status on tool_called
TOOL_STATUS_OK = "ok"
TOOL_STATUS_ERROR = "error"
# status on request_completed / request_failed
REQUEST_STATUS_COMPLETED = "completed"
REQUEST_STATUS_FAILED = "failed"

# error_code values
ERROR_LLM = "llm_error"  # every provider the turn could use failed
ERROR_INTERNAL = "internal_error"  # an unexpected exception in the orchestrator
ERROR_CLIENT_DISCONNECTED = "client_disconnected"  # the stream was abandoned before it ended
ERROR_MCP = "mcp_error"  # the MCP server failed or the tool reported an error
ERROR_MCP_TIMEOUT = "mcp_timeout"  # the MCP call timed out

# Headers the web UI sends (amendment G12)
SESSION_ID_HEADER = "X-Session-Id"
CLIENT_HEADER = "X-Client"
APP_VERSION_HEADER = "X-App-Version"
REQUEST_ID_HEADER = "X-Request-Id"

_UUID = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}")
_CLIENT = re.compile(r"[a-z0-9._-]{1,32}")
_APP_VERSION = re.compile(r"[A-Za-z0-9._+-]{1,64}")


def new_id() -> str:
    """A random UUID v4 string, used for request_id and event_id."""
    return str(uuid.uuid4())


def rfc3339_utc(moment: datetime) -> str:
    """Format a time as RFC 3339 in UTC with millisecond precision, e.g. 2026-11-11T20:05:40.123Z."""
    return moment.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def user_key(salt: str, client_ip: str) -> str:
    """Lowercase hex HMAC-SHA256 of the client IP, keyed with the telemetry salt."""
    return hmac.new(salt.encode("utf-8"), client_ip.encode("utf-8"), hashlib.sha256).hexdigest()


def clean_session_id(value: Optional[str]) -> Optional[str]:
    """The X-Session-Id value in lowercase when it is a UUID, else None."""
    value = (value or "").strip()
    return value.lower() if _UUID.fullmatch(value) else None


def clean_client(value: Optional[str]) -> Optional[str]:
    """The X-Client value when it matches ^[a-z0-9._-]{1,32}$, else None."""
    value = (value or "").strip()
    return value if _CLIENT.fullmatch(value) else None


def clean_app_version(value: Optional[str]) -> Optional[str]:
    """The X-App-Version value when it matches ^[A-Za-z0-9._+-]{1,64}$, else None."""
    value = (value or "").strip()
    return value if _APP_VERSION.fullmatch(value) else None
