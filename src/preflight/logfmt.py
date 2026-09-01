"""JSON log lines for operators (stderr)."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone

_CONFIGURED = False

_EXTRA_KEYS = (
    "event",
    "request_id",
    "session",
    "action",
    "cost_realized",
    "cost_baseline",
    "latency_ms",
    "error",
)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key in _EXTRA_KEYS:
            val = getattr(record, key, None)
            if val is not None:
                payload[key] = val
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: int = logging.INFO) -> None:
    """Attach a JSON handler to the `preflight` logger.

    Call this only from an application entry point (the proxy server or CLI) —
    never from library code. It takes over the `preflight` logger (adds a stderr
    handler, disables propagation), which an embedding application must be free
    to opt into rather than have forced on it by ``import preflight``.

    Idempotent for the handler, but the log level is always (re)applied so a
    later call can raise or lower verbosity.
    """
    global _CONFIGURED
    logger = logging.getLogger("preflight")
    logger.setLevel(level)
    if _CONFIGURED or any(
        isinstance(h, logging.StreamHandler) and not isinstance(h, logging.NullHandler)
        for h in logger.handlers
    ):
        _CONFIGURED = True
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)
    logger.propagate = False
    _CONFIGURED = True
