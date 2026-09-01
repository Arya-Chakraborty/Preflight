"""Preflight: a local cost-optimizing gateway for frontier LLM APIs."""

import logging

from preflight.config import Settings, load_settings

__version__ = "0.4.0"
__all__ = ["Settings", "load_settings", "wrap", "__version__"]

# Library best practice: never emit logs (or hijack the root logger) just because
# `preflight` was imported. Applications call `preflight.logfmt.configure_logging()`
# — done automatically by the proxy server and CLI — to opt into structured logs.
logging.getLogger("preflight").addHandler(logging.NullHandler())


def wrap(settings: Settings | None = None):
    """Library-mode entry point: returns an in-process Preflight client.

    Imported lazily so that `import preflight` stays cheap.
    """
    from preflight.client import PreflightClient

    return PreflightClient(settings or load_settings())
