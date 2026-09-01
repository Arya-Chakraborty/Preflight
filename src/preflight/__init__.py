"""Preflight: a local cost-optimizing gateway for frontier LLM APIs."""

from preflight.config import Settings, load_settings

__version__ = "0.3.0"
__all__ = ["Settings", "load_settings", "wrap", "__version__"]


def wrap(settings: Settings | None = None):
    """Library-mode entry point: returns an in-process Preflight client.

    Imported lazily so that `import preflight` stays cheap.
    """
    from preflight.client import PreflightClient

    return PreflightClient(settings or load_settings())
