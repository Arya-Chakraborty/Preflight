"""Configuration: preflight.yaml + PREFLIGHT_* environment variables."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ACTIONS = ("A1", "A2", "A3", "A4", "A5")
COMPOSITE_ACTIONS = ("A2A3", "A4A3")
ALL_ACTIONS = ACTIONS + COMPOSITE_ACTIONS


def canonical_action(action: str) -> str:
    """Map composite actions onto the estimator's A1–A5 one-hot basis."""
    if action in ACTIONS:
        return action
    if action.startswith("A4"):
        return "A4"
    if action.startswith("A2"):
        return "A2"
    if "A3" in action:
        return "A3"
    return "A5"


class ProviderCacheRule(BaseModel):
    """How a provider bills prompt-cache hits (multipliers on the input price)."""

    min_prefix_tokens: int = 1024
    read_mult: float = 0.50
    write_mult: float = 1.00
    ttl_s: int = 300


DEFAULT_CACHE_RULES: dict[str, ProviderCacheRule] = {
    "openai": ProviderCacheRule(min_prefix_tokens=1024, read_mult=0.50, write_mult=1.00, ttl_s=300),
    "anthropic": ProviderCacheRule(min_prefix_tokens=1024, read_mult=0.10, write_mult=1.25, ttl_s=300),
    "gemini": ProviderCacheRule(min_prefix_tokens=2048, read_mult=0.25, write_mult=1.00, ttl_s=3600),
    "default": ProviderCacheRule(),
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PREFLIGHT_", env_nested_delimiter="__", extra="ignore"
    )

    data_dir: Path = Path.home() / ".preflight"
    host: str = "127.0.0.1"
    port: int = 8411

    # Decision thresholds
    theta_high: float = 0.79
    theta_low: float = 0.60
    lambda_fail: float = 0.02
    tau: float = 0.25
    epsilon: float = 0.05

    # Action toggles
    enable_cache: bool = True
    enable_context_reuse: bool = True
    enable_compression: bool = True
    enable_grounding: bool = True
    enable_compose: bool = True  # A2A3 / A4A3: inject then compress tail
    fixed_action: str | None = None  # force one action (baseline mode)

    # Analyzer
    embedder: str = "auto"  # auto | sentence-transformers | hashing | off
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    hashing_dim: int = 256
    analyzer_timeout_ms: int = 250
    assembler_timeout_ms: int = 1000

    # Memory
    semantic_ttl_s: int = 86400
    context_ttl_s: int = 86400

    # Compression
    min_tail_tokens: int = 1500
    compression_rate: float = 0.5

    # Risk / quality
    retry_cost_mult: float = 1.5
    false_hit_alpha: float = 1.0
    audit_rate: float = 0.02
    retry_similarity: float = 0.85
    retry_window: int = 3
    auto_refit_every: int = 0  # 0 disables; else refit estimators every N logged rows
    uncertainty_fallback: bool = True
    uncertainty_n_min: int = 0  # 0 disables; prefer A5 when best action has fewer obs
    thompson_sampling: bool = False

    # Proxy ops
    api_key: str | None = None  # if set, require Authorization: Bearer or x-api-key
    spend_cap_usd: float | None = None  # global realized-spend ceiling
    session_spend_cap_usd: float | None = None

    # Cold-start priors
    prior_output_tokens: int = 256
    prior_pfail: dict[str, float] = Field(
        default_factory=lambda: {"A1": 0.02, "A2": 0.06, "A3": 0.08, "A4": 0.04, "A5": 0.05}
    )

    cache_rules: dict[str, ProviderCacheRule] = Field(
        default_factory=lambda: dict(DEFAULT_CACHE_RULES)
    )

    @field_validator("fixed_action")
    @classmethod
    def _valid_action(cls, v: str | None) -> str | None:
        if v is not None and v not in ALL_ACTIONS:
            raise ValueError(f"fixed_action must be one of {ALL_ACTIONS}, got {v!r}")
        return v

    @field_validator("data_dir", mode="before")
    @classmethod
    def _expand(cls, v):
        return Path(str(v)).expanduser()

    def cache_rule_for(self, provider: str) -> ProviderCacheRule:
        return self.cache_rules.get(provider, self.cache_rules.get("default", ProviderCacheRule()))

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from a YAML file (if present) merged with environment variables.

    Search order when no explicit path is given: ./preflight.yaml, ~/.preflight/preflight.yaml.
    Environment variables always win over file values (pydantic-settings behavior:
    init kwargs lose to env), which is what we want for deployment overrides.
    """
    candidates: list[Path] = []
    if config_path is not None:
        candidates = [Path(config_path)]
    else:
        candidates = [Path("preflight.yaml"), Path.home() / ".preflight" / "preflight.yaml"]

    file_values: dict = {}
    for cand in candidates:
        if cand.is_file():
            with open(cand) as fh:
                file_values = yaml.safe_load(fh) or {}
            break

    settings = Settings(**file_values)
    settings.ensure_dirs()
    return settings
