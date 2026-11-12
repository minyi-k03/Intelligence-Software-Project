"""Configuration loader and schema definitions.

References:
- docs/INTERFACES.md §2
- AGENTS.md (temperature must be 0, no hardcoded values in pipeline code)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REL_CONFIG = Path("configs/default.yaml")
_VALID_SECTIONS = frozenset(
    {"model", "budget", "cap", "retrieval", "compress", "p1", "run", "data"}
)


def _is_valid_int(val: Any) -> bool:
    """Return True if val is an int and not a bool."""
    return isinstance(val, int) and not isinstance(val, bool)


@dataclass
class ModelConfig:
    tag: str = "qwen2.5:7b-instruct-q4_K_M"
    temperature: int = 0
    num_ctx: int = 8192
    keep_alive: str = "5m"

    def __post_init__(self) -> None:
        if not self.tag or not isinstance(self.tag, str):
            raise ValueError(f"model.tag must be a non-empty string. Got: {self.tag!r}")
        if isinstance(self.temperature, bool) or self.temperature != 0:
            raise ValueError(
                f"model.temperature must strictly be 0 (ADR-0003, AGENTS.md). Got: {self.temperature!r}"
            )
        if not _is_valid_int(self.num_ctx) or self.num_ctx <= 0:
            raise ValueError(f"model.num_ctx must be a positive int. Got: {self.num_ctx!r}")
        if not self.keep_alive or not isinstance(self.keep_alive, str):
            raise ValueError(f"model.keep_alive must be a non-empty string. Got: {self.keep_alive!r}")


@dataclass
class BudgetConfig:
    max_calls: int = 4
    max_requery: int = 2

    def __post_init__(self) -> None:
        if not _is_valid_int(self.max_calls) or self.max_calls <= 0:
            raise ValueError(f"budget.max_calls must be a positive int. Got: {self.max_calls!r}")
        if not _is_valid_int(self.max_requery) or self.max_requery < 0:
            raise ValueError(f"budget.max_requery cannot be negative. Got: {self.max_requery!r}")


@dataclass
class CapConfig:
    first: int = 512
    finalize: int = 128
    judge: int = 256
    retry_multiplier: int = 2

    def __post_init__(self) -> None:
        for name in ("first", "finalize", "judge", "retry_multiplier"):
            val = getattr(self, name)
            if not _is_valid_int(val) or val <= 0:
                raise ValueError(f"cap.{name} must be a positive int. Got: {val!r}")


@dataclass
class RetrievalConfig:
    k: int = 3

    def __post_init__(self) -> None:
        if not _is_valid_int(self.k) or self.k <= 0:
            raise ValueError(f"retrieval.k must be a positive int. Got: {self.k!r}")


@dataclass
class CompressConfig:
    m: int = 3
    window: int = 1

    def __post_init__(self) -> None:
        if not _is_valid_int(self.m) or self.m <= 0:
            raise ValueError(f"compress.m must be a positive int. Got: {self.m!r}")
        if not _is_valid_int(self.window) or self.window < 0:
            raise ValueError(f"compress.window cannot be negative. Got: {self.window!r}")


@dataclass
class P1Config:
    max_steps: int = 7

    def __post_init__(self) -> None:
        if not _is_valid_int(self.max_steps) or self.max_steps <= 0:
            raise ValueError(f"p1.max_steps must be a positive int. Got: {self.max_steps!r}")


@dataclass
class RunConfig:
    timeout_s: int = 120

    def __post_init__(self) -> None:
        if not _is_valid_int(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError(f"run.timeout_s must be a positive int. Got: {self.timeout_s!r}")


@dataclass
class DataConfig:
    seed: int = 42

    def __post_init__(self) -> None:
        if not _is_valid_int(self.seed):
            raise ValueError(f"data.seed must be an int. Got: {self.seed!r}")


@dataclass
class Settings:
    """Unified configuration for M(EBA) pipelines and evaluations."""

    model: ModelConfig = field(default_factory=ModelConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    cap: CapConfig = field(default_factory=CapConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    compress: CompressConfig = field(default_factory=CompressConfig)
    p1: P1Config = field(default_factory=P1Config)
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert settings to a nested dictionary snapshot (for RUN_META.json)."""
        return asdict(self)

    def get(self, dot_key: str, default: Any = None) -> Any:
        """Access configuration via dotted key (e.g., 'model.tag')."""
        parts = dot_key.split(".")
        target: Any = self
        for part in parts:
            if hasattr(target, part):
                target = getattr(target, part)
            elif isinstance(target, dict) and part in target:
                target = target[part]
            else:
                return default
        return target


_CACHED_SETTINGS: Optional[Settings] = None
_CACHED_PATH: Optional[Path] = None


def _resolve_config_path(config_path: Optional[str | Path] = None) -> Path:
    """Resolve configuration path relative to CWD or repository root."""
    if config_path is not None:
        candidate = Path(config_path)
        if candidate.exists():
            return candidate.resolve()
        repo_candidate = REPO_ROOT / candidate
        if repo_candidate.exists():
            return repo_candidate.resolve()
        raise FileNotFoundError(f"Config file not found: {config_path}")

    cwd_default = _DEFAULT_REL_CONFIG
    if cwd_default.exists():
        return cwd_default.resolve()
    repo_default = REPO_ROOT / _DEFAULT_REL_CONFIG
    if repo_default.exists():
        return repo_default.resolve()
    raise FileNotFoundError(
        f"Default config file not found at {_DEFAULT_REL_CONFIG} or {repo_default}"
    )


def load_settings(config_path: Optional[str | Path] = None) -> Settings:
    """Load and validate configuration from a YAML file.

    Args:
        config_path: Path to YAML config file. If None, defaults to 'configs/default.yaml'.

    Returns:
        Settings: Validated configuration instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If configuration values or section names are invalid.
    """
    path = _resolve_config_path(config_path)
    content = path.read_text(encoding="utf-8")

    if yaml is not None:
        raw = yaml.safe_load(content) or {}
    else:
        import json

        try:
            raw = json.loads(content)
        except Exception:
            raw = _simple_yaml_parse(content)

    if not isinstance(raw, dict):
        raise ValueError(f"Configuration root must be a mapping/dict, got {type(raw).__name__}")

    unknown_sections = set(raw.keys()) - _VALID_SECTIONS
    if unknown_sections:
        raise ValueError(f"Unknown top-level config sections: {sorted(unknown_sections)}")

    return Settings(
        model=ModelConfig(**(raw.get("model") or {})),
        budget=BudgetConfig(**(raw.get("budget") or {})),
        cap=CapConfig(**(raw.get("cap") or {})),
        retrieval=RetrievalConfig(**(raw.get("retrieval") or {})),
        compress=CompressConfig(**(raw.get("compress") or {})),
        p1=P1Config(**(raw.get("p1") or {})),
        run=RunConfig(**(raw.get("run") or {})),
        data=DataConfig(**(raw.get("data") or {})),
    )


def get_settings(config_path: Optional[str | Path] = None, reload: bool = False) -> Settings:
    """Get the cached singleton settings instance.

    Args:
        config_path: Path to YAML config file.
        reload: Force reload from disk if True.

    Returns:
        Settings: Cached or newly loaded configuration.
    """
    global _CACHED_SETTINGS, _CACHED_PATH
    resolved = _resolve_config_path(config_path)
    if _CACHED_SETTINGS is None or reload or _CACHED_PATH != resolved:
        _CACHED_SETTINGS = load_settings(resolved)
        _CACHED_PATH = resolved
    return _CACHED_SETTINGS


def _simple_yaml_parse(text: str) -> dict[str, Any]:
    """Lightweight fallback parser for nested 2-level key-values if pyyaml is missing."""
    result: dict[str, Any] = {}
    current_section: Optional[str] = None

    for line in text.splitlines():
        line = line.split("#")[0].rstrip()
        if not line:
            continue
        if not line.startswith(" ") and line.endswith(":"):
            current_section = line[:-1].strip()
            result[current_section] = {}
        elif line.startswith("  ") and ":" in line and current_section:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val.lstrip("-").isdigit():
                val_typed: Any = int(val)
            elif val.replace(".", "", 1).lstrip("-").isdigit() and val.count(".") == 1:
                val_typed = float(val)
            elif val.lower() == "true":
                val_typed = True
            elif val.lower() == "false":
                val_typed = False
            else:
                val_typed = val
            result[current_section][key] = val_typed

    return result
