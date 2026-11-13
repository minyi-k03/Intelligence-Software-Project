"""Configuration loader and schema definitions.

References:
- docs/INTERFACES.md §2
- AGENTS.md (temperature must be 0, no hardcoded model tag/temperature/num_ctx in code)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Optional, Union

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_REL_CONFIG = Path("configs/default.yaml")
_REQUIRED_SECTIONS = (
    "model",
    "budget",
    "cap",
    "retrieval",
    "compress",
    "p1",
    "run",
    "data",
)


def _is_valid_int(val: Any) -> bool:
    """Return True if val is an int and not a bool."""
    return isinstance(val, int) and not isinstance(val, bool)


@dataclass
class ModelConfig:
    tag: str
    temperature: int
    num_ctx: int
    keep_alive: Union[str, int]

    def __post_init__(self) -> None:
        if not self.tag or not isinstance(self.tag, str):
            raise ValueError(f"model.tag must be a non-empty string. Got: {self.tag!r}")
        if isinstance(self.temperature, bool) or self.temperature != 0:
            raise ValueError(
                f"model.temperature must strictly be 0 (ADR-0003, AGENTS.md). Got: {self.temperature!r}"
            )
        if not _is_valid_int(self.num_ctx) or self.num_ctx <= 0:
            raise ValueError(f"model.num_ctx must be a positive int. Got: {self.num_ctx!r}")
        if isinstance(self.keep_alive, bool):
            raise ValueError(f"model.keep_alive must be a string or int, not bool. Got: {self.keep_alive!r}")
        if isinstance(self.keep_alive, str):
            if not self.keep_alive.strip():
                raise ValueError("model.keep_alive string cannot be empty.")
        elif isinstance(self.keep_alive, int):
            if self.keep_alive < -1:
                raise ValueError(f"model.keep_alive int must be >= -1. Got: {self.keep_alive!r}")
        else:
            raise ValueError(f"model.keep_alive must be str or int. Got: {self.keep_alive!r}")


@dataclass
class BudgetConfig:
    max_calls: int
    max_requery: int

    def __post_init__(self) -> None:
        if not _is_valid_int(self.max_calls) or self.max_calls <= 0:
            raise ValueError(f"budget.max_calls must be a positive int. Got: {self.max_calls!r}")
        if not _is_valid_int(self.max_requery) or self.max_requery < 0:
            raise ValueError(f"budget.max_requery cannot be negative. Got: {self.max_requery!r}")


@dataclass
class CapConfig:
    first: int
    finalize: int
    judge: int
    retry_multiplier: int

    def __post_init__(self) -> None:
        for name in ("first", "finalize", "judge", "retry_multiplier"):
            val = getattr(self, name)
            if not _is_valid_int(val) or val <= 0:
                raise ValueError(f"cap.{name} must be a positive int. Got: {val!r}")


@dataclass
class RetrievalConfig:
    k: int

    def __post_init__(self) -> None:
        if not _is_valid_int(self.k) or self.k <= 0:
            raise ValueError(f"retrieval.k must be a positive int. Got: {self.k!r}")


@dataclass
class CompressConfig:
    m: int
    window: int

    def __post_init__(self) -> None:
        if not _is_valid_int(self.m) or self.m <= 0:
            raise ValueError(f"compress.m must be a positive int. Got: {self.m!r}")
        if not _is_valid_int(self.window) or self.window < 0:
            raise ValueError(f"compress.window cannot be negative. Got: {self.window!r}")


@dataclass
class P1Config:
    max_steps: int

    def __post_init__(self) -> None:
        if not _is_valid_int(self.max_steps) or self.max_steps <= 0:
            raise ValueError(f"p1.max_steps must be a positive int. Got: {self.max_steps!r}")


@dataclass
class RunConfig:
    timeout_s: int

    def __post_init__(self) -> None:
        if not _is_valid_int(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError(f"run.timeout_s must be a positive int. Got: {self.timeout_s!r}")


@dataclass
class DataConfig:
    seed: int

    def __post_init__(self) -> None:
        if not _is_valid_int(self.seed):
            raise ValueError(f"data.seed must be an int. Got: {self.seed!r}")


@dataclass
class Settings:
    """Unified configuration for M(EBA) pipelines and evaluations."""

    model: ModelConfig
    budget: BudgetConfig
    cap: CapConfig
    retrieval: RetrievalConfig
    compress: CompressConfig
    p1: P1Config
    run: RunConfig
    data: DataConfig

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


def _build_section(cls: type, section_name: str, data: Any) -> Any:
    """Validate and instantiate a required configuration section dataclass."""
    if not isinstance(data, dict):
        raise ValueError(
            f"Config section '{section_name}' must be a mapping/dict, got {type(data).__name__}"
        )

    expected_keys = {f.name for f in fields(cls)}
    provided_keys = set(data.keys())

    missing_keys = sorted(expected_keys - provided_keys)
    if missing_keys:
        raise ValueError(
            f"Missing required keys in '{section_name}': {missing_keys}"
        )

    unknown_keys = sorted(provided_keys - expected_keys)
    if unknown_keys:
        raise ValueError(
            f"Unknown keys in '{section_name}': {unknown_keys}"
        )

    return cls(**data)


def load_settings(config_path: Optional[str | Path] = None) -> Settings:
    """Load and strictly validate configuration from a YAML file.

    All sections and keys defined in docs/INTERFACES.md §2 are required;
    no hardcoded defaults are substituted if any section or key is missing.

    Args:
        config_path: Path to YAML config file. If None, defaults to 'configs/default.yaml'.

    Returns:
        Settings: Validated configuration instance.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
        ValueError: If the YAML is empty, has missing/unknown sections, or invalid values.
    """
    path = _resolve_config_path(config_path)
    content = path.read_text(encoding="utf-8")

    if yaml is not None:
        raw = yaml.safe_load(content)
    else:
        import json

        try:
            raw = json.loads(content)
        except Exception:
            raw = _simple_yaml_parse(content)

    if raw is None or not isinstance(raw, dict):
        raise ValueError(
            f"Configuration file '{path}' must define a non-empty YAML mapping, got {type(raw).__name__}"
        )

    required_set = set(_REQUIRED_SECTIONS)
    provided_set = set(raw.keys())

    missing_sections = sorted(required_set - provided_set)
    if missing_sections:
        raise ValueError(f"Missing required config sections: {missing_sections}")

    unknown_sections = sorted(provided_set - required_set)
    if unknown_sections:
        raise ValueError(f"Unknown top-level config sections: {unknown_sections}")

    return Settings(
        model=_build_section(ModelConfig, "model", raw["model"]),
        budget=_build_section(BudgetConfig, "budget", raw["budget"]),
        cap=_build_section(CapConfig, "cap", raw["cap"]),
        retrieval=_build_section(RetrievalConfig, "retrieval", raw["retrieval"]),
        compress=_build_section(CompressConfig, "compress", raw["compress"]),
        p1=_build_section(P1Config, "p1", raw["p1"]),
        run=_build_section(RunConfig, "run", raw["run"]),
        data=_build_section(DataConfig, "data", raw["data"]),
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
        if not line[0].isspace() and line.endswith(":"):
            current_section = line[:-1].strip()
            result[current_section] = {}
        elif line[0].isspace() and ":" in line and current_section:
            key, val = line.split(":", 1)
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if val.lstrip("-").isdigit() and val.count("-") <= 1:
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
        else:
            raise ValueError(f"Invalid YAML syntax in fallback parser: {line!r}")

    return result
