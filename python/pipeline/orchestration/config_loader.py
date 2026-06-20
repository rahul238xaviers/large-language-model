"""Configuration loader with layered resolution.

Resolution order (later layers win):
  1. configs/base/*.yaml  – base defaults, merged alphabetically
  2. configs/profiles/<profile>.yaml  – hardware/environment profile overrides
  3. Environment variables  – runtime overrides via known mapping
  4. CLI flags  – passed in by calling code after load()
"""

import os
from pathlib import Path
from typing import Any

import yaml


class ConfigLoader:
    """Loads and merges layered YAML configuration."""

    def __init__(self, config_dir: Path = Path("configs")) -> None:
        self.config_dir = Path(config_dir)

    def load(self, profile: str = "local-dev") -> dict[str, Any]:
        """Return the fully resolved configuration dict."""
        base = self._load_base()
        profile_cfg = self._load_profile(profile)
        merged = self._deep_merge(base, profile_cfg)
        resolved = self._apply_env_overrides(merged)

        # Git repository root (parent of python/)
        repo_root = Path(__file__).resolve().parents[3]

        # Resolve runs_dir relative to git root
        runs_dir_str = resolved.get("runs_dir", "runs")
        runs_dir = Path(runs_dir_str)
        if not runs_dir.is_absolute():
            resolved["runs_dir"] = str(repo_root / runs_dir)

        # Resolve cache_dir relative to git root
        download_cfg = resolved.setdefault("download", {})
        cache_dir_str = download_cfg.get("cache_dir", "data/datasets")
        cache_dir = Path(cache_dir_str)
        if not cache_dir.is_absolute():
            download_cfg["cache_dir"] = str(repo_root / cache_dir)

        return resolved

    # ------------------------------------------------------------------ #
    # Private helpers                                                       #
    # ------------------------------------------------------------------ #

    def _load_base(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        base_dir = self.config_dir / "base"
        if not base_dir.exists():
            return result
        for yaml_file in sorted(base_dir.glob("*.yaml")):
            with open(yaml_file) as fh:
                content = yaml.safe_load(fh) or {}
            result = self._deep_merge(result, content)
        return result

    def _load_profile(self, profile: str) -> dict[str, Any]:
        profile_path = self.config_dir / "profiles" / f"{profile}.yaml"
        if not profile_path.exists():
            return {}
        with open(profile_path) as fh:
            return yaml.safe_load(fh) or {}

    def _apply_env_overrides(self, cfg: dict[str, Any]) -> dict[str, Any]:
        env_map: dict[str, tuple[str, ...]] = {
            "HF_TOKEN": ("download", "hf_token"),
            "PIPELINE_CACHE_DIR": ("download", "cache_dir"),
            "PIPELINE_RUNS_DIR": ("runs_dir",),
            "OLLAMA_BASE_URL": ("generation", "ollama_base_url"),
            "OLLAMA_MODEL": ("generation", "ollama_model"),
        }
        for env_var, path in env_map.items():
            val = os.environ.get(env_var)
            if val is not None:
                self._set_nested(cfg, path, val)
        return cfg

    @staticmethod
    def _set_nested(d: dict[str, Any], path: tuple[str, ...], value: Any) -> None:
        for key in path[:-1]:
            d = d.setdefault(key, {})
        d[path[-1]] = value

    @staticmethod
    def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        result = dict(base)
        for key, val in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(val, dict):
                result[key] = ConfigLoader._deep_merge(result[key], val)
            else:
                result[key] = val
        return result
