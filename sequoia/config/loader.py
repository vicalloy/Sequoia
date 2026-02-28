"""Configuration loading utilities."""

import json
from pathlib import Path

from .schema import Config

# Global cache for loaded configuration
_config: Config | None = None


def get_config_path() -> Path:
    """Get the default configuration file path."""
    return Path.home() / ".sequoia" / "config.json"


def load_config(config_path: Path | None = None) -> Config:
    """
    Load configuration from file or create default.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Loaded configuration object.
    """
    path = config_path or get_config_path()

    if path.exists():
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            config = Config.model_validate(data)
            return config
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Warning: Failed to load config from {path}: {e}")
            print("Using default configuration.")

    config = Config()
    return config


def get_config(config_path: Path | None = None) -> Config:
    """
    Get configuration from cache or load if not yet loaded.

    Args:
        config_path: Optional path to config file. Uses default if not provided.

    Returns:
        Configuration object
    """
    if _config is not None:
        return _config

    # If not cached, load the config
    return load_config(config_path)


def save_config(config: Config | None = None, config_path: Path | None = None) -> None:
    """
    Save configuration to file.

    Args:
        config: Configuration to save. If None, uses cached config or default.
        config_path: Optional path to save to. Uses default if not provided.
    """
    path = config_path or get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)

    # If no config provided, use cached config or create a default
    if config is None:
        config = _config if _config is not None else Config()

    data = config.model_dump(by_alias=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
