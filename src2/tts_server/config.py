"""Configuration for the TTS API server."""

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import yaml
from dotenv import load_dotenv
from loguru import logger


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} patterns with environment variables."""
    if isinstance(value, str):
        pattern = r"\$\{([^}]+)\}"
        matches = re.findall(pattern, value)
        for var_name in matches:
            env_value = os.getenv(var_name)
            if env_value is not None:
                value = value.replace(f"${{{var_name}}}", env_value)
            else:
                logger.warning(f"Environment variable {var_name} not found")
        return value
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


@dataclass
class ServerConfig:
    """TTS API server configuration."""

    host: str = "0.0.0.0"
    port: int = 8000
    model_dir: str = "models/TTS"
    default_voice: str = "af_alloy"
    default_model: str = "kokoro-v1"

    # Auth
    api_key: Optional[str] = None
    api_key_required: bool = False

    # Limits
    max_text_length: int = 5000
    request_timeout_seconds: float = 60.0

    # Logging
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "ServerConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()

        # Load .env file from project root
        dotenv_path = Path.cwd() / ".env"
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
            logger.debug(f"Loaded environment variables from {dotenv_path}")

        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}

            data = _substitute_env_vars(data)
            logger.info(f"Loaded TTS server config from {config_path}")
            return cls.from_dict(data)
        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            return cls()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ServerConfig":
        """Create configuration from dictionary."""
        config = cls()
        for key in (
            "host",
            "port",
            "model_dir",
            "default_voice",
            "default_model",
            "api_key",
            "api_key_required",
            "max_text_length",
            "request_timeout_seconds",
            "log_level",
        ):
            if key in data:
                setattr(config, key, data[key])
        return config


def load_config(config_path: Optional[str] = None) -> ServerConfig:
    """Load server configuration from file or defaults."""
    if config_path:
        return ServerConfig.from_yaml(config_path)
    # Try default paths
    for path in ["configs/tts_server_config.yaml", "src2/configs/tts_server_config.yaml"]:
        if Path(path).exists():
            return ServerConfig.from_yaml(path)
    logger.info("No config file found, using defaults")
    return ServerConfig()
