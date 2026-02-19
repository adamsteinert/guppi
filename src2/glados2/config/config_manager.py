"""Configuration management for GLaDOS 2.0."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
import os
import re
import yaml

from dotenv import load_dotenv
from loguru import logger


def _substitute_env_vars(value: Any) -> Any:
    """Recursively substitute ${VAR} patterns with environment variables."""
    if isinstance(value, str):
        # Match ${VAR_NAME} pattern
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, value)
        for var_name in matches:
            env_value = os.getenv(var_name)
            if env_value is not None:
                value = value.replace(f'${{{var_name}}}', env_value)
            else:
                logger.warning(f"Environment variable {var_name} not found")
        return value
    elif isinstance(value, dict):
        return {k: _substitute_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_substitute_env_vars(item) for item in value]
    return value


@dataclass
class AudioConfig:
    """Audio configuration settings."""
    sample_rate: int = 16000
    vad_threshold: float = 0.8
    buffer_size_ms: int = 800
    pause_limit_ms: int = 1000
    microphone_muted: bool = False
    speaker_muted: bool = False
    volume: float = 1.0
    interruptible: bool = True


@dataclass 
class LLMConfig:
    """LLM configuration settings."""
    provider: str = "ollama"
    model: str = "llama3.2"
    completion_url: str = "http://localhost:11434/api/generate"
    api_key: Optional[str] = None
    system_prompt: str = "You are GLaDOS, the sarcastic AI from Portal."
    max_tokens: int = 2048
    temperature: float = 0.7
    streaming: bool = True


@dataclass
class TTSConfig:
    """Text-to-speech configuration settings."""
    voice: str = "glados"
    speed: float = 1.0
    quality: str = "high"
    # Long response summarization
    summarize_long_responses: bool = True
    max_speech_words: int = 150  # ~30 seconds at typical speech rate


@dataclass
class UIConfig:
    """UI configuration settings."""
    theme: str = "dark"
    show_debug: bool = False
    auto_scroll: bool = True
    font_size: str = "medium"
    compact_mode: bool = False


@dataclass
class MCPServerConfig:
    """Configuration for a single MCP server."""
    name: str = ""
    command: str = ""
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class ToolsConfig:
    """Tools configuration settings."""
    enabled: bool = True
    mcp_servers: List[MCPServerConfig] = field(default_factory=list)
    tool_timeout_seconds: float = 30.0
    max_tool_iterations: int = 5


@dataclass
class GladosConfig:
    """Main configuration for GLaDOS 2.0."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

    # Global settings
    wake_word: Optional[str] = None
    wake_word_variants: str = ""
    announcement: Optional[str] = None
    log_level: str = "INFO"

    # Salutation and valediction (spoken on start/stop)
    salutation: Optional[str] = None  # Greeting spoken when app starts
    valediction: Optional[str] = None  # Farewell spoken when app stops
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "GladosConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)

        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()

        # Load .env file from project root
        dotenv_path = Path.cwd() / '.env'
        if dotenv_path.exists():
            load_dotenv(dotenv_path)
            logger.debug(f"Loaded environment variables from {dotenv_path}")

        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}

            # Substitute environment variables
            data = _substitute_env_vars(data)

            logger.info(f"Loaded configuration from {config_path}")
            return cls.from_dict(data)

        except Exception as e:
            logger.error(f"Error loading config from {config_path}: {e}")
            logger.info("Using default configuration")
            return cls()
            
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GladosConfig":
        """Create configuration from dictionary."""
        config = cls()
        
        # Load audio settings
        if "audio" in data:
            audio_data = data["audio"]
            config.audio = AudioConfig(
                sample_rate=audio_data.get("sample_rate", config.audio.sample_rate),
                vad_threshold=audio_data.get("vad_threshold", config.audio.vad_threshold),
                buffer_size_ms=audio_data.get("buffer_size_ms", config.audio.buffer_size_ms),
                pause_limit_ms=audio_data.get("pause_limit_ms", config.audio.pause_limit_ms),
                microphone_muted=audio_data.get("microphone_muted", config.audio.microphone_muted),
                speaker_muted=audio_data.get("speaker_muted", config.audio.speaker_muted),
                volume=audio_data.get("volume", config.audio.volume),
                interruptible=audio_data.get("interruptible", config.audio.interruptible),
            )
            
        # Load LLM settings
        if "llm" in data:
            llm_data = data["llm"]
            config.llm = LLMConfig(
                provider=llm_data.get("provider", config.llm.provider),
                model=llm_data.get("model", config.llm.model),
                completion_url=llm_data.get("completion_url", config.llm.completion_url),
                api_key=llm_data.get("api_key", config.llm.api_key),
                system_prompt=llm_data.get("system_prompt", config.llm.system_prompt),
                max_tokens=llm_data.get("max_tokens", config.llm.max_tokens),
                temperature=llm_data.get("temperature", config.llm.temperature),
                streaming=llm_data.get("streaming", config.llm.streaming),
            )
            
        # Load TTS settings
        if "tts" in data:
            tts_data = data["tts"]
            config.tts = TTSConfig(
                voice=tts_data.get("voice", config.tts.voice),
                speed=tts_data.get("speed", config.tts.speed),
                quality=tts_data.get("quality", config.tts.quality),
                summarize_long_responses=tts_data.get("summarize_long_responses", config.tts.summarize_long_responses),
                max_speech_words=tts_data.get("max_speech_words", config.tts.max_speech_words),
            )
            
        # Load UI settings
        if "ui" in data:
            ui_data = data["ui"]
            config.ui = UIConfig(
                theme=ui_data.get("theme", config.ui.theme),
                show_debug=ui_data.get("show_debug", config.ui.show_debug),
                auto_scroll=ui_data.get("auto_scroll", config.ui.auto_scroll),
                font_size=ui_data.get("font_size", config.ui.font_size),
                compact_mode=ui_data.get("compact_mode", config.ui.compact_mode),
            )

        # Load tools settings
        if "tools" in data:
            tools_data = data["tools"]
            mcp_servers = []
            for server_data in tools_data.get("mcp_servers", []):
                mcp_servers.append(MCPServerConfig(
                    name=server_data.get("name", ""),
                    command=server_data.get("command", ""),
                    args=server_data.get("args", []),
                    env=server_data.get("env", {}),
                    enabled=server_data.get("enabled", True),
                ))
            config.tools = ToolsConfig(
                enabled=tools_data.get("enabled", config.tools.enabled),
                mcp_servers=mcp_servers,
                tool_timeout_seconds=tools_data.get("tool_timeout_seconds", config.tools.tool_timeout_seconds),
                max_tool_iterations=tools_data.get("max_tool_iterations", config.tools.max_tool_iterations),
            )

        # Load global settings
        config.wake_word = data.get("wake_word", config.wake_word)
        config.wake_word_variants = data.get("wake_word_variants", config.wake_word_variants)
        config.announcement = data.get("announcement", config.announcement)
        config.log_level = data.get("log_level", config.log_level)
        config.salutation = data.get("salutation", config.salutation)
        config.valediction = data.get("valediction", config.valediction)

        return config
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "audio": {
                "sample_rate": self.audio.sample_rate,
                "vad_threshold": self.audio.vad_threshold,
                "buffer_size_ms": self.audio.buffer_size_ms,
                "pause_limit_ms": self.audio.pause_limit_ms,
                "microphone_muted": self.audio.microphone_muted,
                "speaker_muted": self.audio.speaker_muted,
                "volume": self.audio.volume,
                "interruptible": self.audio.interruptible,
            },
            "llm": {
                "provider": self.llm.provider,
                "model": self.llm.model,
                "completion_url": self.llm.completion_url,
                "api_key": self.llm.api_key,
                "system_prompt": self.llm.system_prompt,
                "max_tokens": self.llm.max_tokens,
                "temperature": self.llm.temperature,
                "streaming": self.llm.streaming,
            },
            "tts": {
                "voice": self.tts.voice,
                "speed": self.tts.speed,
                "quality": self.tts.quality,
                "summarize_long_responses": self.tts.summarize_long_responses,
                "max_speech_words": self.tts.max_speech_words,
            },
            "ui": {
                "theme": self.ui.theme,
                "show_debug": self.ui.show_debug,
                "auto_scroll": self.ui.auto_scroll,
                "font_size": self.ui.font_size,
            },
            "tools": {
                "enabled": self.tools.enabled,
                "tool_timeout_seconds": self.tools.tool_timeout_seconds,
                "max_tool_iterations": self.tools.max_tool_iterations,
                "mcp_servers": [
                    {
                        "name": s.name,
                        "command": s.command,
                        "args": s.args,
                        "env": s.env,
                        "enabled": s.enabled,
                    }
                    for s in self.tools.mcp_servers
                ],
            },
            "wake_word": self.wake_word,
            "wake_word_variants": self.wake_word_variants,
            "announcement": self.announcement,
            "log_level": self.log_level,
            "salutation": self.salutation,
            "valediction": self.valediction,
        }
        
    def save_to_yaml(self, config_path: str | Path) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {config_path}")
        except Exception as e:
            logger.error(f"Error saving config to {config_path}: {e}")


class ConfigManager:
    """Manages configuration loading, validation, and updates."""
    
    def __init__(self, config_path: Optional[str | Path] = None):
        self._config_path = Path(config_path) if config_path else Path("configs/glados2_config.yaml")
        self._config = self._load_config()
        
    def _load_config(self) -> GladosConfig:
        """Load configuration from file or create default."""
        if self._config_path.exists():
            return GladosConfig.from_yaml(self._config_path)
        else:
            logger.info(f"Creating default config at {self._config_path}")
            config = GladosConfig()
            config.save_to_yaml(self._config_path)
            return config
            
    def get_config(self) -> GladosConfig:
        """Get current configuration."""
        return self._config
        
    def update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with new values."""
        # TODO: Implement configuration updates
        logger.info("Configuration update requested")
        
    def reload_config(self) -> None:
        """Reload configuration from file."""
        self._config = self._load_config()
        logger.info("Configuration reloaded")