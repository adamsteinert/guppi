"""Configuration management for GLaDOS 2.0."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

from loguru import logger


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
    # TODO: Add more TTS-specific settings


@dataclass
class UIConfig:
    """UI configuration settings."""
    theme: str = "dark"
    show_debug: bool = False
    auto_scroll: bool = True
    font_size: str = "medium"


@dataclass
class GladosConfig:
    """Main configuration for GLaDOS 2.0."""
    audio: AudioConfig = field(default_factory=AudioConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    
    # Global settings
    wake_word: Optional[str] = None
    wake_word_variants: str = ""
    announcement: Optional[str] = None
    log_level: str = "INFO"
    
    @classmethod
    def from_yaml(cls, config_path: str | Path) -> "GladosConfig":
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls()
            
        try:
            with open(config_path, 'r') as f:
                data = yaml.safe_load(f) or {}
                
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
            )
            
        # Load UI settings
        if "ui" in data:
            ui_data = data["ui"]
            config.ui = UIConfig(
                theme=ui_data.get("theme", config.ui.theme),
                show_debug=ui_data.get("show_debug", config.ui.show_debug),
                auto_scroll=ui_data.get("auto_scroll", config.ui.auto_scroll),
                font_size=ui_data.get("font_size", config.ui.font_size),
            )
            
        # Load global settings
        config.wake_word = data.get("wake_word", config.wake_word)
        config.wake_word_variants = data.get("wake_word_variants", config.wake_word_variants)
        config.announcement = data.get("announcement", config.announcement)
        config.log_level = data.get("log_level", config.log_level)
        
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
            },
            "ui": {
                "theme": self.ui.theme,
                "show_debug": self.ui.show_debug,
                "auto_scroll": self.ui.auto_scroll,
                "font_size": self.ui.font_size,
            },
            "wake_word": self.wake_word,
            "wake_word_variants": self.wake_word_variants,
            "announcement": self.announcement,
            "log_level": self.log_level,
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