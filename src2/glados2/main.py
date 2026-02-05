"""Main application entry point for GLaDOS 2.0."""

import asyncio
from pathlib import Path
import sys
from typing import Optional

from loguru import logger

from .core.event_bus import EventBus, EventType
from .core.state_manager import StateManager, AppState
from .audio.audio_manager import AudioManager
from .llm.llm_manager import LLMManager, LLMProvider
from .config.config_manager import ConfigManager, GladosConfig
from .tools.tool_manager import ToolManager
from .ui.app import GladosUI


class GladosApp:
    """
    Main GLaDOS 2.0 application orchestrator.
    
    This class ties together all the components with proper dependency injection
    and lifecycle management.
    """
    
    def __init__(self, config_path: Optional[str | Path] = None):
        # Load configuration
        self._config_manager = ConfigManager(config_path)
        self._config = self._config_manager.get_config()
        
        # Setup logging
        logger.remove()
        logger.add(
            sys.stderr, 
            level=self._config.log_level,
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | <level>{level: <8}</level> | {name}:{function}:{line} - {message}"
        )
        
        # Core components
        self._event_bus = EventBus()
        self._state_manager = StateManager(self._event_bus)
        
        # Managers
        audio_config = {
            'sample_rate': self._config.audio.sample_rate,
            'chunk_size': 1024,
            'microphone_muted': self._config.audio.microphone_muted,
            'speaker_muted': self._config.audio.speaker_muted,
            'volume': self._config.audio.volume,
            'vad_threshold': self._config.audio.vad_threshold,
            'min_speech_duration_ms': 250,
            'min_silence_duration_ms': self._config.audio.pause_limit_ms,
            'voice': self._config.tts.voice,
            'tts_sample_rate': 22050,
        }
        self._audio_manager = AudioManager(self._event_bus, self._state_manager, audio_config)

        # Initialize tool manager if tools are enabled
        self._tool_manager: Optional[ToolManager] = None
        if self._config.tools.enabled:
            self._tool_manager = ToolManager(self._event_bus, self._config.tools)

        # Create LLM manager with tool manager
        self._llm_manager = LLMManager(
            self._event_bus,
            self._state_manager,
            tool_manager=self._tool_manager
        )

        # Configure LLM manager from config
        self._llm_manager.configure_provider(
            LLMProvider[self._config.llm.provider.upper()],
            model=self._config.llm.model,
            completion_url=self._config.llm.completion_url,
            api_key=self._config.llm.api_key,
            temperature=self._config.llm.temperature,
            max_tokens=self._config.llm.max_tokens,
            max_tool_iterations=self._config.tools.max_tool_iterations
        )
        self._llm_manager.set_system_prompt(self._config.llm.system_prompt)

        # UI (optional - can run headless)
        self._ui: Optional[GladosUI] = None
        
        # Subscribe to shutdown events
        self._event_bus.subscribe(EventType.SHUTDOWN_REQUESTED, self._handle_shutdown)
        
    def run_ui(self) -> None:
        """Run the application with UI."""
        logger.info("Starting GLaDOS 2.0 with UI...")

        try:
            # Initialize tool manager (connect to MCP servers)
            if self._tool_manager:
                asyncio.run(self._tool_manager.initialize())

            # Create UI with shared event bus (MUST be passed to constructor
            # so event subscriptions are registered on the shared bus)
            self._ui = GladosUI(event_bus=self._event_bus)
            # Inject other dependencies
            self._ui._state_manager = self._state_manager
            self._ui._audio_manager = self._audio_manager
            self._ui._llm_manager = self._llm_manager
            # Pass salutation config so UI can speak it on startup
            self._ui._salutation = self._config.salutation

            # Run the UI
            self._ui.run()

        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        finally:
            self._cleanup()
            
    async def run_headless(self) -> None:
        """Run the application without UI (for server/embedded use)."""
        logger.info("Starting GLaDOS 2.0 in headless mode...")

        try:
            # Initialize tool manager (connect to MCP servers)
            if self._tool_manager:
                await self._tool_manager.initialize()

            # Initialize components
            self._state_manager.set_state(AppState.IDLE)

            # Main event loop
            while self._state_manager.get_state() != AppState.SHUTTING_DOWN:
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        finally:
            await self._cleanup_async()
            
    def _handle_shutdown(self, event_data: dict) -> None:
        """Handle shutdown request."""
        logger.info("Shutdown requested")
        self._state_manager.set_state(AppState.SHUTTING_DOWN)
        
    def _cleanup(self) -> None:
        """Clean up resources (sync version for UI mode)."""
        logger.info("Cleaning up GLaDOS 2.0...")

        # Cancel any ongoing operations
        if hasattr(self._audio_manager, 'interrupt_playback'):
            self._audio_manager.interrupt_playback()

        if hasattr(self._llm_manager, 'cancel_current_request'):
            self._llm_manager.cancel_current_request()

        # Clean up tool manager
        if self._tool_manager:
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._tool_manager.cleanup())
                loop.close()
            except Exception as e:
                logger.error(f"Error cleaning up tool manager: {e}")

        # Speak valediction if configured
        if self._config.valediction:
            self._speak_valediction()

        logger.info("Cleanup completed")

    async def _cleanup_async(self) -> None:
        """Clean up resources (async version for headless mode)."""
        logger.info("Cleaning up GLaDOS 2.0...")

        # Cancel any ongoing operations
        if hasattr(self._audio_manager, 'interrupt_playback'):
            self._audio_manager.interrupt_playback()

        if hasattr(self._llm_manager, 'cancel_current_request'):
            self._llm_manager.cancel_current_request()

        # Clean up tool manager
        if self._tool_manager:
            await self._tool_manager.cleanup()

        logger.info("Cleanup completed")

    def _speak_valediction(self) -> None:
        """Speak the valediction message (blocking)."""
        try:
            import sounddevice as sd

            logger.info(f"Speaking valediction: {self._config.valediction}")

            # Get the TTS processor from audio manager
            tts = self._audio_manager._tts

            # Synthesize audio synchronously using a new event loop
            # (the main loop may be closed at this point)
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                audio = loop.run_until_complete(
                    tts.synthesize_speech(self._config.valediction)
                )
                loop.close()
            except Exception as e:
                logger.error(f"Error synthesizing valediction: {e}")
                return

            if audio is not None and len(audio) > 0:
                # Get correct sample rate
                info = tts.get_model_info()
                if tts.voice == "glados":
                    sample_rate = info.get("glados_sample_rate", 22050)
                else:
                    sample_rate = info.get("kokoro_sample_rate", 24000)

                # Play audio blocking (wait for completion)
                sd.play(audio, samplerate=sample_rate)
                sd.wait()
                logger.info("Valediction spoken")
            else:
                logger.warning("Failed to synthesize valediction")

        except Exception as e:
            logger.error(f"Error speaking valediction: {e}")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="GLaDOS 2.0 Voice Assistant")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/glados2_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without UI"
    )
    
    args = parser.parse_args()
    
    app = GladosApp(config_path=args.config)
    
    if args.headless:
        asyncio.run(app.run_headless())
    else:
        app.run_ui()


if __name__ == "__main__":
    main()