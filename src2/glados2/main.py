"""Main application entry point for GLaDOS 2.0."""

import asyncio
from pathlib import Path
import sys
from typing import Optional

from loguru import logger

from .core.event_bus import EventBus, EventType
from .core.state_manager import StateManager, AppState
from .audio.audio_manager import AudioManager
from .llm.llm_manager import LLMManager
from .config.config_manager import ConfigManager, GladosConfig
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
        self._audio_manager = AudioManager(self._event_bus, self._state_manager)
        self._llm_manager = LLMManager(self._event_bus, self._state_manager)
        
        # UI (optional - can run headless)
        self._ui: Optional[GladosUI] = None
        
        # Subscribe to shutdown events
        self._event_bus.subscribe(EventType.SHUTDOWN_REQUESTED, self._handle_shutdown)
        
    def run_ui(self) -> None:
        """Run the application with UI."""
        logger.info("Starting GLaDOS 2.0 with UI...")
        
        try:
            # Create and inject dependencies into UI
            self._ui = GladosUI()
            self._ui._event_bus = self._event_bus
            self._ui._state_manager = self._state_manager
            
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
            # Initialize components
            self._state_manager.set_state(AppState.IDLE)
            
            # Main event loop
            while self._state_manager.get_state() != AppState.SHUTTING_DOWN:
                await asyncio.sleep(0.1)
                
        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        finally:
            self._cleanup()
            
    def _handle_shutdown(self, event_data: dict) -> None:
        """Handle shutdown request."""
        logger.info("Shutdown requested")
        self._state_manager.set_state(AppState.SHUTTING_DOWN)
        
    def _cleanup(self) -> None:
        """Clean up resources."""
        logger.info("Cleaning up GLaDOS 2.0...")
        
        # Cancel any ongoing operations
        if hasattr(self._audio_manager, 'interrupt_playback'):
            self._audio_manager.interrupt_playback()
            
        if hasattr(self._llm_manager, 'cancel_current_request'):
            self._llm_manager.cancel_current_request()
            
        logger.info("Cleanup completed")


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