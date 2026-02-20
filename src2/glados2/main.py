"""Main application entry point for GLaDOS 2.0."""

import asyncio
from pathlib import Path
import sys
from typing import Optional

from loguru import logger

from .core.event_bus import EventBus, EventType
from .core.state_manager import StateManager, AppState
from .core.worker_loop import WorkerLoop
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

        # Persistent worker loop for LLM + MCP tool operations.
        # MCP connections and LLM requests must share the same event loop
        # so that loop-bound MCP stdio sessions remain usable during tool execution.
        self._worker_loop = WorkerLoop(name="llm-tools")
        self._worker_loop.start()

        # Create LLM manager with tool manager and worker loop
        self._llm_manager = LLMManager(
            self._event_bus,
            self._state_manager,
            tool_manager=self._tool_manager,
            worker_loop=self._worker_loop,
        )

        # Configure LLM manager from config
        self._llm_manager.configure_provider(
            LLMProvider[self._config.llm.provider.upper()],
            model=self._config.llm.model,
            completion_url=self._config.llm.completion_url,
            api_key=self._config.llm.api_key,
            temperature=self._config.llm.temperature,
            max_tokens=self._config.llm.max_tokens,
            max_tool_iterations=self._config.tools.max_tool_iterations,
            # TTS summarization settings
            summarize_long_responses=self._config.tts.summarize_long_responses,
            max_speech_words=self._config.tts.max_speech_words,
            # Gemini thinking settings
            thinking_enabled=self._config.llm.thinking_enabled,
            thinking_level=self._config.llm.thinking_level,
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
            # Create UI with shared event bus (MUST be passed to constructor
            # so event subscriptions are registered on the shared bus)
            self._ui = GladosUI(event_bus=self._event_bus)
            # Inject other dependencies
            self._ui._state_manager = self._state_manager
            self._ui._audio_manager = self._audio_manager
            self._ui._llm_manager = self._llm_manager
            # Pass tool manager and worker loop so the UI can initialize
            # tools asynchronously after Textual takes control of the
            # terminal. Initializing before Textual starts blocks the main
            # thread, which corrupts the display and drops keyboard input.
            self._ui._tool_manager = self._tool_manager
            self._ui._worker_loop = self._worker_loop
            # Pass salutation config so UI can speak it on startup
            self._ui._salutation = self._config.salutation
            # Apply compact mode from config
            self._ui._compact_mode = self._config.ui.compact_mode

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
            # Initialize audio models
            self._audio_manager.initialize_models()

            # Initialize tools on the worker loop
            if self._tool_manager:
                await self._worker_loop.run_await(self._tool_manager.initialize())

            # Initialize components
            self._state_manager.set_state(AppState.IDLE)

            # Main event loop
            while self._state_manager.get_state() != AppState.SHUTTING_DOWN:
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("Application interrupted by user")
        finally:
            await self._cleanup_async()

    async def run_debug(self) -> None:
        """
        Run in debug mode with terminal input loop.

        This mode:
        - Uses simple terminal input (no TUI)
        - Does not listen for voice input
        - Outputs debug info to stdout/stderr
        - Still plays audio responses via TTS
        """
        print("\n" + "=" * 60)
        print("GLaDOS 2.0 - Debug Mode")
        print("=" * 60)
        print("Type your messages and press Enter to send.")
        print("Type 'quit' or 'exit' to stop.")
        print("Type 'clear' to clear conversation history.")
        print("Type 'tools' to list available tools.")
        print("=" * 60 + "\n")

        try:
            # Initialize audio models
            self._audio_manager.initialize_models()

            # Initialize tools on the worker loop (same loop where LLM
            # requests and tool calls will execute)
            if self._tool_manager:
                print("Initializing tools...")
                await self._worker_loop.run_await(self._tool_manager.initialize())
                tool_names = self._tool_manager.get_tool_names()
                if tool_names:
                    print(f"Tools available: {', '.join(tool_names)}")
                else:
                    print("No tools configured.")
                print()

            # Set state to idle
            self._state_manager.set_state(AppState.IDLE)

            # Subscribe to events for debug output
            self._event_bus.subscribe(
                EventType.LLM_RESPONSE_CHUNK,
                lambda d: print(d.get("chunk", ""), end="", flush=True)
            )
            self._event_bus.subscribe(
                EventType.LLM_RESPONSE_COMPLETED,
                lambda d: print("\n")  # Newline after response
            )
            self._event_bus.subscribe(
                EventType.TOOL_EXECUTION_STARTED,
                lambda d: print(f"\n[Tool: {d.get('tool_name')}] Executing...")
            )
            self._event_bus.subscribe(
                EventType.TOOL_EXECUTION_COMPLETED,
                lambda d: print(f"[Tool: {d.get('tool_name')}] Completed in {d.get('duration_ms', 0):.0f}ms")
            )
            self._event_bus.subscribe(
                EventType.TOOL_EXECUTION_ERROR,
                lambda d: print(f"[Tool: {d.get('tool_name')}] Error: {d.get('error')}")
            )

            # Speak salutation if configured
            if self._config.salutation:
                print(f"GLaDOS: {self._config.salutation}")
                await self._audio_manager.synthesize_and_play(self._config.salutation)

            # Main input loop
            while True:
                try:
                    # Get user input
                    user_input = await asyncio.get_event_loop().run_in_executor(
                        None, lambda: input("You: ")
                    )
                    user_input = user_input.strip()

                    if not user_input:
                        continue

                    # Handle special commands
                    if user_input.lower() in ("quit", "exit"):
                        print("\nShutting down...")
                        break

                    if user_input.lower() == "clear":
                        self._llm_manager.clear_conversation_history()
                        print("Conversation history cleared.\n")
                        continue

                    if user_input.lower() == "tools":
                        if self._tool_manager and self._tool_manager.has_tools():
                            print("\nAvailable tools:")
                            for name in self._tool_manager.get_tool_names():
                                print(f"  - {name}")
                            print()
                        else:
                            print("No tools available.\n")
                        continue

                    # Send message to LLM
                    print("\nGLaDOS: ", end="", flush=True)

                    # Publish user message to trigger LLM response
                    # The LLM manager will handle the response and publish MESSAGE_RECEIVED
                    # which triggers TTS playback via audio manager
                    self._event_bus.publish(EventType.MESSAGE_RECEIVED, {
                        "role": "user",
                        "content": user_input
                    })

                    # Wait for response to complete
                    while self._state_manager.get_state() == AppState.CALLING_LLM:
                        await asyncio.sleep(0.05)

                    # Wait a bit more for TTS to start if needed
                    await asyncio.sleep(0.1)

                    # Wait for any audio playback to complete
                    while self._state_manager.get_state() in (
                        AppState.GENERATING_TTS,
                        AppState.PLAYING_AUDIO
                    ):
                        await asyncio.sleep(0.1)

                except EOFError:
                    # Handle Ctrl+D
                    print("\nShutting down...")
                    break

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
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

        # Clean up tool manager on the worker loop (same loop where it was initialized)
        if self._tool_manager and self._worker_loop.is_running:
            try:
                future = self._worker_loop.run(self._tool_manager.cleanup())
                future.result(timeout=10)
            except Exception as e:
                logger.error(f"Error cleaning up tools: {e}")

        self._worker_loop.stop()

        # Speak valediction if configured
        if self._config.valediction:
            self._speak_valediction()

        logger.info("Cleanup completed")

    async def _cleanup_async(self) -> None:
        """Clean up resources (async version for headless/debug mode)."""
        logger.info("Cleaning up GLaDOS 2.0...")

        # Cancel any ongoing operations
        if hasattr(self._audio_manager, 'interrupt_playback'):
            self._audio_manager.interrupt_playback()

        if hasattr(self._llm_manager, 'cancel_current_request'):
            self._llm_manager.cancel_current_request()

        # Clean up tool manager on the worker loop
        if self._tool_manager and self._worker_loop.is_running:
            await self._worker_loop.run_await(self._tool_manager.cleanup())

        self._worker_loop.stop()

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


def _say(text: str, config_path: str, outfile: Optional[str] = None) -> None:
    """Synthesize speech from text, optionally saving to a file.

    Args:
        text: Text to speak.
        config_path: Path to configuration YAML.
        outfile: If provided, save audio to this WAV file instead of playing.
    """
    import struct
    import wave

    import numpy as np
    import sounddevice as sd

    from .audio.tts_processor import TTSProcessor
    from .config.config_manager import ConfigManager

    config = ConfigManager(config_path).get_config()
    # Resolve model dir: walk up from cwd until we find models/TTS
    search = Path.cwd()
    model_dir = "models/TTS"
    for _ in range(5):
        candidate = search / "models" / "TTS"
        if candidate.exists():
            model_dir = str(candidate)
            break
        search = search.parent
    tts = TTSProcessor(voice=config.tts.voice, model_dir=model_dir)

    audio = asyncio.run(tts.synthesize_speech(text))
    if audio is None or len(audio) == 0:
        print("Error: TTS synthesis failed.")
        return

    # Determine sample rate for the active voice
    info = tts.get_model_info()
    if tts.voice == "glados":
        sr = info.get("glados_sample_rate", tts.sample_rate)
    else:
        sr = info.get("kokoro_sample_rate", tts.sample_rate)

    if outfile:
        # Normalize float32 audio to int16 WAV
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / peak
        audio_int16 = (audio * 32767).astype(np.int16)
        with wave.open(outfile, "w") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sr)
            wf.writeframes(struct.pack(f"<{len(audio_int16)}h", *audio_int16))
        print(f"Saved to {outfile}")
    else:
        sd.play(audio, samplerate=sr)
        sd.wait()


def main():
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="GLaDOS 2.0 Voice Assistant")
    # Top-level args so "python -m glados2.main --config ..." works without subcommand
    parser.add_argument(
        "--config", type=str, default="configs/glados2_config.yaml",
        help="Path to configuration file",
    )
    top_mode = parser.add_mutually_exclusive_group()
    top_mode.add_argument("--headless", action="store_true", help="Run without UI")
    top_mode.add_argument("--debug", action="store_true", help="Debug mode (terminal input, audio output)")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # --- run ---
    run_parser = subparsers.add_parser("run", help="Run the voice assistant")
    run_parser.add_argument(
        "--config", type=str, default="configs/glados2_config.yaml",
        help="Path to configuration file",
    )
    run_mode = run_parser.add_mutually_exclusive_group()
    run_mode.add_argument("--headless", action="store_true", help="Run without UI")
    run_mode.add_argument("--debug", action="store_true", help="Debug mode (terminal input, audio output)")

    # --- say ---
    say_parser = subparsers.add_parser("say", help="Speak text through the speaker")
    say_parser.add_argument("text", type=str, help="Text to speak")
    say_parser.add_argument(
        "--config", type=str, default="configs/glados2_config.yaml",
        help="Path to configuration file",
    )

    # --- saytofile ---
    stf_parser = subparsers.add_parser("saytofile", help="Save speech to a WAV file")
    stf_parser.add_argument("text", type=str, help="Text to speak")
    stf_parser.add_argument(
        "--outfile", type=str, default="output.wav",
        help="Output WAV file path (default: output.wav)",
    )
    stf_parser.add_argument(
        "--config", type=str, default="configs/glados2_config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Default to "run" when no subcommand is given (backwards-compatible)
    if args.command is None or args.command == "run":
        config = getattr(args, "config", "configs/glados2_config.yaml")
        app = GladosApp(config_path=config)
        if getattr(args, "headless", False):
            asyncio.run(app.run_headless())
        elif getattr(args, "debug", False):
            asyncio.run(app.run_debug())
        else:
            app.run_ui()
    elif args.command == "say":
        _say(args.text, args.config)
    elif args.command == "saytofile":
        _say(args.text, args.config, outfile=args.outfile)


if __name__ == "__main__":
    main()