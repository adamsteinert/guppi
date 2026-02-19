"""Stable Text UI for GLaDOS 2.0."""

import asyncio
from pathlib import Path
import sys
import threading
from typing import ClassVar, Optional

from loguru import logger
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.screen import ModalScreen, Screen
from textual.widgets import Footer, Header, Input, Label, RichLog, Static

from ..core.event_bus import EventBus, EventType
from ..core.state_manager import StateManager, AppState


class ConversationLog(RichLog):
    """A stable conversation log widget that properly handles updates."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.wrap = True
        self.markup = True
        
    def add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation log with proper formatting."""
        if role == "user":
            self.write(f"[blue]User:[/] {content}")
        elif role == "assistant":
            self.write(f"[green]GLaDOS:[/] {content}")
        elif role == "system":
            self.write(f"[dim]System:[/] {content}")
        else:
            self.write(f"[yellow]{role}:[/] {content}")


class StatusDisplay(Static):
    """Widget to display current application status."""
    
    def __init__(self, **kwargs):
        super().__init__("Status: Initializing...", **kwargs)
        self._current_status = "initializing"
    
    def update_status(self, status: str, details: str = "") -> None:
        """Update the status display."""
        self._current_status = status
        status_text = f"Status: {status.title()}"
        if details:
            status_text += f" - {details}"
        self.update(status_text)


class HelpScreen(ModalScreen[None]):
    """Help screen with key bindings and usage information."""
    
    BINDINGS: ClassVar = [("escape", "app.pop_screen", "Close")]
    
    def compose(self) -> ComposeResult:
        help_text = """
[bold]GLaDOS 2.0 - Help[/bold]

[yellow]Command Mode (Default):[/yellow]
Use single keys to control GLaDOS:
• [bold]P[/bold] - Enter text input mode
• [bold]L[/bold] - Start/Stop voice listening
• [bold]I[/bold] - Interrupt current response
• [bold]T[/bold] - Toggle debug view
• [bold]M[/bold] - Toggle microphone mute
• [bold]S[/bold] - Toggle speaker mute
• [bold]C[/bold] - Toggle compact mode
• [bold]H[/bold] - Show this help
• [bold]Q[/bold] - Quit application

[yellow]Text Input Mode:[/yellow]
• Press [bold]P[/bold] in command mode to start typing
• Type your message normally
• Press [bold]Enter[/bold] to send message
• Press [bold]Esc[/bold] to return to command mode

[yellow]Status Indicators:[/yellow]
• [green]Listening[/green] - Waiting for your voice
• [blue]Processing[/blue] - Understanding your input
• [red]Speaking[/red] - GLaDOS is responding
• [yellow]Idle[/yellow] - Ready for interaction

Press [bold]Esc[/bold] to close this help screen.
        """
        with Container(id="help_dialog"):
            yield Static(help_text)
    
    def on_mount(self) -> None:
        dialog = self.query_one("#help_dialog")
        dialog.border_title = "Help"
        dialog.border_subtitle = "Press Esc to close"


class GladosUI(App[None]):
    """
    Stable Text UI for GLaDOS 2.0.
    
    Key improvements over original:
    - Proper event-driven architecture
    - Clean separation of UI and business logic
    - Stable state management
    - Better error handling and recovery
    """
    
    TITLE = "GLaDOS 2.0"
    SUB_TITLE = "Stable Voice Assistant"
    
    BINDINGS: ClassVar = [
        Binding("q", "quit", "Quit", show=True),
        Binding("h", "help", "Help", show=True),
        Binding("l", "toggle_listening", "Listen", show=True),
        Binding("i", "interrupt", "Interrupt", show=True),
        Binding("m", "toggle_microphone", "Mic", show=True),
        Binding("s", "toggle_speaker", "Speaker", show=True),
        Binding("t", "toggle_debug", "Debug", show=True),
        Binding("p", "enter_text_mode", "Type", show=True),
        Binding("c", "toggle_compact", "Compact", show=True),
    ]
    
    CSS = """
    #main_container {
        layout: horizontal;
    }

    #conversation_area {
        width: 3fr;
        border: solid $primary;
        margin: 1;
        height: 1fr;
    }

    #conversation_log {
        height: 1fr;
    }

    #debug_log {
        height: 1fr;
        display: none;
    }

    #input_container {
        height: 3;
        margin-top: 1;
    }

    #message_input {
        width: 1fr;
    }

    #message_input:disabled {
        opacity: 0.6;
    }

    #status_area {
        width: 1fr;
        border: solid $secondary;
        margin: 1;
    }

    #status_display {
        height: 3;
        border: solid $accent;
        margin: 1;
    }

    #system_info {
        border: solid $accent;
        margin: 1;
    }

    #help_dialog {
        width: 80%;
        height: 80%;
        border: solid $primary;
    }

    #compact_container {
        display: none;
        height: auto;
        margin: 0 1;
        border: solid $secondary;
    }

    #compact_status_line {
        height: 1;
        padding: 0 1;
    }

    #compact_device_line {
        height: 1;
        padding: 0 1;
    }
    """
    
    def __init__(self, event_bus: Optional[EventBus] = None):
        super().__init__()
        # Use provided EventBus or create a new one
        # IMPORTANT: For events to work across components, pass in a shared EventBus!
        self._event_bus = event_bus if event_bus is not None else EventBus()
        self._state_manager = StateManager(self._event_bus)
        self._conversation_log: Optional[ConversationLog] = None
        self._debug_log: Optional[RichLog] = None
        self._status_display: Optional[StatusDisplay] = None
        self._message_input: Optional[Input] = None
        self._show_debug = False
        self._logger_sink_id = None
        self._text_input_mode = False  # Track if we're in text input mode
        self._compact_mode = False  # Compact mode (may be overridden by config injection)

        # Tracked state for compact display updates
        self._current_state_text = "Idle"
        self._current_audio_text = "[green]Ready[/]"

        # Managers will be injected by main app
        self._audio_manager = None
        self._llm_manager = None
        self._tool_manager = None  # Initialized on Textual's event loop

        # Salutation to speak on startup (injected by main app)
        self._salutation: Optional[str] = None

        # Register for events
        self._event_bus.subscribe(EventType.STATE_CHANGED, self._on_state_changed)
        self._event_bus.subscribe(EventType.MESSAGE_RECEIVED, self._on_message_received)
        self._event_bus.subscribe(EventType.AUDIO_STATUS_CHANGED, self._on_audio_status_changed)
        self._event_bus.subscribe(EventType.LLM_RESPONSE_COMPLETED, self._on_llm_response)
        self._event_bus.subscribe(EventType.LLM_RESPONSE_CHUNK, self._on_llm_response_chunk)

    @property
    def event_bus(self) -> EventBus:
        """Get the event bus for publishing events to the UI."""
        return self._event_bus

    def display_message(self, role: str, content: str) -> None:
        """
        Directly display a message in the conversation log.

        This is a convenience method that bypasses the event bus.
        Use this for testing or when you have direct access to the UI.

        Args:
            role: "user", "assistant", or "system"
            content: The message content
        """
        if self._conversation_log and content:
            self._conversation_log.add_message(role, content)
            logger.debug(f"Direct message displayed: {role}: {content[:30]}...")

    def display_response(self, content: str) -> None:
        """Convenience method to display an assistant response."""
        self.display_message("assistant", content)

    def compose(self) -> ComposeResult:
        """Compose the main UI layout."""
        yield Header(show_clock=True)

        # Compact mode container (hidden by default, toggled with C key)
        with Container(id="compact_container"):
            yield Static(
                "[bold]●[/] Idle     Audio: [green]Ready[/]     LLM: [green]Connected[/]",
                id="compact_status_line",
            )
            yield Static(
                "Mic: [dim]Loading...[/]     Speaker: [dim]Loading...[/]",
                id="compact_device_line",
            )

        with Container(id="main_container"):
            # Main conversation area
            with Vertical(id="conversation_area"):
                self._conversation_log = ConversationLog(id="conversation_log")
                yield self._conversation_log

                # Debug log (hidden by default)
                self._debug_log = RichLog(id="debug_log", wrap=True, markup=True)
                yield self._debug_log

                # Text input at bottom
                with Container(id="input_container"):
                    self._message_input = Input(
                        placeholder="Press 'P' to type a message...",
                        id="message_input",
                        disabled=True
                    )
                    yield self._message_input

            # Status and controls area
            with Container(id="status_area"):
                self._status_display = StatusDisplay(id="status_display")
                yield self._status_display

                with Container(id="system_info"):
                    yield Static("Audio: [green]Ready[/]", id="audio_status")
                    yield Static("LLM: [green]Connected[/]", id="llm_status")
                    yield Static("Memory: [green]Active[/]", id="memory_status")
                    yield Static("Mic: [dim]Loading...[/]", id="mic_device")
                    yield Static("Speaker: [dim]Loading...[/]", id="speaker_device")

        yield Footer()
        
    def _logger_sink(self, message: str) -> None:
        """Custom logger sink that writes to the debug log widget."""
        if self._debug_log:
            msg = message.rstrip()
            # Check if we're on the main thread or a background thread
            # call_from_thread only works from background threads
            if threading.current_thread() is threading.main_thread():
                # On main thread - call directly
                try:
                    self._debug_log.write(msg)
                except Exception:
                    pass  # Silently ignore if widget not ready
            else:
                # On background thread - use call_from_thread
                try:
                    self.call_from_thread(self._debug_log.write, msg)
                except Exception:
                    pass  # Silently ignore if app not ready

    def on_mount(self) -> None:
        """Initialize the application after mounting."""
        # Remove default logger handlers to prevent terminal output
        logger.remove()

        # Add custom logger sink to capture log output (all levels to debug widget)
        self._logger_sink_id = logger.add(
            self._logger_sink,
            format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
            colorize=True,
            level="TRACE"  # Capture all log levels (TRACE and above)
        )

        logger.info("GLaDOS 2.0 UI starting...")

        # Initialize state
        self._state_manager.set_state(AppState.INITIALIZING)

        # Start in command mode (don't focus input field)
        self._text_input_mode = False
        if self._message_input:
            self._message_input.disabled = True

        # Apply compact mode if set by config injection
        if self._compact_mode:
            self._apply_compact_mode()

        # Start background services (stubbed for now)
        self.call_later(self._initialize_services)
        
    def _initialize_services(self) -> None:
        """Initialize background services.

        Tool manager initialization happens here (on Textual's event loop)
        so that MCP subprocess connections are bound to the running loop.
        """
        logger.info("Initializing services...")

        # Initialize tool manager on Textual's event loop so MCP
        # connections (async streams) live on the same loop used later
        # for tool execution.
        if self._tool_manager:
            asyncio.create_task(self._initialize_tools())
        else:
            self._finish_initialization()

    async def _initialize_tools(self) -> None:
        """Initialize tool manager asynchronously on the running event loop."""
        try:
            await self._tool_manager.initialize()
            tool_names = self._tool_manager.get_tool_names()
            if tool_names:
                logger.info(f"Tools available: {', '.join(tool_names)}")
                if self._conversation_log:
                    self._conversation_log.add_message(
                        "system", f"Tools loaded: {', '.join(tool_names)}"
                    )
            else:
                logger.info("No tools discovered from MCP servers")
        except Exception as e:
            logger.error(f"Failed to initialize tools: {e}")

        self._finish_initialization()

    def _finish_initialization(self) -> None:
        """Complete initialization after tools are ready."""
        self._state_manager.set_state(AppState.IDLE)

        if self._conversation_log:
            self._conversation_log.add_message("system", "GLaDOS 2.0 initialized successfully")

        logger.info("All services initialized successfully")

        # Update audio device info
        self._update_audio_device_info()

        # Speak salutation if configured
        if self._salutation and self._audio_manager:
            logger.info(f"Speaking salutation: {self._salutation}")
            asyncio.create_task(self._audio_manager.synthesize_and_play(self._salutation))

    def _update_audio_device_info(self) -> None:
        """Query and display current audio input/output devices."""
        try:
            import sounddevice as sd

            # Get default devices
            default_input = sd.query_devices(kind='input')
            default_output = sd.query_devices(kind='output')

            # Get device names (truncate if too long)
            mic_name = default_input.get('name', 'Unknown') if default_input else 'None'
            speaker_name = default_output.get('name', 'Unknown') if default_output else 'None'

            # Truncate long names
            max_len = 25
            if len(mic_name) > max_len:
                mic_name = mic_name[:max_len-3] + "..."
            if len(speaker_name) > max_len:
                speaker_name = speaker_name[:max_len-3] + "..."

            # Update UI widgets
            mic_widget = self.query_one("#mic_device", Static)
            speaker_widget = self.query_one("#speaker_device", Static)

            mic_widget.update(f"Mic: [cyan]{mic_name}[/]")
            speaker_widget.update(f"Speaker: [cyan]{speaker_name}[/]")

            # Update compact device line
            if self._compact_mode:
                self._update_compact_devices()

            logger.debug(f"Audio devices - Input: {mic_name}, Output: {speaker_name}")

        except Exception as e:
            logger.warning(f"Could not query audio devices: {e}")
            try:
                mic_widget = self.query_one("#mic_device", Static)
                speaker_widget = self.query_one("#speaker_device", Static)
                mic_widget.update("Mic: [red]Error[/]")
                speaker_widget.update("Speaker: [red]Error[/]")
            except Exception:
                pass
            
    def _on_state_changed(self, event_data: dict) -> None:
        """Handle state changes from the state manager."""
        new_state = event_data.get("new_state")
        old_state = event_data.get("old_state")
        if self._status_display and new_state:
            self._status_display.update_status(new_state.value)

        # Track state for compact display
        if new_state:
            self._current_state_text = new_state.value.replace("_", " ").title()
            if self._compact_mode:
                self._update_compact_status()

        # Log state transition
        if old_state:
            logger.debug(f"State transition: {old_state.value} → {new_state.value}")
        else:
            logger.debug(f"State set to: {new_state.value}")
            
    def _on_message_received(self, event_data: dict) -> None:
        """Handle new messages (user, assistant, system)."""
        role = event_data.get("role", "unknown")
        content = event_data.get("content", "")
        source = event_data.get("source", "unknown")

        logger.debug(f"_on_message_received: role={role}, source={source}, len={len(content)}")

        # Skip assistant messages - they're handled by _on_llm_response to avoid duplicates
        if role == "assistant":
            logger.debug("Skipping assistant message (handled by _on_llm_response)")
            return

        # Skip user messages from text_input - already added directly in on_input_submitted
        if role == "user" and source == "text_input":
            logger.debug("Skipping text_input user message (already displayed)")
            return

        if self._conversation_log and content:
            self._conversation_log.add_message(role, content)
            logger.debug(f"Message added to log: {role}: {content[:30]}...")
        else:
            logger.warning(f"Could not add message: log={self._conversation_log is not None}, content={bool(content)}")
            
    def _on_audio_status_changed(self, event_data: dict) -> None:
        """Handle audio status changes."""
        status = event_data.get("status", "unknown")
        audio_widget = self.query_one("#audio_status", Static)
        if status == "listening":
            audio_widget.update("Audio: [blue]Listening[/]")
            self._current_audio_text = "[blue]Listening[/]"
        elif status == "processing":
            audio_widget.update("Audio: [yellow]Processing[/]")
            self._current_audio_text = "[yellow]Processing[/]"
        elif status == "speaking":
            audio_widget.update("Audio: [red]Speaking[/]")
            self._current_audio_text = "[red]Speaking[/]"
        else:
            audio_widget.update("Audio: [green]Ready[/]")
            self._current_audio_text = "[green]Ready[/]"

        # Update compact view if active
        if self._compact_mode:
            self._update_compact_status()

        # Log audio status change
        logger.debug(f"Audio status changed to: {status}")

    def _on_llm_response(self, event_data: dict) -> None:
        """Handle completed LLM responses."""
        # LLM manager publishes with "full_response" key
        content = event_data.get("full_response", "") or event_data.get("content", "")
        logger.debug(f"_on_llm_response called with content length: {len(content)}")

        if content and self._conversation_log:
            # Try direct call first (works if on main thread)
            try:
                self._conversation_log.add_message("assistant", content)
                logger.debug("Added assistant message directly")
            except Exception as e:
                # Fall back to call_from_thread if needed
                logger.debug(f"Direct call failed ({e}), trying call_from_thread")
                try:
                    self.call_from_thread(
                        self._conversation_log.add_message,
                        "assistant",
                        content
                    )
                except Exception as e2:
                    logger.error(f"Failed to add message: {e2}")

        if content:
            logger.info(f"LLM response displayed: {content[:50]}...")

    def _on_llm_response_chunk(self, event_data: dict) -> None:
        """Handle streaming LLM response chunks."""
        is_first = event_data.get("is_first", False)
        is_last = event_data.get("is_last", False)

        # For streaming, we could update incrementally
        # For now, we'll just log chunks and wait for complete response
        if is_first:
            logger.debug("LLM response streaming started")
        if is_last:
            logger.debug("LLM response streaming completed")

    def action_help(self) -> None:
        """Show help screen."""
        self.push_screen(HelpScreen())
        
    def action_toggle_listening(self) -> None:
        """Toggle listening state."""
        current_state = self._state_manager.get_state()

        if current_state == AppState.LISTENING:
            # Stop listening and process any collected audio
            logger.info("Stopping listening - processing collected audio")
            self._event_bus.publish(EventType.LISTENING_STOPPED, {"reason": "user_request"})
            self._state_manager.set_state(AppState.IDLE)
            self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "ready"})
            logger.info("Stopped listening")

        elif current_state == AppState.PLAYING_AUDIO:
            # Cancel playback, then start listening
            logger.info("Cancelling playback to start listening")
            if self._audio_manager:
                self._audio_manager.interrupt_playback()
            # Transition through IDLE so the playback path can clean up,
            # then immediately into LISTENING.
            self._state_manager.set_state(AppState.IDLE)
            self._state_manager.set_state(AppState.LISTENING)
            self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "listening"})
            if self._conversation_log:
                self._conversation_log.add_message("system", "Listening...")
            logger.info("Started listening (interrupted playback)")

        elif current_state in (AppState.IDLE, AppState.ERROR):
            self._state_manager.set_state(AppState.LISTENING)
            self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "listening"})
            if self._conversation_log:
                self._conversation_log.add_message("system", "Listening...")
            logger.info("Started listening")

        else:
            # PROCESSING_AUDIO, CALLING_LLM, GENERATING_TTS — ignore,
            # the pipeline is busy and listening will resume when the
            # user presses 'l' again after it returns to IDLE.
            logger.info(f"Cannot toggle listening in state {current_state.value}")
            
    def on_key(self, event: events.Key) -> None:
        """Handle key presses globally."""
        # Handle Escape in text input mode
        if event.key == "escape" and self._text_input_mode:
            self._exit_text_mode()
            event.prevent_default()
            event.stop()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input field."""
        if event.input.id == "message_input" and self._text_input_mode:
            message = event.value.strip()
            if message and self._message_input:
                # Clear the input
                self._message_input.value = ""

                # Add to conversation log
                if self._conversation_log:
                    self._conversation_log.add_message("user", message)

                # Publish message to event bus for LLM processing
                self._event_bus.publish(EventType.MESSAGE_RECEIVED, {
                    "role": "user",
                    "content": message,
                    "source": "text_input"
                })

                logger.info(f"User text message: {message}")

                # Exit text input mode after sending
                self._exit_text_mode()

    def action_interrupt(self) -> None:
        """Interrupt current operation and stop audio playback."""
        logger.info("Interrupt requested - stopping all audio")

        # Publish interrupt event for other components to handle
        self._event_bus.publish(EventType.INTERRUPT_REQUESTED, {})

        # Stop audio playback immediately using sounddevice
        try:
            import sounddevice as sd
            sd.stop()
            logger.debug("Audio playback stopped via sounddevice")
        except Exception as e:
            logger.warning(f"Could not stop audio playback: {e}")

        # Update state
        self._state_manager.set_state(AppState.IDLE)
        self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "ready"})

        # Show feedback in conversation log
        if self._conversation_log:
            self._conversation_log.add_message("system", "Interrupted")
        
    def action_toggle_microphone(self) -> None:
        """Toggle microphone mute."""
        if self._audio_manager:
            current = self._audio_manager.is_microphone_muted()
            self._audio_manager.set_microphone_muted(not current)
            status = "muted" if not current else "unmuted"
            if self._conversation_log:
                self._conversation_log.add_message("system", f"Microphone {status}")
        logger.info("Microphone toggle requested")

    def action_toggle_speaker(self) -> None:
        """Toggle speaker mute."""
        if self._audio_manager:
            current = self._audio_manager.is_speaker_muted()
            self._audio_manager.set_speaker_muted(not current)
            status = "muted" if not current else "unmuted"
            if self._conversation_log:
                self._conversation_log.add_message("system", f"Speaker {status}")
        logger.info("Speaker toggle requested")
        
    def action_enter_text_mode(self) -> None:
        """Enter text input mode."""
        if not self._text_input_mode:
            self._text_input_mode = True
            if self._message_input:
                self._message_input.disabled = False
                self._message_input.focus()
                # Update placeholder to show mode
                self._message_input.placeholder = "Type message (Enter to send, Esc to cancel)..."
            logger.info("Entered text input mode")

    def _exit_text_mode(self) -> None:
        """Exit text input mode and return to command mode."""
        if self._text_input_mode:
            self._text_input_mode = False
            if self._message_input:
                self._message_input.value = ""  # Clear any typed text
                self._message_input.disabled = True
                self._message_input.blur()
                self._message_input.placeholder = "Press 'P' to type a message..."
            logger.info("Exited text input mode")

    def action_toggle_debug(self) -> None:
        """Toggle between conversation and debug view."""
        self._show_debug = not self._show_debug

        if self._show_debug:
            # Show debug, hide conversation
            if self._conversation_log:
                self._conversation_log.styles.display = "none"
            if self._debug_log:
                self._debug_log.styles.display = "block"
            logger.info("Switched to debug view")
        else:
            # Show conversation, hide debug
            if self._conversation_log:
                self._conversation_log.styles.display = "block"
            if self._debug_log:
                self._debug_log.styles.display = "none"
            logger.info("Switched to conversation view")

    def action_toggle_compact(self) -> None:
        """Toggle between compact and full UI mode."""
        self._compact_mode = not self._compact_mode
        self._apply_compact_mode()
        logger.info(f"Switched to {'compact' if self._compact_mode else 'full'} mode")

    def _apply_compact_mode(self) -> None:
        """Apply the current compact mode state to the UI."""
        main = self.query_one("#main_container")
        compact = self.query_one("#compact_container")
        if self._compact_mode:
            main.styles.display = "none"
            compact.styles.display = "block"
            self._update_compact_status()
            self._update_compact_devices()
        else:
            main.styles.display = "block"
            compact.styles.display = "none"

    def _update_compact_status(self) -> None:
        """Rebuild the compact status line from current state."""
        try:
            widget = self.query_one("#compact_status_line", Static)
            widget.update(
                f"[bold]●[/] {self._current_state_text}     "
                f"Audio: {self._current_audio_text}     "
                f"LLM: [green]Connected[/]"
            )
        except Exception:
            pass

    def _update_compact_devices(self) -> None:
        """Rebuild the compact device line from current device info."""
        try:
            mic_text = self.query_one("#mic_device", Static).renderable
            speaker_text = self.query_one("#speaker_device", Static).renderable
            compact_device = self.query_one("#compact_device_line", Static)
            compact_device.update(f"{mic_text}     {speaker_text}")
        except Exception:
            pass

    def action_quit(self) -> None:
        """Quit the application cleanly."""
        logger.info("Shutting down GLaDOS 2.0...")

        # Clean up tool manager while Textual's event loop is still running,
        # since MCP connections are bound to this loop.
        if self._tool_manager:
            asyncio.create_task(self._cleanup_and_exit())
        else:
            self._finalize_quit()

    async def _cleanup_and_exit(self) -> None:
        """Clean up tool manager then exit."""
        try:
            await self._tool_manager.cleanup()
        except Exception as e:
            logger.error(f"Error cleaning up tool manager: {e}")
        self._finalize_quit()

    def _finalize_quit(self) -> None:
        """Final quit steps after async cleanup."""
        # Remove the custom logger sink
        if self._logger_sink_id is not None:
            logger.remove(self._logger_sink_id)
            self._logger_sink_id = None

        self._state_manager.set_state(AppState.SHUTTING_DOWN)
        self.exit()


def main():
    """Entry point for the GLaDOS 2.0 UI."""
    # Create a shared event bus that can be used by other components
    shared_event_bus = EventBus()

    try:
        # Pass the shared event bus to the UI
        app = GladosUI(event_bus=shared_event_bus)

        # Example: Other components can use the same event bus
        # llm_manager = LLMManager(event_bus=shared_event_bus)
        # audio_manager = AudioManager(event_bus=shared_event_bus)

        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()