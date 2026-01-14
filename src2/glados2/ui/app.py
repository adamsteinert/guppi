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

        # Managers will be injected by main app
        self._audio_manager = None
        self._llm_manager = None

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

        # Start background services (stubbed for now)
        self.call_later(self._initialize_services)
        
    def _initialize_services(self) -> None:
        """Initialize background services."""
        logger.info("Initializing services...")

        # TODO: Initialize audio, LLM, and other services
        # For now, just simulate successful initialization
        self._state_manager.set_state(AppState.IDLE)

        if self._conversation_log:
            self._conversation_log.add_message("system", "GLaDOS 2.0 initialized successfully")

        logger.info("All services initialized successfully")
            
    def _on_state_changed(self, event_data: dict) -> None:
        """Handle state changes from the state manager."""
        new_state = event_data.get("new_state")
        old_state = event_data.get("old_state")
        if self._status_display and new_state:
            self._status_display.update_status(new_state.value)

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
        elif status == "processing":
            audio_widget.update("Audio: [yellow]Processing[/]")
        elif status == "speaking":
            audio_widget.update("Audio: [red]Speaking[/]")
        else:
            audio_widget.update("Audio: [green]Ready[/]")

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

        if current_state == AppState.IDLE:
            self._state_manager.set_state(AppState.LISTENING)
            self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "listening"})
            logger.info("Started listening")
        elif current_state == AppState.LISTENING:
            self._state_manager.set_state(AppState.IDLE)
            self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "ready"})
            logger.info("Stopped listening")
            
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

    def action_quit(self) -> None:
        """Quit the application cleanly."""
        logger.info("Shutting down GLaDOS 2.0...")

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