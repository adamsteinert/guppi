"""Stable Text UI for GLaDOS 2.0."""

import asyncio
from pathlib import Path
import sys
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

[yellow]Text Input:[/yellow]
• Type in the input field at the bottom
• Press [bold]Enter[/bold] to send your message
• Text input is the default mode

[yellow]Key Bindings:[/yellow]
• [bold]Cmd+L[/bold] - Start/Stop voice listening
• [bold]Cmd+I[/bold] - Interrupt current response
• [bold]Cmd+Q[/bold] - Quit application
• [bold]Cmd+H[/bold] - Show this help

[yellow]Audio Controls:[/yellow]
• [bold]Cmd+M[/bold] - Toggle microphone mute
• [bold]Cmd+S[/bold] - Toggle speaker mute

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
        Binding("cmd+q", "quit", "Quit"),
        Binding("cmd+h", "help", "Help"),
        Binding("cmd+l", "toggle_listening", "Listen"),
        Binding("cmd+i", "interrupt", "Interrupt"),
        Binding("cmd+m", "toggle_microphone", "Mic"),
        Binding("cmd+s", "toggle_speaker", "Speaker"),
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

    #input_container {
        height: 3;
        margin-top: 1;
    }

    #message_input {
        width: 1fr;
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
    
    def __init__(self):
        super().__init__()
        self._event_bus = EventBus()
        self._state_manager = StateManager(self._event_bus)
        self._conversation_log: Optional[ConversationLog] = None
        self._status_display: Optional[StatusDisplay] = None
        self._message_input: Optional[Input] = None

        # Managers will be injected by main app
        self._audio_manager = None
        self._llm_manager = None
        
        # Register for events
        self._event_bus.subscribe(EventType.STATE_CHANGED, self._on_state_changed)
        self._event_bus.subscribe(EventType.MESSAGE_RECEIVED, self._on_message_received)
        self._event_bus.subscribe(EventType.AUDIO_STATUS_CHANGED, self._on_audio_status_changed)
        
    def compose(self) -> ComposeResult:
        """Compose the main UI layout."""
        yield Header(show_clock=True)

        with Container(id="main_container"):
            # Main conversation area
            with Vertical(id="conversation_area"):
                self._conversation_log = ConversationLog(id="conversation_log")
                yield self._conversation_log

                # Text input at bottom
                with Container(id="input_container"):
                    self._message_input = Input(
                        placeholder="Type your message here and press Enter...",
                        id="message_input"
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
        
    def on_mount(self) -> None:
        """Initialize the application after mounting."""
        logger.info("GLaDOS 2.0 UI starting...")

        # Initialize state
        self._state_manager.set_state(AppState.INITIALIZING)

        # Focus the input field by default
        if self._message_input:
            self._message_input.focus()

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
            
    def _on_state_changed(self, event_data: dict) -> None:
        """Handle state changes from the state manager."""
        new_state = event_data.get("new_state")
        if self._status_display and new_state:
            self._status_display.update_status(new_state.value)
            
    def _on_message_received(self, event_data: dict) -> None:
        """Handle new messages."""
        role = event_data.get("role", "unknown")
        content = event_data.get("content", "")
        
        if self._conversation_log:
            self._conversation_log.add_message(role, content)
            
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
            
    def action_help(self) -> None:
        """Show help screen."""
        self.push_screen(HelpScreen())
        
    def action_toggle_listening(self) -> None:
        """Toggle listening state."""
        current_state = self._state_manager.get_state()
        
        if current_state == AppState.IDLE:
            self._state_manager.set_state(AppState.LISTENING)
            self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "listening"})
            logger.info("Started listening...")
        elif current_state == AppState.LISTENING:
            self._state_manager.set_state(AppState.IDLE)  
            self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "ready"})
            logger.info("Stopped listening...")
            
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle Enter key in input field."""
        if event.input.id == "message_input":
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

                logger.info(f"User sent text message: {message}")
            
    def action_interrupt(self) -> None:
        """Interrupt current operation."""
        logger.info("Interrupt requested")
        self._state_manager.set_state(AppState.IDLE)
        self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "interrupted"})
        
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
        
    def action_quit(self) -> None:
        """Quit the application cleanly."""
        logger.info("Shutting down GLaDOS 2.0...")
        self._state_manager.set_state(AppState.SHUTTING_DOWN)
        self.exit()


def main():
    """Entry point for the GLaDOS 2.0 UI."""
    try:
        app = GladosUI()
        app.run()
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)


if __name__ == "__main__":
    main()