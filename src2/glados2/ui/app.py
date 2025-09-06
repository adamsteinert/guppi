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
from textual.widgets import Footer, Header, Label, RichLog, Static

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

[yellow]Key Bindings:[/yellow]
• [bold]Space[/bold] - Start/Stop listening
• [bold]Enter[/bold] - Send typed message  
• [bold]Ctrl+C[/bold] - Interrupt current response
• [bold]Q[/bold] - Quit application
• [bold]?[/bold] - Show this help

[yellow]Audio Controls:[/yellow]
• [bold]M[/bold] - Toggle microphone mute
• [bold]S[/bold] - Toggle speaker mute
• [bold]V[/bold] - Adjust volume (not yet implemented)

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
        Binding("q", "quit", "Quit"),
        Binding("question_mark", "help", "Help", key_display="?"),
        Binding("space", "toggle_listening", "Listen"),
        Binding("enter", "send_message", "Send"),
        Binding("ctrl+c", "interrupt", "Interrupt"),
        Binding("m", "toggle_microphone", "Mic"),
        Binding("s", "toggle_speaker", "Speaker"),
    ]
    
    CSS = """
    #main_container {
        layout: horizontal;
    }
    
    #conversation_area {
        width: 3fr;
        border: solid $primary;
        margin: 1;
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
        self._state_manager = StateManager()
        self._conversation_log: Optional[ConversationLog] = None
        self._status_display: Optional[StatusDisplay] = None
        
        # Register for events
        self._event_bus.subscribe(EventType.STATE_CHANGED, self._on_state_changed)
        self._event_bus.subscribe(EventType.MESSAGE_RECEIVED, self._on_message_received)
        self._event_bus.subscribe(EventType.AUDIO_STATUS_CHANGED, self._on_audio_status_changed)
        
    def compose(self) -> ComposeResult:
        """Compose the main UI layout."""
        yield Header(show_clock=True)
        
        with Container(id="main_container"):
            # Main conversation area
            with Container(id="conversation_area"):
                self._conversation_log = ConversationLog(id="conversation_log")
                yield self._conversation_log
                
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
            
    def action_send_message(self) -> None:
        """Send a typed message (placeholder)."""
        # TODO: Implement text input for manual message sending
        if self._conversation_log:
            self._conversation_log.add_message("user", "[Typed message functionality not yet implemented]")
            
    def action_interrupt(self) -> None:
        """Interrupt current operation."""
        logger.info("Interrupt requested")
        self._state_manager.set_state(AppState.IDLE)
        self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "interrupted"})
        
    def action_toggle_microphone(self) -> None:
        """Toggle microphone mute."""
        # TODO: Implement microphone control
        logger.info("Microphone toggle requested")
        
    def action_toggle_speaker(self) -> None:
        """Toggle speaker mute."""
        # TODO: Implement speaker control  
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