# GLaDOS 2.0 Interaction Guide

## Overview
GLaDOS 2.0 uses a dual-mode interface optimized for terminal use: **Command Mode** and **Text Input Mode**.

## Command Mode (Default)

When you start GLaDOS, you're in **Command Mode**. Use single keystrokes to control the application:

### Main Commands
- **P** - Enter text input mode (to type messages)
- **L** - Start/Stop voice listening
- **I** - Interrupt current response
- **T** - Toggle debug view
- **M** - Toggle microphone mute
- **S** - Toggle speaker mute
- **H** - Show help screen
- **Q** - Quit application

## Text Input Mode

Press **P** in command mode to enter text input mode:

### In Text Input Mode:
- Type your message normally (all keys work as expected)
- Press **Enter** to send your message to GLaDOS
- Press **Esc** to cancel and return to command mode
- After sending a message, you automatically return to command mode

### Visual Indicator:
- Input field is **dimmed** in command mode
- Input field is **bright** and active in text input mode
- Placeholder text shows current mode

## Usage Examples

### Text Interaction Workflow
1. Start GLaDOS: `./run_glados2.sh`
2. You're in command mode (input is dimmed)
3. Press **P** to enter text input mode
4. Type your message
5. Press **Enter** to send (automatically returns to command mode)
6. GLaDOS will respond with streaming text
7. Press **P** again for your next message

### Voice Interaction Workflow
1. In command mode, press **L** to start listening
2. Speak your message
3. Press **L** again to stop listening (or wait for automatic detection)
4. GLaDOS will transcribe and respond

### Mixed Mode
You can freely switch between text and voice:
- Press **P** for text messages
- Press **L** for voice messages
- Press **I** to interrupt long responses
- Press **M** to mute microphone when not using voice

## Status Indicators

The right panel shows real-time status:
- **Audio**: Current audio state (Ready/Listening/Speaking)
- **LLM**: Connection status to Gemini
- **Status**: Current application state

## Debug View

Press **T** in command mode to toggle between conversation and debug view:
- **Conversation View** (default): Shows your chat with GLaDOS
- **Debug View**: Shows real-time event logging including:
  - State changes (INITIALIZING → IDLE → LISTENING, etc.)
  - Message events (user input, LLM responses)
  - Audio status changes (listening, processing, speaking)
  - LLM response chunks (streaming text)
  - Service initialization events
  - User actions (keyboard shortcuts, mode changes)

The debug view is useful for:
- Troubleshooting issues
- Understanding the event flow
- Monitoring LLM streaming responses
- Tracking audio pipeline state

## Tips

1. **Command Mode First**: You start in command mode - use single keys for commands
2. **P for Typing**: Press 'P' to type a message, Enter to send, Esc to cancel
3. **Quick Help**: Press 'H' anytime to see all key bindings
4. **Safe Interruption**: Press 'I' to stop long responses
5. **Voice Optional**: Voice features are completely optional (press 'L')
6. **Conversation Flow**: Both text and voice messages appear in the same log
7. **Debug Mode**: Press 'T' to see what's happening under the hood
8. **Terminal Friendly**: No modifier keys needed - works great in any terminal

## Running the Application

```bash
# Start with default Gemini config
./run_glados2.sh

# Start with custom config
./run_glados2.sh --config path/to/config.yaml

# Start in headless mode (no UI)
./run_glados2.sh --headless
```

## Architecture Benefits

- **Event-Driven**: Text and voice both use the same event bus
- **Unified Processing**: LLM processes all messages the same way
- **Consistent UI**: Same conversation log for all interaction modes
- **Non-Blocking**: Async design prevents UI freezing
- **Gemini Integration**: Real-time streaming responses from Gemini
