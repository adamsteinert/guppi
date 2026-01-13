# GLaDOS 2.0 Text Input Mode

## Overview
GLaDOS 2.0 now supports text input as the default interaction mode, with voice commands accessible via keyboard shortcuts.

## Text Input
- **Default Mode**: Text input is active by default when you start the application
- **How to Use**: Simply type your message in the input field at the bottom of the screen
- **Send Message**: Press `Enter` to send your message to GLaDOS
- **Input Field**: Automatically focused on startup for immediate typing

## Updated Key Bindings

All control commands now use **Cmd+** (Command key) combinations to avoid conflicts with text input:

### Main Controls
- **Cmd+Q** - Quit application
- **Cmd+H** - Show help screen
- **Cmd+I** - Interrupt current response
- **Cmd+T** - Toggle debug view

### Voice Controls
- **Cmd+L** - Start/Stop voice listening mode
- **Cmd+M** - Toggle microphone mute
- **Cmd+S** - Toggle speaker mute

### Navigation
- **Esc** - Close help screen (when open)
- **Enter** - Send typed message (in text input field)

## Usage Examples

### Text Interaction (Default)
1. Start GLaDOS: `./run_glados2.sh`
2. Type your message in the input field
3. Press `Enter` to send
4. GLaDOS will respond with streaming text
5. Continue the conversation by typing more messages

### Voice Interaction
1. Press `Cmd+L` to start voice listening
2. Speak your message
3. Press `Cmd+L` again to stop listening (or wait for automatic detection)
4. GLaDOS will transcribe and respond

### Mixed Mode
You can freely switch between text and voice:
- Type messages when convenient
- Use `Cmd+L` for voice when hands-free
- Interrupt long responses with `Cmd+I`
- Mute microphone with `Cmd+M` when not using voice

## Status Indicators

The right panel shows real-time status:
- **Audio**: Current audio state (Ready/Listening/Speaking)
- **LLM**: Connection status to Gemini
- **Status**: Current application state

## Debug View

Press **Cmd+T** to toggle between conversation view and debug view:
- **Conversation View** (default): Shows your chat with GLaDOS
- **Debug View**: Shows real-time event logging including:
  - State changes (INITIALIZING → IDLE → LISTENING, etc.)
  - Message events (user input, LLM responses)
  - Audio status changes (listening, processing, speaking)
  - LLM response chunks (streaming text)
  - Service initialization events
  - User actions (keyboard shortcuts, button clicks)

The debug view is useful for:
- Troubleshooting issues
- Understanding the event flow
- Monitoring LLM streaming responses
- Tracking audio pipeline state

## Tips

1. **Text is Default**: No need to enable anything - just start typing
2. **Quick Help**: Press `Cmd+H` anytime to see all key bindings
3. **Safe Interruption**: Use `Cmd+I` to stop long responses
4. **Voice Optional**: Voice features are completely optional
5. **Conversation Flow**: Both text and voice messages appear in the same conversation log
6. **Debug Mode**: Press `Cmd+T` to see what's happening under the hood

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
