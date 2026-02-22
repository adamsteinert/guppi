# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GLaDOS is a voice assistant project built in Python using uv for dependency management. It combines real-time audio processing, speech recognition, text-to-speech, and LLM integration for low-latency voice interactions.

**Active codebase:** `src2/` - Event-driven architecture with pub-sub messaging, tool/MCP integration, and Textual TUI.

> `src/` is **deprecated** and should not be used for new development.

## Development Commands

### Running GLaDOS
- `./run_glados2.sh` - Quick start with default Gemini config
- `cd src2 && uv run python -m glados2.main --config ../configs/glados2_gem_config.yaml` - Start with specific config
- `cd src2 && uv run python -m glados2.main --headless --config ../configs/glados2_gem_config.yaml` - Headless mode (no TUI)
- `cd src2 && uv run python -m glados2.main --debug --config ../configs/glados2_gem_config.yaml` - Debug mode (terminal input, no voice)
- `cd src2 && uv run python -m glados2.main say "text"` - Generate and play speech
- `cd src2 && uv run python -m glados2.main saytofile "text" --outfile audio.wav` - Save speech to file

### Model Download (uses v1 CLI)
- `uv run glados download` - Download required AI model files (ASR, TTS, VAD models)

### Testing and Development
- `cd src2 && uv run pytest tests/ -v` - Run tests
- `ruff check` - Run linting
- `ruff format` - Format code

### Configuration Files
- `configs/glados2_gem_config.yaml` - Gemini with GLaDOS voice
- `configs/glados2_bm_george_config.yaml` - Gemini with bm_george voice + MCP tools
- `configs/glados2_mcp_config.yaml` - Gemini with MCP tools + salutation/valediction
- `configs/glados2_config.yaml` - Ollama (local) with GLaDOS voice

## Architecture Overview

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed diagram.

### Directory Structure
```
src2/glados2/
├── main.py                 # GladosApp orchestrator, CLI entry point
├── core/
│   ├── event_bus.py        # Central pub-sub message bus
│   ├── state_manager.py    # Thread-safe state machine
│   └── worker_loop.py      # Persistent async event loop for MCP/LLM
├── audio/
│   ├── audio_manager.py    # Audio pipeline orchestrator + device monitoring
│   ├── vad_processor.py    # Voice Activity Detection (Silero VAD v5)
│   ├── asr_processor.py    # Speech Recognition (Nemo Parakeet CTC)
│   ├── tts_processor.py    # TTS factory (routes to GLaDOS or Kokoro)
│   ├── tts_glados.py       # GLaDOS voice (VITS/Piper, 22050 Hz)
│   └── tts_kokoro.py       # Kokoro voices (26 voices, 24000 Hz)
├── llm/
│   └── llm_manager.py      # LLM provider management, streaming, tool calling
├── tools/
│   ├── tool_types.py       # ToolSpec, ToolCall, ToolResult types
│   ├── tool_manager.py     # Tool registry, execution, format conversion
│   └── mcp_client.py       # MCP server stdio client
├── config/
│   └── config_manager.py   # YAML config with env var substitution
└── ui/
    └── app.py              # Textual TUI (conversation log, status, keybindings)
```

### Core Architecture

All components communicate exclusively through the **EventBus** pub-sub system. No direct coupling between components.

**GladosApp** (`main.py`) is the top-level orchestrator that creates and wires all components via dependency injection.

**EventBus** (`core/event_bus.py`) - Central message bus supporting sync and async subscribers.

**StateManager** (`core/state_manager.py`) - Thread-safe state machine with enforced transitions:
```
INITIALIZING → IDLE → LISTENING → PROCESSING_AUDIO → CALLING_LLM → GENERATING_TTS → PLAYING_AUDIO → IDLE
```
- Any state can transition to SHUTTING_DOWN or ERROR
- PLAYING_AUDIO → LISTENING (for continuous listening)

**WorkerLoop** (`core/worker_loop.py`) - Persistent background event loop in a dedicated thread. Ensures MCP stdio sessions and LLM requests share the same loop.

### Event Types
```
STATE_CHANGED, MESSAGE_RECEIVED, AUDIO_STATUS_CHANGED, AUDIO_DEVICE_CHANGED,
LLM_RESPONSE_STARTED, LLM_RESPONSE_CHUNK, LLM_RESPONSE_COMPLETED,
TTS_STARTED, TTS_COMPLETED, AUDIO_PLAYBACK_STARTED, AUDIO_PLAYBACK_COMPLETED,
INTERRUPT_REQUESTED, LISTENING_STOPPED, ERROR_OCCURRED, SHUTDOWN_REQUESTED,
TOOL_EXECUTION_STARTED, TOOL_EXECUTION_COMPLETED, TOOL_EXECUTION_ERROR,
MCP_SERVER_CONNECTED, MCP_SERVER_DISCONNECTED, MCP_TOOLS_DISCOVERED
```

### Audio Pipeline

```
Mic (16kHz mono) → VAD (Silero v5, 512-sample windows) → ASR (Nemo Parakeet, mel spectrogram + CTC)
    → MESSAGE_RECEIVED event → LLM → TTS (GLaDOS/Kokoro) → Speaker
```

**VAD:** Stateful LSTM inference, 0.8 threshold, 250ms min speech, 1000ms silence gap, 5s auto-stop.

**ASR:** Requires `audio_signal` (mel spectrogram: batch, 80, time_frames) and `length` (frame count as int64).

**TTS:** Factory pattern routes to GladosSynthesizer (22050 Hz) or KokoroSynthesizer (24000 Hz, 26 voices, IPA phonemes).

### LLM Integration

**Providers:** Ollama (local), OpenAI, Gemini (native google-genai SDK with thinking mode support).

**Tool Calling:** LLM returns tool_calls → ToolManager executes → results fed back to LLM (max 5 iterations).

**Response Summarization:** Long responses are summarized to ~150 words before TTS synthesis.

### MCP/Tool System

**ToolManager** - Registry for tools from MCP servers and built-in sources. Converts between OpenAI and Gemini tool formats. Executes with configurable timeouts (default 30s).

**MCPClient** - Manages stdio-based MCP server connections. Auto-discovers tools on connect.

### UI (Textual TUI)

Key bindings: `q` quit, `l` toggle listening, `i` interrupt, `m` mute mic, `s` mute speaker, `t` debug, `p` text input, `c` compact mode.

All UI callbacks are wrapped to execute on Textual's main thread.

## Development Notes

### Dependencies
- `uv` for package management, Python 3.12+
- ONNX Runtime for model inference (CUDA and CPU variants)
- sounddevice for real-time audio I/O
- google-genai for Gemini, openai for OpenAI/Ollama, mcp for tool protocol
- textual for terminal UI

### Code Style
- Ruff for linting and formatting, line length 120
- mypy in strict mode
- Python 3.12+

### Testing
- Tests in `src2/tests/` using pytest + pytest-asyncio
- Test modules: audio listening, TTS processor, LLM tool awareness, MCP integration, tool manager, worker loop

### Key Constants
- Sample rate: 16kHz
- VAD threshold: 0.8
- VAD window: 512 samples (32ms)
- Buffer size: 800ms
- Pause limit: 1000ms
- Min speech duration: 250ms
- Silence timeout: 5s

### Model Files (`models/`)
- ASR: `nemo-parakeet_tdt_ctc_110m.onnx` + `nemo-parakeet_tdt_ctc_110m_tokens.txt`
- VAD: `silero_vad_v5.onnx`
- TTS: `glados.onnx`, `kokoro-v1.0.fp16.onnx`
- Phonemizer: `phomenizer_en.onnx`

### Important Patterns

**Async Task Scheduling from Sync Callbacks**
EventBus callbacks are synchronous but often need to trigger async operations. Use WorkerLoop for MCP/LLM work, or `_schedule_async_task()` as fallback:
```python
def _schedule_async_task(self, coro) -> None:
    if self._worker_loop and self._worker_loop.is_running:
        self._worker_loop.run(coro)
    else:
        thread = threading.Thread(target=lambda: asyncio.run(coro), daemon=True)
        thread.start()
```

**Audio Device Detection**
PortAudio caches device list. AudioDeviceMonitor refreshes the cache to detect changes:
```python
sd._terminate()
sd._initialize()
```

**Textual App Reserved Attributes**
- `App._main_thread` is a reserved bool in Textual - never use this name for custom methods.
- `App._ready` is a reserved async method - use `_services_ready` instead.

### Configuration System
YAML-based with dataclass config objects (AudioConfig, LLMConfig, TTSConfig, UIConfig, MCPServerConfig, ToolsConfig, GladosConfig). Supports `${VAR_NAME}` environment variable substitution and `.env` files.
