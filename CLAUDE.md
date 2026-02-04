# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GLaDOS is a voice assistant project that recreates the GLaDOS personality from Portal. It's a Python-based system using uv for dependency management that combines real-time audio processing, speech recognition, text-to-speech, and LLM integration for low-latency voice interactions.

**Two versions exist:**
- `src/` - Original GLaDOS implementation (stable, production)
- `src2/` - GLaDOS 2.0 refactor (in development, event-driven architecture)

## Development Commands

### GLaDOS 1.0 (Original - src/)
- `uv run glados download` - Download required AI model files (ASR, TTS, VAD models)
- `uv run glados start --config configs/dev_config.yaml` - Start GLaDOS with development config
- `uv run glados tui --config configs/dev_config.yaml` - Start with terminal UI
- `uv run glados say "text"` - Generate speech from text
- `uv run glados saytofile "text" --outfile temp.mp3` - Save speech to file

### GLaDOS 2.0 (Refactor - src2/)
- `cd src2 && uv run python -m glados2.main --config ../configs/glados2_gem_config.yaml` - Start GLaDOS 2.0
- `cd src2 && uv run pytest tests/ -v` - Run GLaDOS 2.0 tests

### Testing and Development
- `pytest` - Run all tests
- `ruff check` - Run linting
- `ruff format` - Format code
- `./go.sh` - Quick development start (uses dev_config.yaml)

### Configuration Files
- `configs/glados_config.yaml` - Main configuration (v1)
- `configs/dev_config.yaml` - Development configuration (v1)
- `configs/glados2_gem_config.yaml` - GLaDOS 2.0 with Gemini
- `configs/assistant_config.yaml` - Alternative personality configuration

## Architecture Overview

### GLaDOS 1.0 (src/)

**Engine (`src/glados/engine.py`)**
- Main `Glados` class orchestrates all components
- Manages audio processing pipeline: VAD → ASR → LLM → TTS
- Handles real-time audio streams with circular buffering
- Implements wake word detection using Levenshtein distance
- Maintains conversation context and memory

**Audio Processing Pipeline**
- Voice Activity Detection (VAD) using Silero VAD model
- Automatic Speech Recognition (ASR) using Nemo Parakeet model
- Text-to-Speech (TTS) supporting GLaDOS and Kokoro voices
- Real-time audio streaming with sounddevice

**LLM Integration (`src/glados/Extensions/llm/`)**
- `OllamaToolManager` - Integration with Ollama for local LLMs
- `GeminiAgent` - Google Gemini API integration
- `MCPClient` - Model Context Protocol client for tool calling
- `LanguageAgent` - Abstract base for LLM providers

**Extensions System**
- Commands (`src/glados/Extensions/Commands/`) - Extensible command processing
- Tools (`src/glados/Extensions/Tools/`) - Function calling capabilities
- Memory (`src/glados/Extensions/vectors/`) - Vector-based memory storage

### GLaDOS 2.0 (src2/) - Event-Driven Architecture

GLaDOS 2.0 is a complete refactor using a decoupled, event-driven architecture with pub-sub messaging.

**Core Components (`src2/glados2/core/`)**
- `EventBus` - Central pub-sub message bus for decoupled communication
- `StateManager` - Thread-safe application state machine with valid transitions
- Event types: STATE_CHANGED, MESSAGE_RECEIVED, AUDIO_STATUS_CHANGED, AUDIO_DEVICE_CHANGED, LLM_RESPONSE_*, TTS_*, etc.

**Audio System (`src2/glados2/audio/`)**
- `AudioManager` - Manages audio capture, playback, and device monitoring
- `VADProcessor` - Silero VAD v5 with stateful LSTM inference
- `ASRProcessor` - Nemo Parakeet with mel spectrogram features and CTC decoding
- `TTSProcessor` - GLaDOS and Kokoro voice synthesis
- `AudioDeviceMonitor` - Detects system audio device changes (AirPods, speakers, etc.)

**LLM System (`src2/glados2/llm/`)**
- `LLMManager` - Handles LLM interactions with streaming support
- Supports Ollama, OpenAI, and Gemini providers
- Mock responses for testing without API

**UI (`src2/glados2/ui/`)**
- `GladosUI` - Textual-based terminal UI
- Real-time status display, conversation history, waveform visualization

**Key Architecture Patterns:**
- All components communicate via EventBus (no direct coupling)
- Async task scheduling from sync callbacks via `_schedule_async_task()`
- State machine prevents invalid transitions
- Audio pipeline: Capture → VAD → ASR → (publish MESSAGE_RECEIVED) → LLM → TTS → Playback

### Key Constants and Configuration
- Sample rate: 16kHz for audio processing
- VAD threshold: 0.8 for voice detection
- Buffer size: 800ms before VAD detection
- Pause limit: 1000ms before processing audio
- Wake word similarity threshold: 3 (Levenshtein distance)

### Model Files
All models stored in `models/` directory:
- ASR: `nemo-parakeet_tdt_ctc_110m.onnx` + `nemo-parakeet_tdt_ctc_110m_tokens.txt`
- VAD: `silero_vad_v5.onnx`
- TTS: `glados.onnx`, `kokoro-v1.0.fp16.onnx`
- Phonemizer: `phomenizer_en.onnx`

## Development Notes

### Dependencies
- Uses `uv` for fast Python package management
- ONNX Runtime for model inference (supports CUDA, CPU variants)
- sounddevice for real-time audio I/O
- Supports both local (Ollama) and cloud (OpenAI/Gemini) LLMs

### Code Style
- Configured with ruff for linting and formatting
- Line length: 120 characters  
- Python 3.12+ required
- Type hints enforced with mypy in strict mode

### Testing
- Tests located in `tests/` directory (v1) and `src2/tests/` (v2)
- Test modules: commands, engine, spoken text converter, tool calls, memory
- GLaDOS 2.0 tests: audio listening, TTS processor, VAD, ASR
- Use pytest framework with pytest-asyncio for async tests

## GLaDOS 2.0 Development Notes

### Important Patterns

**Async Task Scheduling from Sync Callbacks**
EventBus callbacks are synchronous, but often need to trigger async operations. Use `_schedule_async_task()`:
```python
def _schedule_async_task(self, coro) -> None:
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(coro)
    except RuntimeError:
        # No running loop - run in new thread
        thread = threading.Thread(target=lambda: asyncio.run(coro), daemon=True)
        thread.start()
```

**State Transitions**
The StateManager enforces valid transitions. Key flow:
- IDLE → LISTENING → PROCESSING_AUDIO → CALLING_LLM → GENERATING_TTS → PLAYING_AUDIO → IDLE
- Don't set state in multiple places for the same transition (causes race conditions)

**Audio Device Detection**
PortAudio caches device list. To detect changes (e.g., AirPods → speakers):
```python
sd._terminate()
sd._initialize()
# Now query_devices() returns fresh data
```

**ASR Model Requirements**
Nemo Parakeet requires two inputs:
- `audio_signal`: mel spectrogram (batch, 80, time_frames) - use MelSpectrogramCalculator
- `length`: time frame count as int64 array

**VAD Model Requirements**
Silero VAD v5 is stateful and requires:
- `input`: audio chunk (batch, 512) for 16kHz
- `state`: LSTM state (2, 1, 128) - must persist between calls
- `sr`: sample rate as int64

### Recent Fixes (Session History)

1. **Audio device change detection** - Added AudioDeviceMonitor with PortAudio cache refresh
2. **Async task scheduling** - Added `_schedule_async_task()` to AudioManager and LLMManager
3. **ASR "required inputs 'length'" error** - Fixed by computing mel spectrogram and providing both inputs
4. **LLM not receiving transcriptions** - Fixed race condition where AudioManager set CALLING_LLM state before LLMManager could transition