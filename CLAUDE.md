# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

GLaDOS is a voice assistant project that recreates the GLaDOS personality from Portal. It's a Python-based system using uv for dependency management that combines real-time audio processing, speech recognition, text-to-speech, and LLM integration for low-latency voice interactions.

## Development Commands

### Core Commands
- `uv run glados download` - Download required AI model files (ASR, TTS, VAD models)
- `uv run glados start --config configs/dev_config.yaml` - Start GLaDOS with development config
- `uv run glados tui --config configs/dev_config.yaml` - Start with terminal UI
- `uv run glados say "text"` - Generate speech from text
- `uv run glados saytofile "text" --outfile temp.mp3` - Save speech to file

### Testing and Development
- `pytest` - Run all tests
- `ruff check` - Run linting
- `ruff format` - Format code
- `./go.sh` - Quick development start (uses dev_config.yaml)

### Configuration Files
- `configs/glados_config.yaml` - Main configuration
- `configs/dev_config.yaml` - Development configuration  
- `configs/assistant_config.yaml` - Alternative personality configuration

## Architecture Overview

### Core Components

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

### Key Constants and Configuration
- Sample rate: 16kHz for audio processing
- VAD threshold: 0.8 for voice detection
- Buffer size: 800ms before VAD detection
- Pause limit: 1000ms before processing audio
- Wake word similarity threshold: 3 (Levenshtein distance)

### Model Files
All models stored in `models/` directory:
- ASR: `nemo-parakeet_tdt_ctc_110m.onnx`
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
- Tests located in `tests/` directory
- Test modules: commands, engine, spoken text converter, tool calls, memory
- Use pytest framework