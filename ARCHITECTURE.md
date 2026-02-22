# GLaDOS 2.0 Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           GladosApp (main.py)                           │
│                    Top-level orchestrator & lifecycle                    │
│              Creates, wires, and manages all components                  │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │ dependency injection
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
   ┌─────────────┐    ┌──────────────┐    ┌──────────────┐
   │  EventBus   │    │ StateManager │    │  WorkerLoop  │
   │  (pub-sub)  │    │   (states)   │    │ (async loop) │
   └──────┬──────┘    └──────┬───────┘    └──────┬───────┘
          │                  │                    │
          │    All components communicate         │
          │    through EventBus only              │
          │                                       │
  ┌───────┴───────────────────────────────────────┴──────────┐
  │                                                           │
  ▼                    ▼                    ▼                  ▼
┌──────────────┐ ┌───────────────┐ ┌──────────────┐ ┌──────────────┐
│ AudioManager │ │  LLMManager   │ │ ToolManager  │ │   GladosUI   │
│   (audio/)   │ │   (llm/)      │ │  (tools/)    │ │    (ui/)     │
└──────────────┘ └───────────────┘ └──────────────┘ └──────────────┘
```

## Voice Interaction Pipeline

```
                         ┌──────────────────┐
                         │    Microphone     │
                         │  16kHz mono f32   │
                         └────────┬─────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   PortAudio Stream Callback  │
                    │     → thread-safe queue      │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │       VAD Processor          │
                    │    Silero VAD v5 (ONNX)      │
                    │                              │
                    │  ● 512-sample windows (32ms) │
                    │  ● Stateful LSTM inference   │
                    │  ● Threshold: 0.8            │
                    │  ● Min speech: 250ms         │
                    │  ● Silence gap: 1000ms       │
                    └─────────────┬───────────────┘
                                  │ speech segment
                                  ▼
                    ┌─────────────────────────────┐
                    │       ASR Processor          │
                    │  Nemo Parakeet CTC (ONNX)    │
                    │                              │
                    │  ● Mel spectrogram (80 bins) │
                    │  ● CTC greedy decoding       │
                    │  ● Token filtering           │
                    └─────────────┬───────────────┘
                                  │ transcribed text
                                  ▼
                    ┌─────────────────────────────┐
                    │   EventBus: MESSAGE_RECEIVED │
                    │     (role: "user")           │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
              ┌───────────────────────────────────────────┐
              │              LLM Manager                   │
              │                                            │
              │  Providers:                                │
              │  ┌──────────┐ ┌────────┐ ┌──────────┐    │
              │  │  Ollama   │ │ OpenAI │ │  Gemini  │    │
              │  │ (local)   │ │ (API)  │ │  (API)   │    │
              │  └──────────┘ └────────┘ └──────────┘    │
              │                                            │
              │  ● Streaming responses                     │
              │  ● Tool calling loop (max 5 iterations)    │
              │  ● Conversation history management         │
              │  ● Long response summarization (~150 wds)  │
              └──────────────────┬────────────────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
    ┌───────────────────────┐   ┌─────────────────────────┐
    │    Tool Execution     │   │  EventBus: MESSAGE_RECV │
    │                       │   │   (role: "assistant")    │
    │  ToolManager          │   └────────────┬────────────┘
    │   ├── MCP Client(s)   │                │
    │   │   (stdio servers) │                ▼
    │   └── Built-in tools  │   ┌─────────────────────────┐
    │                       │   │      TTS Processor       │
    │  Results fed back     │   │                          │
    │  to LLM for next      │   │  ┌─────────┐ ┌───────┐ │
    │  iteration            │   │  │ GLaDOS   │ │Kokoro │ │
    └───────────────────────┘   │  │ 22050 Hz │ │24kHz  │ │
                                │  │ VITS     │ │26 vox │ │
                                │  └─────────┘ └───────┘ │
                                └────────────┬────────────┘
                                             │ audio data
                                             ▼
                                ┌─────────────────────────┐
                                │    Audio Playback        │
                                │  ● Chunk-based output    │
                                │  ● Volume scaling        │
                                │  ● Cancellation support  │
                                └────────────┬────────────┘
                                             │
                                             ▼
                                    ┌────────────────┐
                                    │    Speaker     │
                                    └────────────────┘
```

## State Machine

```
                    ┌────────────────┐
                    │ INITIALIZING   │
                    └───────┬────────┘
                            │
                            ▼
            ┌──────► IDLE ◄──────────────────────────┐
            │        │  ▲                             │
            │        │  │                             │
            │        ▼  │                             │
            │    LISTENING                            │
            │        │  ▲                             │
            │        │  └── (PLAYING_AUDIO can        │
            │        ▼       loop back here)          │
            │  PROCESSING_AUDIO                       │
            │        │                                │
            │        ▼                                │
            │   CALLING_LLM                           │
            │        │                                │
            │        ▼                                │
            │  GENERATING_TTS                         │
            │        │                                │
            │        ▼                                │
            │  PLAYING_AUDIO ─────────────────────────┘
            │
            │  (any state)
            │     │    │
            │     ▼    ▼
            │  ERROR  SHUTTING_DOWN
            │     │
            └─────┘
```

## EventBus Communication

```
┌──────────────┐   STATE_CHANGED          ┌──────────────┐
│ StateManager │ ─────────────────────►   │   All Subs   │
└──────────────┘                          └──────────────┘

┌──────────────┐   MESSAGE_RECEIVED       ┌──────────────┐
│ AudioManager │ ─── (role: user) ────►   │  LLMManager  │
└──────────────┘                          └──────┬───────┘
       ▲                                         │
       │              MESSAGE_RECEIVED           │
       └──────── (role: assistant) ──────────────┘

┌──────────────┐   LLM_RESPONSE_CHUNK     ┌──────────────┐
│  LLMManager  │ ────────────────────►    │   GladosUI   │
└──────────────┘                          └──────────────┘

┌──────────────┐   TOOL_EXECUTION_*       ┌──────────────┐
│ ToolManager  │ ────────────────────►    │   GladosUI   │
└──────────────┘                          └──────────────┘

┌──────────────┐   TTS_*, PLAYBACK_*      ┌──────────────┐
│ AudioManager │ ────────────────────►    │   GladosUI   │
└──────────────┘                          └──────────────┘
```

## Tool Calling Flow

```
LLMManager receives LLM response with tool_calls
        │
        ▼
┌─────────────────────────────────────────┐
│         For each tool_call:             │
│                                         │
│  1. Publish TOOL_EXECUTION_STARTED      │
│  2. ToolManager.execute_tool()          │
│     ├── Lookup tool in registry         │
│     ├── Route to MCP client or built-in │
│     └── asyncio.wait_for(timeout=30s)   │
│  3. Publish TOOL_EXECUTION_COMPLETED    │
│  4. Add ToolResult to conversation      │
│                                         │
│  Loop back to LLM with results          │
│  (max 5 iterations)                     │
└─────────────────────────────────────────┘
```

## Component Details

| Component | File | Responsibility |
|-----------|------|----------------|
| **GladosApp** | `main.py` | Orchestrator, lifecycle, dependency injection |
| **EventBus** | `core/event_bus.py` | Pub-sub messaging (sync + async subscribers) |
| **StateManager** | `core/state_manager.py` | Thread-safe state machine, transition validation |
| **WorkerLoop** | `core/worker_loop.py` | Persistent async loop for MCP/LLM operations |
| **AudioManager** | `audio/audio_manager.py` | Audio I/O, pipeline orchestration, device monitoring |
| **VADProcessor** | `audio/vad_processor.py` | Silero VAD v5, speech segment detection |
| **ASRProcessor** | `audio/asr_processor.py` | Nemo Parakeet, mel spectrogram, CTC decoding |
| **TTSProcessor** | `audio/tts_processor.py` | Factory routing to GLaDOS or Kokoro synthesizer |
| **GladosSynthesizer** | `audio/tts_glados.py` | GLaDOS voice (VITS/Piper, 22050 Hz) |
| **KokoroSynthesizer** | `audio/tts_kokoro.py` | Kokoro voices (26 voices, 24000 Hz, IPA) |
| **LLMManager** | `llm/llm_manager.py` | LLM providers, streaming, tool loop, summarization |
| **ToolManager** | `tools/tool_manager.py` | Tool registry, execution, format conversion |
| **MCPClient** | `tools/mcp_client.py` | MCP stdio server connection, tool discovery |
| **ConfigManager** | `config/config_manager.py` | YAML config loading, env var substitution |
| **GladosUI** | `ui/app.py` | Textual TUI, conversation display, keybindings |
