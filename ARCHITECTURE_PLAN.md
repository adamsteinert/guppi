# GLaDOS 2.0 Architecture Plan

## Overview

This document outlines the architectural redesign of GLaDOS from the unstable original to a more robust and maintainable system. The new architecture addresses the core issues of thread safety, proper state management, clean separation of concerns, and reliable audio handling.

## Problems with Original Architecture

### Current Issues Identified
1. **Threading Problems**: Complex, hard-to-debug threading with queues and callbacks
2. **Tight Coupling**: Direct dependencies between UI, audio, and LLM components
3. **Inconsistent State**: No centralized state management leading to race conditions
4. **Poor Cancellation**: Audio playback and LLM requests don't cancel cleanly
5. **UI Instability**: TUI crashes and behaves unpredictably
6. **Error Handling**: Errors in one component can crash the entire application

### Original Architecture Issues
- `Glados` class in `engine.py` tries to do everything (God object antipattern)
- Threading logic mixed throughout codebase
- Audio callbacks directly modify UI state
- No clean shutdown mechanism
- Hard to test individual components

## New Architecture Design

### Core Principles
1. **Single Responsibility**: Each component has one clear purpose
2. **Loose Coupling**: Components communicate via event bus, not direct calls
3. **Centralized State**: All application state managed in one place
4. **Async/Await**: Modern async patterns instead of raw threading
5. **Graceful Degradation**: Components can fail without bringing down the system
6. **Dependency Injection**: Clear dependency management for testing

### Component Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   UI Layer      │    │  Core System     │    │  External APIs  │
│                 │    │                  │    │                 │
│ ┌─────────────┐ │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
│ │  Textual UI │ │    │ │  Event Bus   │ │    │ │ Ollama/LLM  │ │
│ │   (TUI)     │ │◄──►│ │              │ │◄──►│ │   APIs      │ │
│ └─────────────┘ │    │ └──────────────┘ │    │ └─────────────┘ │
│                 │    │ ┌──────────────┐ │    │ ┌─────────────┐ │
└─────────────────┘    │ │  State Mgr   │ │    │ │  Audio HW   │ │
                       │ │              │ │    │ │ (Microphone)│ │
┌─────────────────┐    │ └──────────────┘ │    │ └─────────────┘ │
│  Audio Layer    │    └──────────────────┘    └─────────────────┘
│                 │                          
│ ┌─────────────┐ │    ┌──────────────────┐  
│ │ Audio Mgr   │ │    │   LLM Layer      │  
│ │ (VAD/TTS)   │ │    │                  │  
│ └─────────────┘ │    │ ┌──────────────┐ │  
└─────────────────┘    │ │   LLM Mgr    │ │  
                       │ │ (Ollama/etc) │ │  
                       │ └──────────────┘ │  
                       └──────────────────┘  
```

### Key Components

#### 1. Event Bus (`core/event_bus.py`)
- **Purpose**: Decoupled communication between all components
- **Features**: 
  - Synchronous and asynchronous event publishing
  - Type-safe event definitions
  - Error isolation (one subscriber failure doesn't affect others)
- **Events**: State changes, audio events, LLM responses, UI interactions

#### 2. State Manager (`core/state_manager.py`) 
- **Purpose**: Centralized, thread-safe application state management
- **Features**:
  - Atomic state transitions with validation
  - State history tracking
  - Invalid transition prevention
- **States**: IDLE, LISTENING, PROCESSING_AUDIO, CALLING_LLM, GENERATING_TTS, PLAYING_AUDIO, ERROR, SHUTTING_DOWN

#### 3. Audio Manager (`audio/audio_manager.py`)
- **Purpose**: All audio operations (input/output, VAD, TTS)
- **Features**:
  - Cancellable audio playback with proper interruption
  - Clean microphone/speaker mute controls
  - Volume management
  - Robust error recovery
- **Improvements**: No more complex threading, clean cancellation

#### 4. LLM Manager (`llm/llm_manager.py`)
- **Purpose**: All LLM interactions with multiple provider support
- **Features**:
  - Streaming and non-streaming responses
  - Clean request cancellation
  - Conversation history management
  - Provider abstraction (Ollama, OpenAI, Gemini)
- **Improvements**: Proper async/await, cancellation support

#### 5. UI Application (`ui/app.py`)
- **Purpose**: Stable Textual-based user interface
- **Features**:
  - Event-driven updates (no polling or direct callbacks)
  - Proper conversation log management
  - Clear status indicators
  - Help system and key bindings
- **Improvements**: No more crashes, clean event handling

#### 6. Configuration Manager (`config/config_manager.py`)
- **Purpose**: Type-safe configuration management
- **Features**:
  - YAML-based configuration files
  - Default value handling
  - Runtime configuration updates
  - Validation and error handling

## Implementation Strategy

### Phase 1: Core Infrastructure ✅
- [x] Event bus with proper error handling
- [x] State manager with transition validation
- [x] Configuration management system
- [x] Basic project structure

### Phase 2: Stable UI Foundation ✅
- [x] Clean Textual UI without crashes
- [x] Event-driven conversation log
- [x] Status display system
- [x] Help and key binding system

### Phase 3: Audio System Rebuild
- [ ] Replace threading with async/await patterns
- [ ] Implement proper audio cancellation
- [ ] Add VAD integration with event bus
- [ ] Create TTS pipeline with interruption support
- [ ] Test audio state machine thoroughly

### Phase 4: LLM Integration
- [ ] Implement streaming LLM responses
- [ ] Add proper request cancellation
- [ ] Create provider abstraction layer
- [ ] Implement conversation memory management
- [ ] Add function calling support

### Phase 5: System Integration
- [ ] Wire all components together via event bus
- [ ] Implement complete audio → LLM → TTS → UI pipeline
- [ ] Add comprehensive error handling
- [ ] Create graceful shutdown sequence

### Phase 6: Advanced Features
- [ ] Wake word detection
- [ ] Voice interruption during playback
- [ ] Memory/context management
- [ ] Plugin system for extensions
- [ ] Performance monitoring

## Benefits of New Architecture

### Stability Improvements
- **Thread Safety**: Event bus eliminates race conditions
- **Clean Cancellation**: Async/await enables proper operation cancellation
- **Error Isolation**: Component failures don't cascade
- **Predictable State**: Centralized state prevents inconsistencies

### Maintainability Improvements  
- **Loose Coupling**: Components can be tested and modified independently
- **Clear Interfaces**: Event types define clear contracts
- **Single Responsibility**: Each class has one clear purpose
- **Dependency Injection**: Easy to mock and test

### User Experience Improvements
- **Reliable UI**: No more crashes or hangs
- **Responsive Controls**: Proper interruption and cancellation
- **Clear Status**: Users always know what the system is doing
- **Graceful Errors**: Better error messages and recovery

## Migration Strategy

The new architecture is being built in parallel (`src2/`) to avoid disrupting the existing system. Once stable, we can:

1. **Gradual Migration**: Move users to new system incrementally  
2. **Feature Parity**: Ensure all original features work in new system
3. **Performance Testing**: Validate latency and reliability improvements
4. **Backward Compatibility**: Support existing configuration files
5. **Documentation**: Update all documentation for new architecture

## Testing Strategy

### Component Testing
- Unit tests for each manager class
- Event bus testing with mock subscribers
- State manager transition validation
- Configuration loading and validation

### Integration Testing  
- Full pipeline testing (audio → LLM → TTS → UI)
- Error recovery scenarios
- Cancellation and interruption testing
- Performance and latency testing

### User Acceptance Testing
- UI responsiveness and stability
- Audio quality and timing
- Configuration management
- Error message clarity

## Future Enhancements

The new architecture enables several future improvements:

1. **Plugin System**: Event bus makes it easy to add new capabilities
2. **Remote Control**: HTTP API can publish events to control GLaDOS
3. **Multi-Modal**: Easy to add vision, tool calling, etc.
4. **Clustering**: Multiple GLaDOS instances could share state
5. **Metrics**: Built-in performance and usage monitoring
6. **A/B Testing**: Easy to swap components for experimentation

This architecture provides a solid foundation for both current stability needs and future feature development.