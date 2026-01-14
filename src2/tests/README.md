# GLaDOS 2.0 Tests

Comprehensive test suite for the GLaDOS 2.0 voice assistant system.

## Test Organization

Tests are organized by component and type:

- **Unit Tests**: Fast tests with no external dependencies
- **Integration Tests**: Tests requiring models, hardware, or external services
- **Audio Tests**: Tests that produce or play audio
- **Model Tests**: Tests requiring ONNX models

## Running Tests

### Quick Start

```bash
# Run all unit tests (fast)
cd src2
pytest tests/

# Or use the test runner script
./run_tests.sh
```

### Test Markers

Tests are marked for selective execution:

```bash
# Run only unit tests (no integration)
pytest tests/ -m "not integration"

# Run only integration tests
pytest tests/ -m "integration"

# Run only TTS tests
pytest tests/test_tts_processor.py

# Run tests with verbose output
pytest tests/ -v

# Run specific test
pytest tests/test_tts_processor.py::TestTTSProcessor::test_initialization
```

### Available Markers

- `unit` - Fast unit tests (default)
- `integration` - Tests requiring models/hardware
- `slow` - Long-running tests
- `audio` - Tests that play or record audio
- `model` - Tests requiring ONNX models

## Test Files

### `test_tts_processor.py`

Comprehensive TTS processor tests including:

- **TestTTSProcessor**: Core TTS functionality
  - Initialization and configuration
  - Voice selection and listing
  - Synthesis cancellation
  - Audio post-processing (DC removal, normalization, fade)

- **TestTTSGLaDOSSpecific**: GLaDOS voice tests
  - Phoneme to ID conversion
  - GLaDOS model synthesis

- **TestTTSKokoroSpecific**: Kokoro voice tests
  - Voice selection (bm_george, bf_emma, etc.)
  - Voice embedding generation
  - Multi-voice support

- **TestTTSIntegration**: Full pipeline tests
  - Real model synthesis
  - Audio quality validation
  - Multiple voice switching

## Validation Scripts

### `validate_tts_bm_george.py`

Standalone validation script for the bm_george Kokoro voice:

```bash
# Run TTS validation and hear the results
python src2/validate_tts_bm_george.py
```

This script:
- Tests TTS initialization
- Lists available voices
- Synthesizes test phrases
- Plays audio through speakers
- Saves audio files to `output/` directory
- Validates audio quality (DC offset, RMS energy, clipping)

**Output Files:**
- `output/bm_george_fallback.wav` - Fallback synthesis (when model not loaded)
- `output/bm_george_model.wav` - Real model synthesis (when model available)
- `output/bm_george_multiple.wav` - Multiple sentences combined

## Requirements

Test dependencies:
```bash
pytest>=7.0.0
pytest-asyncio>=0.21.0
pytest-cov>=4.0.0  # Optional, for coverage reports
```

All test dependencies are included in the main `pyproject.toml`.

## Writing Tests

### Test Structure

```python
import pytest
from glados2.audio.tts_processor import TTSProcessor

class TestMyComponent:
    @pytest.fixture
    def component(self):
        """Create component for testing."""
        return TTSProcessor(voice="bm_george")

    def test_something(self, component):
        """Test description."""
        result = component.do_something()
        assert result is not None

    @pytest.mark.asyncio
    async def test_async_something(self, component):
        """Test async functionality."""
        result = await component.async_do_something()
        assert result is not None

    @pytest.mark.integration
    def test_with_model(self, component):
        """Test requiring real model."""
        if not component.model_loaded():
            pytest.skip("Model not available")
        # Test code here
```

### Best Practices

1. **Use fixtures** for setup/teardown
2. **Mark tests appropriately** (unit, integration, slow)
3. **Skip tests** when dependencies unavailable
4. **Test edge cases** (empty input, cancellation, errors)
5. **Validate outputs** (shape, dtype, range for audio)
6. **Keep unit tests fast** (< 1 second each)

## Continuous Integration

When setting up CI/CD:

```bash
# Install dependencies
uv sync

# Run unit tests only (fast)
pytest tests/ -m "not integration" --tb=short

# Run with coverage
pytest tests/ --cov=glados2 --cov-report=xml
```

## Troubleshooting

### Tests fail with "Model not found"

Integration tests require ONNX models in `models/TTS/`:
- `glados.onnx`
- `kokoro-v1.0.fp16.onnx`
- `phoneme_to_id.pkl`

These tests will skip automatically if models are missing.

### Audio tests fail

Audio tests require:
- `sounddevice` package installed
- Audio output device available
- May need system permissions for audio access

### Async tests fail

Ensure `pytest-asyncio` is installed:
```bash
uv add --dev pytest-asyncio
```

## Coverage

Generate coverage report:

```bash
# HTML report (opens in browser)
pytest tests/ --cov=glados2 --cov-report=html
open htmlcov/index.html

# Terminal report
pytest tests/ --cov=glados2 --cov-report=term

# XML report (for CI)
pytest tests/ --cov=glados2 --cov-report=xml
```

## Performance Testing

For performance benchmarks:

```bash
# Install pytest-benchmark
uv add --dev pytest-benchmark

# Run with timing
pytest tests/ --durations=10
```
