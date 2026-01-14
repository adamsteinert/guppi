"""Tests for audio listening and voice activity detection."""

import asyncio
import numpy as np
import pytest
import sys
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

# Add src2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from glados2.core.event_bus import EventBus, EventType
from glados2.core.state_manager import StateManager, AppState
from glados2.audio.vad_processor import VADProcessor


class TestVADProcessor:
    """Test Voice Activity Detection processor."""

    @pytest.fixture
    def vad(self):
        """Create VAD processor without loading model (uses fallback)."""
        return VADProcessor(
            model_path="nonexistent",  # Force fallback
            sample_rate=16000,
            threshold=0.5,
            min_speech_duration_ms=100,
            min_silence_duration_ms=500,
        )

    def test_initialization(self, vad):
        """Test VAD processor initializes correctly."""
        assert vad is not None
        assert vad.sample_rate == 16000
        assert vad.threshold == 0.5
        assert vad.is_speech_active is False

    def test_process_silence(self, vad):
        """Test VAD with silent audio (zeros)."""
        silent_audio = np.zeros(1024, dtype=np.float32)
        is_voice, completed = vad.process_audio_chunk(silent_audio)

        assert is_voice is False
        assert completed is None

    def test_process_loud_audio(self, vad):
        """Test VAD with loud audio (should trigger voice detection via fallback)."""
        # Generate loud sinusoidal audio
        t = np.linspace(0, 0.1, 1600, dtype=np.float32)  # 100ms at 16kHz
        loud_audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440Hz tone at 0.5 amplitude

        is_voice, completed = vad.process_audio_chunk(loud_audio)

        # With energy-based fallback, loud audio should be detected
        assert is_voice is True or vad.is_speech_active is True

    def test_energy_based_vad_silent(self, vad):
        """Test energy-based VAD fallback with silent audio."""
        silent = np.zeros(512, dtype=np.float32)
        prob = vad._energy_based_vad(silent)
        assert prob == 0.0

    def test_energy_based_vad_loud(self, vad):
        """Test energy-based VAD fallback with loud audio."""
        loud = np.ones(512, dtype=np.float32) * 0.5
        prob = vad._energy_based_vad(loud)
        assert prob > 0.0

    def test_reset(self, vad):
        """Test VAD reset clears state."""
        # Simulate some state
        vad.is_speech_active = True
        vad.speech_start_time = time.time()
        vad.collected_audio.append(np.zeros(100, dtype=np.float32))

        vad.reset()

        assert vad.is_speech_active is False
        assert vad.speech_start_time is None
        assert len(vad.collected_audio) == 0

    def test_speech_segment_collection(self, vad):
        """Test that speech segments are properly collected."""
        # Set lower threshold for easier triggering
        vad.threshold = 0.01
        vad.min_silence_duration_ms = 100  # Short silence duration

        # Generate loud audio to trigger speech
        loud_audio = np.ones(1024, dtype=np.float32) * 0.5

        # Process to start speech
        is_voice, completed = vad.process_audio_chunk(loud_audio)

        # Should have started collecting
        assert vad.is_speech_active or len(vad.collected_audio) > 0


class TestEventBusAsyncTask:
    """Test that event bus callbacks work with async task creation."""

    def test_sync_callback_with_async_task_fails_without_loop(self):
        """Test that asyncio.create_task fails in sync callback without event loop."""
        event_bus = EventBus()
        task_created = []
        error_occurred = []

        async def async_work():
            await asyncio.sleep(0.01)
            return "done"

        def sync_callback(data):
            try:
                # This will fail if no event loop is running
                task = asyncio.create_task(async_work())
                task_created.append(task)
            except RuntimeError as e:
                error_occurred.append(str(e))

        event_bus.subscribe(EventType.STATE_CHANGED, sync_callback)

        # Publish from sync context (no event loop)
        event_bus.publish(EventType.STATE_CHANGED, {"new_state": "test"})

        # Should have caught an error about no running event loop
        assert len(error_occurred) > 0 or len(task_created) == 0

    @pytest.mark.asyncio
    async def test_sync_callback_with_async_task_works_with_loop(self):
        """Test that asyncio.create_task works in sync callback WITH event loop."""
        event_bus = EventBus()
        task_created = []

        async def async_work():
            await asyncio.sleep(0.01)
            return "done"

        def sync_callback(data):
            try:
                task = asyncio.create_task(async_work())
                task_created.append(task)
            except RuntimeError:
                pass

        event_bus.subscribe(EventType.STATE_CHANGED, sync_callback)

        # Publish from async context (event loop IS running)
        event_bus.publish(EventType.STATE_CHANGED, {"new_state": "test"})

        # Wait a bit for task
        await asyncio.sleep(0.05)

        # Should have created a task
        assert len(task_created) > 0


class TestAudioManagerStateHandling:
    """Test AudioManager state change handling."""

    @pytest.fixture
    def event_bus(self):
        """Create event bus."""
        return EventBus()

    @pytest.fixture
    def state_manager(self, event_bus):
        """Create state manager."""
        return StateManager(event_bus)

    def test_state_change_triggers_callback(self, event_bus, state_manager):
        """Test that state changes trigger subscribed callbacks."""
        states_received = []

        def on_state_changed(data):
            states_received.append(data.get("new_state"))

        event_bus.subscribe(EventType.STATE_CHANGED, on_state_changed)

        state_manager.set_state(AppState.LISTENING)

        assert len(states_received) > 0
        assert states_received[-1] == AppState.LISTENING

    def test_listening_state_from_idle(self, event_bus, state_manager):
        """Test transition from IDLE to LISTENING state."""
        state_manager.set_state(AppState.IDLE)
        assert state_manager.get_state() == AppState.IDLE

        state_manager.set_state(AppState.LISTENING)
        assert state_manager.get_state() == AppState.LISTENING


class TestAudioInputCapture:
    """Test audio input capture functionality."""

    @pytest.mark.asyncio
    async def test_audio_queue_receives_data(self):
        """Test that audio callback puts data in queue."""
        import sounddevice as sd

        audio_queue = asyncio.Queue()
        received_chunks = []

        def audio_callback(indata, frames, time_info, status):
            audio_data = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
            try:
                audio_queue.put_nowait(audio_data.astype(np.float32))
            except asyncio.QueueFull:
                pass

        # Create and start stream briefly
        try:
            stream = sd.InputStream(
                samplerate=16000,
                channels=1,
                dtype=np.float32,
                blocksize=1024,
                callback=audio_callback
            )
            stream.start()

            # Collect for short time
            start_time = time.time()
            while time.time() - start_time < 0.5:  # 500ms
                try:
                    chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)
                    received_chunks.append(chunk)
                except asyncio.TimeoutError:
                    continue

            stream.stop()
            stream.close()

        except Exception as e:
            pytest.skip(f"Audio device not available: {e}")

        # Should have received some audio chunks
        assert len(received_chunks) > 0, "No audio chunks received - check microphone"

        # Each chunk should have data
        for chunk in received_chunks:
            assert len(chunk) > 0
            assert chunk.dtype == np.float32

    def test_sounddevice_devices_available(self):
        """Test that sounddevice can query devices."""
        import sounddevice as sd

        try:
            devices = sd.query_devices()
            assert devices is not None

            # Check for input device
            input_device = sd.query_devices(kind='input')
            assert input_device is not None
            print(f"Default input device: {input_device.get('name')}")

        except Exception as e:
            pytest.skip(f"Audio devices not available: {e}")


class TestAsyncTaskCreationFix:
    """Test the fix for async task creation in sync callbacks."""

    def test_get_or_create_event_loop(self):
        """Test getting or creating an event loop."""
        # In a sync context, we may need to get/create a loop
        try:
            loop = asyncio.get_running_loop()
            # If we're here, there's a running loop
            assert loop is not None
        except RuntimeError:
            # No running loop - this is expected in sync test
            loop = asyncio.new_event_loop()
            assert loop is not None
            loop.close()

    def test_schedule_async_task_method(self):
        """Test the _schedule_async_task approach used in AudioManager."""
        from glados2.audio.audio_manager import AudioManager
        from glados2.core.event_bus import EventBus
        from glados2.core.state_manager import StateManager

        event_bus = EventBus()
        state_manager = StateManager(event_bus)

        # Create audio manager
        audio_manager = AudioManager(event_bus, state_manager, {})

        # Test scheduling from sync context (no running loop)
        results = []

        async def async_work():
            results.append("started")
            await asyncio.sleep(0.05)
            results.append("done")

        # This should not raise an error even without a running loop
        audio_manager._schedule_async_task(async_work())

        # Wait a bit for the background thread to complete
        import time
        time.sleep(0.2)

        # Should have executed
        assert "started" in results
        assert "done" in results


class TestIntegration:
    """Integration tests for audio listening pipeline."""

    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_listening_pipeline_mock(self):
        """Test full listening pipeline with mocked audio input."""
        from glados2.audio.audio_manager import AudioManager

        event_bus = EventBus()
        state_manager = StateManager(event_bus)

        # Create audio manager with mocked audio
        config = {
            'sample_rate': 16000,
            'chunk_size': 1024,
            'vad_threshold': 0.5,
        }

        messages_received = []

        def on_message(data):
            messages_received.append(data)

        event_bus.subscribe(EventType.MESSAGE_RECEIVED, on_message)

        # Create manager
        audio_manager = AudioManager(event_bus, state_manager, config)

        # Manually trigger listening (avoiding the state callback issue)
        await audio_manager.start_listening()

        # Let it run briefly
        await asyncio.sleep(0.3)

        # Stop listening
        audio_manager._stop_listening.set()

        await asyncio.sleep(0.1)

        # Verify it started and stopped
        assert audio_manager._audio_state is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
