"""Audio management system for GLaDOS 2.0."""

import asyncio
from typing import Optional, Callable, Tuple
from enum import Enum
import threading
import time
import numpy as np

from loguru import logger
import sounddevice as sd

from ..core.event_bus import EventBus, EventType
from ..core.state_manager import StateManager, AppState
from .vad_processor import VADProcessor
from .asr_processor import ASRProcessor
from .tts_processor import TTSProcessor


class AudioState(Enum):
    """Audio system states."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    PLAYING = "playing"
    MUTED = "muted"
    ERROR = "error"


class AudioDeviceMonitor:
    """
    Monitors system audio devices for changes.

    Polls for device changes periodically and publishes events when
    the default input or output device changes.
    """

    # Class-level lock to prevent conflicts with audio operations
    # when refreshing the device cache
    _refresh_lock = threading.Lock()

    def __init__(self, event_bus: EventBus, poll_interval: float = 2.0):
        self._event_bus = event_bus
        self._poll_interval = poll_interval
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._last_input_device: Optional[str] = None
        self._last_output_device: Optional[str] = None

    def start(self) -> None:
        """Start monitoring for device changes."""
        if self._running:
            return

        self._running = True
        # Initialize with current devices
        self._last_input_device, self._last_output_device = self._get_current_devices()

        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Audio device monitor started")

    def stop(self) -> None:
        """Stop monitoring for device changes."""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=3.0)
            self._monitor_thread = None
        logger.info("Audio device monitor stopped")

    def _get_current_devices(self) -> Tuple[Optional[str], Optional[str]]:
        """Get the names of current default input and output devices.

        Forces a refresh of the PortAudio device cache to detect
        system audio device changes (e.g., switching from AirPods to speakers).
        """
        with self._refresh_lock:
            try:
                # Force PortAudio to refresh its device cache
                # This is necessary because PortAudio caches the device list
                # and won't detect changes without re-initialization
                sd._terminate()
                sd._initialize()

                input_device = sd.query_devices(kind='input')
                output_device = sd.query_devices(kind='output')

                input_name = input_device.get('name') if input_device else None
                output_name = output_device.get('name') if output_device else None

                return input_name, output_name
            except Exception as e:
                logger.warning(f"Error querying audio devices: {e}")
                return None, None

    def _monitor_loop(self) -> None:
        """Main monitoring loop that polls for device changes."""
        while self._running:
            try:
                current_input, current_output = self._get_current_devices()

                input_changed = current_input != self._last_input_device
                output_changed = current_output != self._last_output_device

                if input_changed or output_changed:
                    logger.info(f"Audio device change detected - Input: {self._last_input_device} -> {current_input}, Output: {self._last_output_device} -> {current_output}")

                    self._event_bus.publish(EventType.AUDIO_DEVICE_CHANGED, {
                        "input_device": current_input,
                        "output_device": current_output,
                        "input_changed": input_changed,
                        "output_changed": output_changed,
                        "previous_input": self._last_input_device,
                        "previous_output": self._last_output_device,
                    })

                    self._last_input_device = current_input
                    self._last_output_device = current_output

            except Exception as e:
                logger.warning(f"Error in device monitor loop: {e}")

            time.sleep(self._poll_interval)


class AudioManager:
    """
    Manages all audio operations for GLaDOS 2.0.
    
    This replaces the complex threading and queue management in the original
    with a cleaner, more stable architecture.
    """
    
    def __init__(self, event_bus: EventBus, state_manager: StateManager, config: dict = None):
        self._event_bus = event_bus
        self._state_manager = state_manager
        self._audio_state = AudioState.IDLE
        
        # Audio settings from config
        config = config or {}
        self._sample_rate = config.get('sample_rate', 16000)
        self._chunk_size = config.get('chunk_size', 1024)
        self._microphone_muted = config.get('microphone_muted', False)
        self._speaker_muted = config.get('speaker_muted', False)
        self._volume = config.get('volume', 1.0)
        
        # Audio processors
        self._vad = VADProcessor(
            sample_rate=self._sample_rate,
            threshold=config.get('vad_threshold', 0.8),
            min_speech_duration_ms=config.get('min_speech_duration_ms', 250),
            min_silence_duration_ms=config.get('min_silence_duration_ms', 1000)
        )
        self._asr = ASRProcessor(sample_rate=self._sample_rate)
        self._tts = TTSProcessor(
            voice=config.get('voice', 'glados'),
            sample_rate=config.get('tts_sample_rate', 22050)
        )
        
        # Audio streaming
        self._input_stream: Optional[sd.InputStream] = None
        self._output_stream: Optional[sd.OutputStream] = None
        self._listening_task: Optional[asyncio.Task] = None
        self._playback_task: Optional[asyncio.Task] = None
        
        # Control flags
        self._stop_listening = threading.Event()
        self._playback_cancelled = threading.Event()

        # Listening state tracking
        self._last_audio_time: float = 0.0
        self._silence_timeout_seconds: float = 5.0  # Auto-stop after 5 seconds of silence
        self._force_process_audio = threading.Event()  # Signal to process collected audio now

        # Audio device monitor
        self._device_monitor = AudioDeviceMonitor(event_bus, poll_interval=2.0)

        # Subscribe to relevant events
        self._event_bus.subscribe(EventType.STATE_CHANGED, self._on_state_changed)
        self._event_bus.subscribe(EventType.MESSAGE_RECEIVED, self._on_message_received)
        self._event_bus.subscribe(EventType.LISTENING_STOPPED, self._on_listening_stopped)
        self._event_bus.subscribe(EventType.AUDIO_DEVICE_CHANGED, self._on_audio_device_changed)

        # Start device monitoring
        self._device_monitor.start()
        
    def _on_state_changed(self, event_data: dict) -> None:
        """Handle application state changes.

        Note: This is called from a sync context (EventBus.publish),
        so we need to safely schedule async tasks.
        """
        new_state = event_data.get("new_state")

        if new_state == AppState.LISTENING:
            self._schedule_async_task(self.start_listening())
        elif new_state == AppState.PLAYING_AUDIO:
            # Audio playback will be started via play_audio method
            pass
        elif new_state == AppState.IDLE:
            self._schedule_async_task(self.stop_all_audio())

    def _schedule_async_task(self, coro) -> None:
        """Safely schedule an async task from sync or async context.

        This handles the case where we're called from a sync callback
        (like EventBus.publish) and need to run async code.
        """
        try:
            # Try to get the running event loop
            loop = asyncio.get_running_loop()
            # We have a running loop, create task directly
            loop.create_task(coro)
        except RuntimeError:
            # No running loop - need to run in a new thread with its own loop
            def run_in_thread():
                try:
                    asyncio.run(coro)
                except Exception as e:
                    logger.error(f"Error running async task in thread: {e}")

            thread = threading.Thread(target=run_in_thread, daemon=True)
            thread.start()
            
    def _on_message_received(self, event_data: dict) -> None:
        """Handle incoming messages and trigger TTS for assistant responses."""
        role = event_data.get("role")
        content = event_data.get("content")

        if role == "assistant" and content:
            # Trigger TTS for assistant responses
            logger.info(f"Processing assistant response for TTS: {content[:50]}...")
            self._schedule_async_task(self.synthesize_and_play(content))

    def _on_listening_stopped(self, event_data: dict) -> None:
        """Handle user request to stop listening and process collected audio."""
        reason = event_data.get("reason", "unknown")
        logger.info(f"Listening stopped: {reason}")
        # Signal the listen loop to process any collected audio immediately
        self._force_process_audio.set()
        self._stop_listening.set()

    def _on_audio_device_changed(self, event_data: dict) -> None:
        """Handle system audio device changes."""
        input_changed = event_data.get("input_changed", False)
        output_changed = event_data.get("output_changed", False)
        new_input = event_data.get("input_device")
        new_output = event_data.get("output_device")

        logger.info(f"Audio device changed - Input: {new_input}, Output: {new_output}")

        # If we're currently listening and input device changed, restart the stream
        if input_changed and self._audio_state == AudioState.LISTENING:
            logger.info("Input device changed while listening - restarting audio capture")
            # Close current input stream
            if self._input_stream:
                try:
                    self._input_stream.close()
                    self._input_stream = None
                except Exception as e:
                    logger.warning(f"Error closing input stream: {e}")

            # The stream will be recreated on next listen iteration

        # If we're currently playing and output device changed, we may need to handle it
        # sounddevice typically handles this automatically, but log for debugging
        if output_changed and self._audio_state == AudioState.PLAYING:
            logger.info("Output device changed while playing - audio may switch automatically")

    async def start_listening(self) -> None:
        """Start listening for voice input.

        This method runs the listen loop and blocks until listening stops.
        """
        if self._microphone_muted:
            logger.warning("Cannot start listening: microphone is muted")
            return

        if self._audio_state == AudioState.LISTENING:
            logger.warning("Already listening")
            return

        logger.info("Starting audio listening...")
        self._audio_state = AudioState.LISTENING
        self._stop_listening.clear()
        self._force_process_audio.clear()
        self._last_audio_time = time.time()
        self._vad.reset()

        self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "listening"})

        # Run the listen loop directly (await it so it doesn't exit immediately)
        # This is important when running via asyncio.run() in a thread
        await self._listen_loop()
        
    async def stop_all_audio(self) -> None:
        """Stop all audio operations."""
        logger.info("Stopping all audio operations...")

        # Stop device monitor
        if self._device_monitor:
            self._device_monitor.stop()

        # Stop listening
        self._stop_listening.set()
        if self._listening_task and not self._listening_task.done():
            self._listening_task.cancel()

        # Stop playback
        self._playback_cancelled.set()
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()

        # Close audio streams
        if self._input_stream:
            self._input_stream.close()
            self._input_stream = None

        if self._output_stream:
            self._output_stream.close()
            self._output_stream = None

        self._audio_state = AudioState.IDLE
        self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "idle"})
        
    async def _listen_loop(self) -> None:
        """Main audio listening loop with VAD and ASR processing."""
        try:
            logger.info("Starting audio capture...")

            # Audio callback for input stream
            audio_queue = asyncio.Queue()

            def audio_callback(indata, frames, time_info, status):
                if status:
                    logger.warning(f"Audio input status: {status}")
                # Convert to numpy array and put in queue
                audio_data = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
                try:
                    audio_queue.put_nowait(audio_data.astype(np.float32))
                except asyncio.QueueFull:
                    pass  # Drop frame if queue is full

            # Start input stream
            self._input_stream = sd.InputStream(
                samplerate=self._sample_rate,
                channels=1,
                dtype=np.float32,
                blocksize=self._chunk_size,
                callback=audio_callback
            )
            self._input_stream.start()

            logger.info("Audio capture started, waiting for speech...")

            # Process audio chunks
            while not self._stop_listening.is_set():
                try:
                    # Get audio chunk with timeout
                    audio_chunk = await asyncio.wait_for(audio_queue.get(), timeout=0.1)

                    # Update last audio time when we receive audio
                    current_time = time.time()

                    # Process with VAD
                    is_voice, completed_audio = self._vad.process_audio_chunk(audio_chunk)

                    if is_voice:
                        # Voice detected, update the timer
                        self._last_audio_time = current_time

                    if completed_audio is not None:
                        logger.info("Speech segment detected, transcribing...")
                        await self._process_speech_segment(completed_audio)
                        # Reset timer after processing
                        self._last_audio_time = time.time()

                    # Check for silence timeout (5 seconds without voice)
                    silence_duration = current_time - self._last_audio_time
                    if silence_duration >= self._silence_timeout_seconds:
                        logger.info(f"Silence timeout ({self._silence_timeout_seconds}s) - stopping listening")
                        # Process any collected audio before stopping
                        if self._vad.collected_audio:
                            collected = np.concatenate(self._vad.collected_audio)
                            if len(collected) > self._sample_rate * 0.25:  # At least 250ms
                                logger.info("Processing collected audio before timeout stop")
                                await self._process_speech_segment(collected)
                        self._stop_listening.set()
                        self._state_manager.set_state(AppState.IDLE)
                        self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "ready"})
                        break

                except asyncio.TimeoutError:
                    # Check for force process signal (user pressed 'l' to stop)
                    if self._force_process_audio.is_set():
                        logger.info("Force processing collected audio")
                        if self._vad.collected_audio:
                            collected = np.concatenate(self._vad.collected_audio)
                            if len(collected) > self._sample_rate * 0.1:  # At least 100ms
                                await self._process_speech_segment(collected)
                        self._force_process_audio.clear()
                        break

                    # Also check silence timeout during timeouts
                    current_time = time.time()
                    silence_duration = current_time - self._last_audio_time
                    if silence_duration >= self._silence_timeout_seconds:
                        logger.info(f"Silence timeout during wait ({self._silence_timeout_seconds}s)")
                        if self._vad.collected_audio:
                            collected = np.concatenate(self._vad.collected_audio)
                            if len(collected) > self._sample_rate * 0.25:
                                await self._process_speech_segment(collected)
                        self._stop_listening.set()
                        self._state_manager.set_state(AppState.IDLE)
                        self._event_bus.publish(EventType.AUDIO_STATUS_CHANGED, {"status": "ready"})
                        break

                    continue  # Check stop flag

                except Exception as e:
                    logger.error(f"Error in listen loop: {e}")
                    break

        except Exception as e:
            logger.error(f"Audio listening error: {e}")
        finally:
            if self._input_stream:
                self._input_stream.close()
                self._input_stream = None
            self._vad.reset()
            logger.info("Audio listening stopped")
            
    async def _process_speech_segment(self, audio_data: np.ndarray) -> None:
        """Process a detected speech segment through ASR and send to LLM."""
        try:
            # Update state to processing
            self._state_manager.set_state(AppState.PROCESSING_AUDIO)
            
            # Transcribe audio
            transcription = await self._asr.transcribe_audio(audio_data)
            
            if transcription:
                logger.info(f"Transcription: {transcription}")

                # Publish transcription event
                # Note: LLM manager will handle state transition to CALLING_LLM
                # when it receives this event and starts processing
                self._event_bus.publish(EventType.MESSAGE_RECEIVED, {
                    "role": "user",
                    "content": transcription,
                    "audio_duration": len(audio_data) / self._sample_rate
                })

                # Stay in PROCESSING_AUDIO state - LLM manager will transition
                # to CALLING_LLM when it starts processing the message

            else:
                logger.warning("Transcription failed")
                self._state_manager.set_state(AppState.IDLE)
                
        except Exception as e:
            logger.error(f"Error processing speech segment: {e}")
            self._state_manager.set_state(AppState.ERROR)
        
    async def play_audio(self, audio_data: np.ndarray, interruptible: bool = True) -> bool:
        """
        Play audio data with proper cancellation support.
        
        Args:
            audio_data: Audio samples as numpy array
            interruptible: Whether playback can be interrupted
            
        Returns:
            bool: True if playback completed successfully, False if interrupted
        """
        if self._speaker_muted:
            logger.info("Audio playback skipped: speaker is muted")
            return False
            
        if audio_data is None or len(audio_data) == 0:
            logger.warning("Empty audio data for playback")
            return False
            
        logger.info(f"Starting audio playback: {len(audio_data)} samples")
        self._audio_state = AudioState.PLAYING
        self._playback_cancelled.clear()
        
        # Calculate duration
        duration = len(audio_data) / self._tts.sample_rate
        
        self._event_bus.publish(EventType.AUDIO_PLAYBACK_STARTED, {
            "interruptible": interruptible,
            "duration": duration
        })
        
        try:
            # Apply volume
            audio_data = audio_data * self._volume
            
            # Create playback task
            self._playback_task = asyncio.create_task(
                self._play_audio_async(audio_data, interruptible)
            )
            
            result = await self._playback_task
            
            if result:
                logger.info("Audio playback completed successfully")
                self._event_bus.publish(EventType.AUDIO_PLAYBACK_COMPLETED, {"completed": True})
            else:
                logger.info("Audio playback was interrupted")
                
            return result
            
        except asyncio.CancelledError:
            logger.info("Audio playback cancelled")
            return False
        except Exception as e:
            logger.error(f"Error during audio playback: {e}")
            self._audio_state = AudioState.ERROR
            return False
        finally:
            if self._audio_state == AudioState.PLAYING:
                self._audio_state = AudioState.IDLE
                
    async def _play_audio_async(self, audio_data: np.ndarray, interruptible: bool) -> bool:
        """Asynchronous audio playback with sounddevice."""
        try:
            # Ensure audio is 1D for mono playback
            if audio_data.ndim > 1:
                audio_data = audio_data.flatten()

            # Use sounddevice for playback
            playback_finished = asyncio.Event()
            playback_success = [True]  # Use list to modify from callback

            def playback_finished_callback():
                playback_finished.set()

            def playback_error_callback(error):
                logger.error(f"Audio playback error: {error}")
                playback_success[0] = False
                playback_finished.set()

            # Start playback (mono audio - channels inferred from shape)
            sd.play(
                audio_data,
                samplerate=self._tts.sample_rate,
                blocking=False
            )
            
            # Simulate finished callback since sd.play doesn't have one
            def check_playback():
                import time
                time.sleep(len(audio_data) / self._tts.sample_rate)
                playback_finished_callback()
                
            import threading
            threading.Thread(target=check_playback, daemon=True).start()
            
            # Wait for playback to finish or cancellation
            while not playback_finished.is_set():
                if self._playback_cancelled.is_set():
                    sd.stop()  # Stop playback
                    return False
                    
                await asyncio.sleep(0.01)  # Check every 10ms
                
            return playback_success[0]
            
        except Exception as e:
            logger.error(f"Error in async audio playback: {e}")
            return False
                
    async def synthesize_and_play(self, text: str, voice: str = None) -> bool:
        """
        Synthesize text to speech and play it.
        
        Args:
            text: Text to synthesize
            voice: Voice to use (optional)
            
        Returns:
            bool: True if synthesis and playback succeeded
        """
        try:
            # Set state to TTS generation
            self._state_manager.set_state(AppState.GENERATING_TTS)
            
            self._event_bus.publish(EventType.TTS_STARTED, {
                "text": text,
                "voice": voice or self._tts.voice
            })
            
            # Synthesize speech
            audio_data = await self._tts.synthesize_speech(text, voice)
            
            if audio_data is None:
                logger.error("TTS synthesis failed")
                self._state_manager.set_state(AppState.ERROR)
                return False
                
            self._event_bus.publish(EventType.TTS_COMPLETED, {
                "success": True,
                "duration": len(audio_data) / self._tts.sample_rate
            })
            
            # Set state to playing audio
            self._state_manager.set_state(AppState.PLAYING_AUDIO)
            
            # Play the synthesized audio
            success = await self.play_audio(audio_data, interruptible=True)
            
            # Return to idle state
            if success:
                self._state_manager.set_state(AppState.IDLE)
            else:
                self._state_manager.set_state(AppState.ERROR)
                
            return success
            
        except Exception as e:
            logger.error(f"Error in synthesize and play: {e}")
            self._state_manager.set_state(AppState.ERROR)
            return False
    
    def interrupt_playback(self) -> None:
        """Interrupt current audio playback."""
        if self._playback_task and not self._playback_task.done():
            logger.info("Interrupting audio playback")
            self._playback_cancelled.set()
            
    def set_microphone_muted(self, muted: bool) -> None:
        """Set microphone mute state."""
        self._microphone_muted = muted
        logger.info(f"Microphone {'muted' if muted else 'unmuted'}")
        
    def set_speaker_muted(self, muted: bool) -> None:
        """Set speaker mute state."""
        self._speaker_muted = muted
        logger.info(f"Speaker {'muted' if muted else 'unmuted'}")
        
    def set_volume(self, volume: float) -> None:
        """Set audio volume (0.0 to 1.0)."""
        self._volume = max(0.0, min(1.0, volume))
        logger.info(f"Volume set to {self._volume:.1%}")
        
    def get_audio_state(self) -> AudioState:
        """Get current audio state."""
        return self._audio_state
        
    def is_microphone_muted(self) -> bool:
        """Check if microphone is muted."""
        return self._microphone_muted
        
    def is_speaker_muted(self) -> bool:
        """Check if speaker is muted."""
        return self._speaker_muted
        
    def get_volume(self) -> float:
        """Get current volume."""
        return self._volume