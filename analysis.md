# GLaDOS 2.0 TUI Hang Analysis

The TUI application hangs when listening for voice activation or shortly after playing back audio. Below are the suspected root causes, organized by severity.

---

## Critical Issue 1: `_schedule_async_task` runs in a separate thread with its own event loop, deadlocking Textual

**Files:** `src2/glados2/audio/audio_manager.py:203-223`, `src2/glados2/llm/llm_manager.py:87-109`

When the TUI runs via `app.run()`, Textual owns the asyncio event loop on the main thread. But EventBus callbacks are **synchronous** (`publish()` at `src2/glados2/core/event_bus.py:76`), and when those sync callbacks call `_schedule_async_task()`, the `asyncio.get_running_loop()` call may or may not find Textual's loop depending on which thread is calling.

**The critical problem:** When `_schedule_async_task` falls through to the `RuntimeError` branch (no running loop found), it spawns `asyncio.run(coro)` in a **new daemon thread with a completely separate event loop**. This means:
- `start_listening()` runs in thread A with loop A
- `synthesize_and_play()` runs in thread B with loop B
- Both call `self._state_manager.set_state()` which publishes events via `EventBus.publish()`
- Those events call more subscribers that try to schedule more async tasks
- **Result:** Multiple independent event loops fighting over shared mutable state, with `threading.Event` and `asyncio.Event` objects that are **not cross-loop safe**

**Specific hang scenario:** `_request_cancelled` in LLMManager is created as `asyncio.Event()` at `src2/glados2/llm/llm_manager.py:53`, but `asyncio.Event` is **not thread-safe** and must only be used within a single event loop. When `send_message` runs in one loop and `cancel_current_request` is called from another, the event may never be seen.

---

## Critical Issue 2: State transition to IDLE triggers `stop_all_audio()` which kills the device monitor permanently

**File:** `src2/glados2/audio/audio_manager.py:187-201`

```python
def _on_state_changed(self, event_data: dict) -> None:
    new_state = event_data.get("new_state")
    if new_state == AppState.IDLE:
        self._schedule_async_task(self.stop_all_audio())
```

Every single transition to `IDLE` calls `stop_all_audio()`, which at line 301 **permanently stops the device monitor** (`self._device_monitor.stop()`) and joins the monitor thread with a 3-second timeout. After the first IDLE transition, the device monitor is dead. But more critically:

- After audio playback completes, `synthesize_and_play()` sets state to IDLE (line 616)
- This triggers `_on_state_changed` -> `stop_all_audio()`
- `stop_all_audio()` calls `self._device_monitor.stop()` which calls `self._monitor_thread.join(timeout=3.0)` -- **a 3-second blocking call inside an EventBus callback**
- EventBus callbacks are synchronous, so this **blocks the entire event dispatch chain for 3 seconds**

---

## Critical Issue 3: Recursive event publishing inside EventBus callbacks causes re-entrant state transitions

**Files:** `src2/glados2/core/state_manager.py:67`, `src2/glados2/audio/audio_manager.py:187-201`

`StateManager.set_state()` holds `self._lock` (an `RLock`) and calls `self._event_bus.publish(STATE_CHANGED)`. Subscribers like `AudioManager._on_state_changed` react by calling `set_state()` again (e.g., setting IDLE after silence timeout at line 389), which re-enters the `RLock` and publishes **another** STATE_CHANGED event while the first one is still being dispatched.

This creates nested event dispatch chains that can pile up unpredictably. For example:
1. `set_state(IDLE)` -> publishes STATE_CHANGED
2. AudioManager._on_state_changed -> schedules `stop_all_audio()`
3. Meanwhile, LLM finishes and tries `set_state(IDLE)` from its thread
4. State is already IDLE, but now there's a race between the stop_all_audio from step 2 and whatever comes next

---

## Critical Issue 4: `AudioDeviceMonitor` calls `sd._terminate()` / `sd._initialize()` every 2 seconds, racing with active streams

**File:** `src2/glados2/audio/audio_manager.py:79-83`

```python
sd._terminate()
sd._initialize()
```

This runs every 2 seconds in a background thread. If an `sd.InputStream` or `sd.play()` is active at the same time, **terminating PortAudio underneath them will crash or hang the audio stream**. The `_refresh_lock` only prevents concurrent refreshes -- it doesn't synchronize with `sd.InputStream.start()`, `sd.play()`, or `sd.stop()`.

**This is the most likely cause of hanging during listening or after playback** -- the device monitor poll fires while audio I/O is active, yanks PortAudio out from under it, and the stream callback never returns.

---

## Critical Issue 5: `asyncio.Queue` used across thread boundaries

**File:** `src2/glados2/audio/audio_manager.py:332-340`

The `audio_callback` from sounddevice runs on a **PortAudio callback thread**, and calls `audio_queue.put_nowait()` on an `asyncio.Queue`. But `asyncio.Queue` is **not thread-safe** -- it's designed for use within a single asyncio event loop. Calling it from a PortAudio thread can corrupt internal state or silently lose data. This should use `queue.Queue` (from stdlib) or `janus` for cross-thread async queues.

---

## Moderate Issue 6: `_play_audio_async` spin-waits with `time.sleep` in a daemon thread

**File:** `src2/glados2/audio/audio_manager.py:553-559`

```python
def check_playback():
    import time
    time.sleep(len(audio_data) / self._tts.sample_rate)
    playback_finished_callback()
```

This calculates the expected playback duration and sleeps for that exact time. If `sd.play()` takes slightly longer (e.g., due to buffer underruns, device switching, or sample rate mismatch), the finished callback fires **before playback actually ends**. Conversely, if `sd.stop()` is called (interrupt), the thread keeps sleeping until the calculated duration elapses, during which the state machine thinks playback is still active.

---

## Summary of likely hang scenarios

| Symptom | Root Cause |
|---------|-----------|
| Hang while listening | AudioDeviceMonitor calls `sd._terminate()` while InputStream is active (Issue 4) |
| Hang after audio playback | `stop_all_audio()` blocks for 3s joining the monitor thread inside an EventBus callback (Issue 2) |
| Hang after audio playback | Multiple event loops fighting over state, async tasks scheduled on dead loops (Issue 1) |
| Intermittent freeze | `asyncio.Queue.put_nowait()` from PortAudio thread corrupts queue state (Issue 5) |
| UI becomes unresponsive | Re-entrant state transitions cause cascading EventBus dispatches (Issue 3) |
