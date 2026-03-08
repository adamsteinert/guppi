"""WebSocket TTS endpoint matching ElevenLabs stream-input API."""

import json

from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from loguru import logger

from ..engine.audio_converter import AudioConverter, OutputFormat

router = APIRouter(tags=["websocket"])


@router.websocket("/text-to-speech/{voice_id}/stream-input")
async def websocket_tts(
    websocket: WebSocket,
    voice_id: str,
    model_id: str = Query(default="eleven_multilingual_v2"),
    output_format: str = Query(default="mp3_44100_128"),
):
    """WebSocket endpoint for real-time text-to-speech streaming.

    Protocol:
    1. Client sends init message: {"text": " ", "voice_settings": {...}, "xi_api_key": "..."}
    2. Client sends text chunks: {"text": "chunk", "try_trigger_generation": true}
    3. Client sends end signal: {"text": ""}
    4. Server responds with binary audio chunks
    """
    await websocket.accept()

    engine = websocket.app.state.engine
    config = websocket.app.state.config
    fmt = OutputFormat.parse(output_format)

    text_buffer = ""
    voice_settings: dict = {}
    speed = 1.0

    logger.info(f"WebSocket TTS connection: voice={voice_id}, format={output_format}")

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON received: {raw[:100]}")
                continue

            text = data.get("text", "")

            # Init message (text is single space)
            if text == " " and ("voice_settings" in data or "xi_api_key" in data):
                voice_settings = data.get("voice_settings", {})
                speed = voice_settings.get("speed", 1.0) if voice_settings else 1.0

                # Auth check
                if config.api_key_required and config.api_key:
                    api_key = data.get("xi_api_key")
                    if api_key != config.api_key:
                        await websocket.close(code=4001, reason="Invalid API key")
                        return
                continue

            # End-of-stream (empty text)
            if text == "":
                if text_buffer.strip():
                    await _synthesize_and_send(websocket, engine, text_buffer, voice_id, model_id, fmt, speed)
                    text_buffer = ""
                # Signal end with empty bytes
                await websocket.send_bytes(b"")
                continue

            # Accumulate text
            text_buffer += text

            # Flush if buffer is large enough and generation is requested
            if data.get("try_trigger_generation", True) and len(text_buffer) > 200:
                await _synthesize_and_send(websocket, engine, text_buffer, voice_id, model_id, fmt, speed)
                text_buffer = ""

    except WebSocketDisconnect:
        logger.info("WebSocket TTS client disconnected")
    except Exception as e:
        logger.error(f"WebSocket TTS error: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except Exception:
            pass


async def _synthesize_and_send(
    websocket: WebSocket,
    engine,
    text: str,
    voice_id: str,
    model_id: str,
    fmt: OutputFormat,
    speed: float,
) -> None:
    """Synthesize text and send audio chunks over WebSocket."""
    try:
        audio_float32, sr = await engine.synthesize(text, voice_id, model_id, speed)
        audio_bytes = AudioConverter.convert(audio_float32, sr, fmt)

        # Send in chunks
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            await websocket.send_bytes(audio_bytes[i : i + chunk_size])

    except Exception as e:
        logger.error(f"WebSocket synthesis error: {e}")
        # Send error as JSON
        await websocket.send_json({"error": str(e)})
