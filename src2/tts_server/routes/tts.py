"""Core TTS endpoints matching ElevenLabs API."""

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import Response, StreamingResponse
from loguru import logger

from ..engine.audio_converter import AudioConverter, OutputFormat
from ..models.requests import TTSRequest

router = APIRouter(tags=["text-to-speech"])


@router.post("/text-to-speech/{voice_id}")
async def text_to_speech(
    voice_id: str,
    body: TTSRequest,
    request: Request,
    output_format: str = Query(default="mp3_44100_128"),
    enable_logging: bool = Query(default=True),
):
    """Convert text to speech audio. Returns the complete audio file."""
    engine = request.app.state.engine
    fmt = OutputFormat.parse(output_format)

    speed = body.voice_settings.speed if body.voice_settings else 1.0

    logger.info(f"TTS request: type=standard, voice={voice_id}, format={output_format}, streaming=false")
    logger.info(f"TTS text: {body.text[:200]}")

    try:
        audio_float32, native_sr = await engine.synthesize(
            text=body.text,
            voice_id=voice_id,
            model_id=body.model_id,
            speed=speed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    audio_bytes = AudioConverter.convert(audio_float32, native_sr, fmt)

    logger.info(f"TTS response: voice={voice_id}, format={output_format}, size={len(audio_bytes)} bytes")

    return Response(
        content=audio_bytes,
        media_type=fmt.content_type,
        headers={
            "Content-Disposition": f'attachment; filename="speech.{fmt.extension}"',
        },
    )


@router.post("/text-to-speech/{voice_id}/stream")
async def text_to_speech_stream(
    voice_id: str,
    body: TTSRequest,
    request: Request,
    output_format: str = Query(default="mp3_44100_128"),
    enable_logging: bool = Query(default=True),
):
    """Convert text to speech audio with streaming response."""
    engine = request.app.state.engine
    fmt = OutputFormat.parse(output_format)

    speed = body.voice_settings.speed if body.voice_settings else 1.0

    logger.info(f"TTS request: type=stream, voice={voice_id}, format={output_format}, streaming=true")
    logger.info(f"TTS text: {body.text[:200]}")

    try:
        audio_float32, native_sr = await engine.synthesize(
            text=body.text,
            voice_id=voice_id,
            model_id=body.model_id,
            speed=speed,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    audio_bytes = AudioConverter.convert(audio_float32, native_sr, fmt)

    logger.info(f"TTS response: voice={voice_id}, format={output_format}, size={len(audio_bytes)} bytes")

    async def generate_chunks():
        chunk_size = 4096
        for i in range(0, len(audio_bytes), chunk_size):
            yield audio_bytes[i : i + chunk_size]

    return StreamingResponse(
        generate_chunks(),
        media_type=fmt.content_type,
    )
