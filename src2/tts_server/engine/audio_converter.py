"""Audio format conversion: float32 numpy arrays to MP3/WAV/PCM/ulaw."""

import io
import wave
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from loguru import logger


class FormatType(Enum):
    MP3 = "mp3"
    PCM = "pcm"
    WAV = "wav"
    ULAW = "ulaw"
    ALAW = "alaw"
    OPUS = "opus"


@dataclass
class OutputFormat:
    """Parsed output format specification."""

    format_type: FormatType
    sample_rate: int
    bitrate: Optional[int] = None  # For MP3/Opus

    @property
    def content_type(self) -> str:
        mapping = {
            FormatType.MP3: "audio/mpeg",
            FormatType.PCM: "application/octet-stream",
            FormatType.WAV: "audio/wav",
            FormatType.ULAW: "audio/basic",
            FormatType.ALAW: "audio/basic",
            FormatType.OPUS: "audio/opus",
        }
        return mapping[self.format_type]

    @property
    def extension(self) -> str:
        return self.format_type.value

    @staticmethod
    def parse(format_string: str) -> "OutputFormat":
        """Parse ElevenLabs format strings like 'mp3_44100_128', 'pcm_24000', 'ulaw_8000'."""
        parts = format_string.split("_")
        fmt = parts[0].lower()

        if fmt == "mp3":
            sr = int(parts[1]) if len(parts) > 1 else 44100
            br = int(parts[2]) if len(parts) > 2 else 128
            return OutputFormat(FormatType.MP3, sr, br)
        elif fmt == "pcm":
            sr = int(parts[1]) if len(parts) > 1 else 16000
            return OutputFormat(FormatType.PCM, sr)
        elif fmt == "wav":
            sr = int(parts[1]) if len(parts) > 1 else 44100
            return OutputFormat(FormatType.WAV, sr)
        elif fmt == "ulaw":
            sr = int(parts[1]) if len(parts) > 1 else 8000
            return OutputFormat(FormatType.ULAW, sr)
        elif fmt == "alaw":
            sr = int(parts[1]) if len(parts) > 1 else 8000
            return OutputFormat(FormatType.ALAW, sr)
        elif fmt == "opus":
            sr = int(parts[1]) if len(parts) > 1 else 48000
            br = int(parts[2]) if len(parts) > 2 else 128
            return OutputFormat(FormatType.OPUS, sr, br)
        else:
            logger.warning(f"Unknown format '{format_string}', defaulting to mp3_44100_128")
            return OutputFormat(FormatType.MP3, 44100, 128)


class AudioConverter:
    """Converts float32 numpy audio to various output formats."""

    @staticmethod
    def convert(audio: np.ndarray, source_sr: int, fmt: OutputFormat) -> bytes:
        """Convert float32 audio array to the requested output format."""
        # Resample if needed
        if source_sr != fmt.sample_rate:
            audio = AudioConverter._resample(audio, source_sr, fmt.sample_rate)

        # Convert to int16 PCM
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio / max(peak, 1.0)
        pcm_int16 = (audio * 32767).astype(np.int16)

        if fmt.format_type == FormatType.PCM:
            return pcm_int16.tobytes()

        elif fmt.format_type == FormatType.WAV:
            return AudioConverter._encode_wav(pcm_int16, fmt.sample_rate)

        elif fmt.format_type == FormatType.MP3:
            return AudioConverter._encode_mp3(pcm_int16, fmt.sample_rate, fmt.bitrate or 128)

        elif fmt.format_type in (FormatType.ULAW, FormatType.ALAW):
            return AudioConverter._encode_ulaw(pcm_int16)

        else:
            logger.warning(f"Unsupported format {fmt.format_type}, returning WAV")
            return AudioConverter._encode_wav(pcm_int16, fmt.sample_rate)

    @staticmethod
    def _resample(audio: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        """Resample audio to target sample rate."""
        if source_sr == target_sr:
            return audio

        num_samples = int(len(audio) * target_sr / source_sr)
        try:
            from scipy.signal import resample

            return resample(audio, num_samples).astype(np.float32)
        except ImportError:
            # Fallback: linear interpolation
            logger.debug("scipy not available, using linear interpolation for resampling")
            indices = np.linspace(0, len(audio) - 1, num_samples)
            return np.interp(indices, np.arange(len(audio)), audio).astype(np.float32)

    @staticmethod
    def _encode_wav(pcm_int16: np.ndarray, sample_rate: int) -> bytes:
        """Encode int16 PCM to WAV bytes."""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_int16.tobytes())
        return buf.getvalue()

    @staticmethod
    def _encode_mp3(pcm_int16: np.ndarray, sample_rate: int, bitrate: int) -> bytes:
        """Encode int16 PCM to MP3 bytes using lameenc."""
        try:
            import lameenc

            encoder = lameenc.Encoder()
            encoder.set_channels(1)
            encoder.set_bit_rate(bitrate)
            encoder.set_in_sample_rate(sample_rate)
            encoder.set_out_sample_rate(sample_rate)
            encoder.set_quality(2)  # High quality

            mp3_data = encoder.encode(pcm_int16.tobytes())
            mp3_data += encoder.flush()
            return bytes(mp3_data)
        except ImportError:
            logger.warning("lameenc not installed, falling back to WAV output for MP3 request")
            return AudioConverter._encode_wav(pcm_int16, sample_rate)

    @staticmethod
    def _encode_ulaw(pcm_int16: np.ndarray) -> bytes:
        """Encode int16 PCM to u-law (G.711) format."""
        import audioop

        return audioop.lin2ulaw(pcm_int16.tobytes(), 2)
