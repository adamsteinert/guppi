#!/usr/bin/env python3
"""
Validate TTS with bm_george voice.

This script tests the text-to-speech synthesis with the Kokoro bm_george voice
and plays the audio so you can hear it.
"""

import asyncio
import sys
from pathlib import Path
import numpy as np

# Add src2 to path
sys.path.insert(0, str(Path(__file__).parent))

from glados2.audio.tts_processor import TTSProcessor
from loguru import logger

try:
    import sounddevice as sd
except ImportError:
    logger.error("sounddevice not available - cannot play audio")
    sd = None

try:
    import soundfile as sf
except ImportError:
    logger.warning("soundfile not available - cannot save audio files")
    sf = None


def print_header(text: str):
    """Print a formatted header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def print_success(text: str):
    """Print success message."""
    print(f"✅ {text}")


def print_error(text: str):
    """Print error message."""
    print(f"❌ {text}")


def print_info(text: str):
    """Print info message."""
    print(f"ℹ️  {text}")


async def test_tts_initialization():
    """Test TTS processor initialization."""
    print_header("TEST 1: TTS Processor Initialization")

    tts = TTSProcessor(
        voice="bm_george",
        model_dir="models/TTS",
        sample_rate=22050
    )

    print_info(f"Voice: {tts.voice}")
    print_info(f"Sample Rate: {tts.sample_rate}")
    print_info(f"Model Directory: {tts.model_dir}")

    # Get model info
    info = tts.get_model_info()
    print_info(f"GLaDOS Model Loaded: {info['glados_loaded']}")
    print_info(f"Kokoro Model Loaded: {info['kokoro_loaded']}")
    print_info(f"Phonemizer Loaded: {info['phonemizer_loaded']}")

    if not info['kokoro_loaded']:
        print_error("Kokoro model not loaded - will use fallback synthesis")
        print_info("To use real model, ensure models/TTS/kokoro-v1.0.fp16.onnx exists")
    else:
        print_success("Kokoro model loaded successfully")

    return tts


async def test_available_voices(tts: TTSProcessor):
    """Test available voices."""
    print_header("TEST 2: Available Voices")

    voices = tts.get_available_voices()
    print_info(f"Total voices available: {len(voices)}")

    # Group by category
    glados_voices = [v for v in voices if v == "glados"]
    male_voices = [v for v in voices if v.startswith(("bm_", "am_"))]
    female_voices = [v for v in voices if v.startswith(("bf_", "af_"))]

    print("\nGLaDOS Voices:")
    for voice in glados_voices:
        print(f"  - {voice}")

    print("\nMale Voices (Kokoro):")
    for voice in sorted(male_voices):
        marker = "👉" if voice == "bm_george" else "  "
        print(f"{marker} - {voice}")

    print("\nFemale Voices (Kokoro):")
    for voice in sorted(female_voices):
        print(f"  - {voice}")

    if "bm_george" in voices:
        print_success("bm_george voice is available")
    else:
        print_error("bm_george voice not found in available voices")


async def test_synthesis_fallback(tts: TTSProcessor):
    """Test synthesis with fallback."""
    print_header("TEST 3: Fallback Synthesis")

    test_text = "Hello, this is George speaking in fallback mode."
    print_info(f"Synthesizing: '{test_text}'")

    audio = await tts.synthesize_speech(test_text)

    if audio is not None:
        print_success("Synthesis completed")
        print_info(f"Audio samples: {len(audio)}")
        print_info(f"Audio duration: {len(audio) / tts.sample_rate:.2f} seconds")
        print_info(f"Audio shape: {audio.shape}")
        print_info(f"Audio dtype: {audio.dtype}")
        print_info(f"Audio range: [{np.min(audio):.4f}, {np.max(audio):.4f}]")

        # Check audio quality
        dc_offset = abs(np.mean(audio))
        rms_energy = np.sqrt(np.mean(audio ** 2))
        print_info(f"DC offset: {dc_offset:.6f}")
        print_info(f"RMS energy: {rms_energy:.6f}")

        if dc_offset < 0.01:
            print_success("DC offset is good (< 0.01)")
        else:
            print_error(f"DC offset too high: {dc_offset:.6f}")

        if rms_energy > 0.001:
            print_success(f"RMS energy is good (> 0.001)")
        else:
            print_error(f"RMS energy too low: {rms_energy:.6f}")

        return audio
    else:
        print_error("Synthesis failed")
        return None


async def test_synthesis_with_model(tts: TTSProcessor):
    """Test synthesis with real Kokoro model if available."""
    print_header("TEST 4: Real Model Synthesis")

    info = tts.get_model_info()
    if not info['kokoro_loaded']:
        print_info("Kokoro model not loaded - skipping this test")
        return None

    test_text = "Hello, I am George, speaking with the Kokoro text-to-speech engine."
    print_info(f"Synthesizing: '{test_text}'")

    audio = await tts.synthesize_speech(test_text)

    if audio is not None:
        print_success("Real model synthesis completed")
        print_info(f"Audio samples: {len(audio)}")
        print_info(f"Audio duration: {len(audio) / tts.sample_rate:.2f} seconds")
        return audio
    else:
        print_error("Real model synthesis failed")
        return None


async def play_audio(audio: np.ndarray, sample_rate: int, description: str):
    """Play audio through speakers."""
    print_header(f"PLAYING: {description}")

    if sd is None:
        print_error("sounddevice not available - cannot play audio")
        return

    if audio is None:
        print_error("No audio to play")
        return

    print_info(f"Duration: {len(audio) / sample_rate:.2f} seconds")
    print_info("Playing audio... (press Ctrl+C to stop)")

    try:
        sd.play(audio, samplerate=sample_rate)
        sd.wait()
        print_success("Playback completed")
    except KeyboardInterrupt:
        sd.stop()
        print_info("Playback stopped by user")
    except Exception as e:
        print_error(f"Playback error: {e}")


async def save_audio(audio: np.ndarray, sample_rate: int, filename: str):
    """Save audio to file."""
    print_header(f"SAVING: {filename}")

    if sf is None:
        print_error("soundfile not available - cannot save audio")
        return

    if audio is None:
        print_error("No audio to save")
        return

    try:
        output_path = Path("output") / filename
        output_path.parent.mkdir(exist_ok=True)

        sf.write(str(output_path), audio, sample_rate)
        print_success(f"Audio saved to: {output_path}")
        print_info(f"File size: {output_path.stat().st_size / 1024:.2f} KB")
    except Exception as e:
        print_error(f"Failed to save audio: {e}")


async def test_multiple_sentences(tts: TTSProcessor):
    """Test synthesis with multiple sentences."""
    print_header("TEST 5: Multiple Sentences")

    sentences = [
        "Hello, I am George.",
        "This is a test of the text-to-speech system.",
        "The quick brown fox jumps over the lazy dog.",
        "How are you doing today?",
        "Testing, one, two, three."
    ]

    all_audio = []

    for i, sentence in enumerate(sentences, 1):
        print_info(f"[{i}/{len(sentences)}] Synthesizing: '{sentence}'")
        audio = await tts.synthesize_speech(sentence)

        if audio is not None:
            all_audio.append(audio)
            print_success(f"  ✓ Completed ({len(audio)} samples)")
        else:
            print_error(f"  ✗ Failed")

    if all_audio:
        # Concatenate with small gaps
        gap_samples = int(tts.sample_rate * 0.3)  # 300ms gap
        gap = np.zeros(gap_samples, dtype=np.float32)

        combined = []
        for audio in all_audio:
            combined.append(audio)
            combined.append(gap)

        combined_audio = np.concatenate(combined[:-1])  # Remove last gap

        print_success(f"Combined {len(all_audio)} sentences")
        print_info(f"Total duration: {len(combined_audio) / tts.sample_rate:.2f} seconds")

        return combined_audio
    else:
        print_error("No audio was synthesized")
        return None


async def main():
    """Main validation function."""
    print("\n" + "🎙️ " * 20)
    print_header("GLaDOS 2.0 - TTS Validation with bm_george Voice")
    print("This script will test text-to-speech synthesis with the Kokoro")
    print("bm_george voice and play the results through your speakers.")
    print("🎙️ " * 20)

    # Test 1: Initialize TTS
    tts = await test_tts_initialization()

    # Test 2: Check available voices
    await test_available_voices(tts)

    # Test 3: Fallback synthesis
    fallback_audio = await test_synthesis_fallback(tts)

    # Test 4: Real model synthesis (if available)
    model_audio = await test_synthesis_with_model(tts)

    # Test 5: Multiple sentences
    multi_audio = await test_multiple_sentences(tts)

    # Play audio samples
    if fallback_audio is not None:
        await play_audio(fallback_audio, tts.sample_rate, "Fallback Synthesis")
        await save_audio(fallback_audio, tts.sample_rate, "bm_george_fallback.wav")

    if model_audio is not None:
        await play_audio(model_audio, tts.sample_rate, "Real Model Synthesis")
        await save_audio(model_audio, tts.sample_rate, "bm_george_model.wav")

    if multi_audio is not None:
        await play_audio(multi_audio, tts.sample_rate, "Multiple Sentences")
        await save_audio(multi_audio, tts.sample_rate, "bm_george_multiple.wav")

    # Final summary
    print_header("VALIDATION SUMMARY")

    if fallback_audio is not None:
        print_success("Fallback synthesis working")
    else:
        print_error("Fallback synthesis failed")

    info = tts.get_model_info()
    if info['kokoro_loaded']:
        if model_audio is not None:
            print_success("Real Kokoro model synthesis working")
        else:
            print_error("Real Kokoro model synthesis failed")
    else:
        print_info("Kokoro model not loaded (using fallback)")

    print("\n" + "🎉 " * 20)
    print("Validation complete!")
    print("🎉 " * 20 + "\n")


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level="INFO",
        format="<level>{level: <8}</level> | {message}"
    )

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Validation interrupted by user")
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()
