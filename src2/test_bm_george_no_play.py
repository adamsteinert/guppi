#!/usr/bin/env python3
"""Quick test of bm_george voice without audio playback."""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from glados2.audio.tts_processor import TTSProcessor


async def main():
    """Test bm_george synthesis."""
    print("Testing bm_george voice...")

    tts = TTSProcessor(voice="bm_george", model_dir="models/TTS", sample_rate=22050)

    # Get model info
    info = tts.get_model_info()
    print(f"Kokoro loaded: {info['kokoro_loaded']}")
    print(f"Voice embeddings loaded: {len(tts.voice_embeddings)}")

    # Test synthesis
    text = "Hello, I am George. This is a test."
    print(f"\nSynthesizing: '{text}'")

    audio = await tts.synthesize_speech(text)

    if audio is not None:
        print(f"✓ Success! Generated {len(audio)} samples ({len(audio)/22050:.2f}s)")
        print(f"✓ bm_george voice is working!")
        return 0
    else:
        print("✗ Failed to synthesize")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
