#!/usr/bin/env python3
"""
Quick TTS test - minimal script to validate bm_george voice works.

This is a simplified version of the full validation script for quick testing.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from glados2.audio.tts_processor import TTSProcessor

try:
    import sounddevice as sd
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False
    print("⚠️  sounddevice not available - will only test synthesis, not playback")


async def main():
    """Quick TTS test."""
    print("\n" + "="*60)
    print("  Quick TTS Test - bm_george Voice")
    print("="*60 + "\n")

    # Initialize TTS
    print("1. Initializing TTS processor...")
    tts = TTSProcessor(
        voice="bm_george",
        model_dir="models/TTS",
        sample_rate=22050
    )
    print(f"   ✓ Voice: {tts.voice}")
    print(f"   ✓ Sample rate: {tts.sample_rate}")

    # Check model status
    info = tts.get_model_info()
    print(f"\n2. Model status:")
    print(f"   - Kokoro loaded: {info['kokoro_loaded']}")
    if not info['kokoro_loaded']:
        print("   ⚠️  Using fallback synthesis (model not found)")
    print(f"   - Available voices: {len(info['available_voices'])}")

    # Synthesize test phrase
    test_text = "Hello, I am George. This is a test of the text-to-speech system."
    print(f"\n3. Synthesizing: '{test_text}'")

    audio = await tts.synthesize_speech(test_text)

    if audio is None:
        print("   ✗ Synthesis FAILED")
        return 1

    print(f"   ✓ Synthesis succeeded")
    print(f"   - Samples: {len(audio):,}")
    print(f"   - Duration: {len(audio) / tts.sample_rate:.2f}s")
    print(f"   - Range: [{audio.min():.3f}, {audio.max():.3f}]")

    # Save to file
    try:
        import soundfile as sf
        output_dir = Path("output")
        output_dir.mkdir(exist_ok=True)
        output_file = output_dir / "quick_test_bm_george.wav"
        sf.write(str(output_file), audio, tts.sample_rate)
        print(f"\n4. Saved audio to: {output_file}")
    except ImportError:
        print("\n4. ⚠️  soundfile not available - cannot save audio")
    except Exception as e:
        print(f"\n4. ⚠️  Could not save audio: {e}")

    # Play audio
    if HAS_SOUND:
        print("\n5. Playing audio...")
        try:
            sd.play(audio, samplerate=tts.sample_rate)
            sd.wait()
            print("   ✓ Playback completed")
        except Exception as e:
            print(f"   ✗ Playback failed: {e}")
    else:
        print("\n5. Skipping playback (sounddevice not available)")

    print("\n" + "="*60)
    print("  ✅ Test Complete!")
    print("="*60 + "\n")

    return 0


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
