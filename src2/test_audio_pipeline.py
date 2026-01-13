#!/usr/bin/env python3
"""Test script for the GLaDOS 2.0 audio pipeline."""

import asyncio
import sys
from pathlib import Path

# Add src2 to path so we can import glados2
sys.path.insert(0, str(Path(__file__).parent))

from glados2.main import GladosApp


async def test_audio_playback():
    """Test playback."""
   
    # Create app instance
    app = GladosApp()
    
    print("✅ App initialized")
    print(f"📊 Audio Manager: {app._audio_manager is not None}")
    print(f"🧠 LLM Manager: {app._llm_manager is not None}")
    print(f"🎯 State Manager: {app._state_manager.get_state().value}")
    
    # Test TTS directly first
    print("\n🔊 Testing TTS synthesis...")
    try:
        success = await app._audio_manager.synthesize_and_play(
            "Hello, this is a test of the GLaDOS text-to-speech system."
        )
        print(f"✅ TTS test: {'Success' if success else 'Failed'}")
    except Exception as e:
        print(f"❌ TTS test failed: {e}")


async def test_audio_pipeline():
    """Test the complete audio pipeline: listening -> ASR -> LLM -> TTS -> playback."""
    print("🎤 Testing GLaDOS 2.0 Audio Pipeline")
    print("=" * 50)
    
    # Create app instance
    app = GladosApp()
    
    print("✅ App initialized")
    print(f"📊 Audio Manager: {app._audio_manager is not None}")
    print(f"🧠 LLM Manager: {app._llm_manager is not None}")
    print(f"🎯 State Manager: {app._state_manager.get_state().value}")
    
    # Test TTS directly first
    print("\n🔊 Testing TTS synthesis...")
    try:
        success = await app._audio_manager.synthesize_and_play(
            "Hello, this is a test of the GLaDOS text-to-speech system."
        )
        print(f"✅ TTS test: {'Success' if success else 'Failed'}")
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
    
    # Test mock LLM response
    print("\n🧠 Testing mock LLM response...")
    try:
        response = await app._llm_manager.send_message("Hello GLaDOS", streaming=False)
        print(f"✅ LLM response: {response}")
    except Exception as e:
        print(f"❌ LLM test failed: {e}")
    
    # Test full pipeline with listening (requires microphone)
    print("\n🎤 Testing audio listening pipeline...")
    print("Speak into your microphone within 10 seconds...")
    
    try:
        # Start listening
        await app._audio_manager.start_listening()
        print("🎧 Listening started...")
        
        # Listen for 10 seconds
        await asyncio.sleep(10)
        
        # Stop listening
        await app._audio_manager.stop_all_audio()
        print("🛑 Listening stopped")
        
    except Exception as e:
        print(f"❌ Listening test failed: {e}")
    
    print("\n🏁 Test completed!")
    print("Check the logs above for any errors or successful operations.")


def main():
    """Main entry point."""
    try:
        asyncio.run(test_audio_playback())
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()