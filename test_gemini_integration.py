#!/usr/bin/env python3
"""Test script for Gemini API integration in GLaDOS 2.0."""

import asyncio
import sys
from pathlib import Path

# Add src2 to path
sys.path.insert(0, str(Path(__file__).parent / 'src2'))

from glados2.main import GladosApp


async def test_gemini_api():
    """Test Gemini API integration end-to-end."""
    print("🧪 Testing GLaDOS 2.0 Gemini API Integration")
    print("=" * 60)

    # Create app with Gemini config
    print("\n📝 Loading Gemini configuration...")
    app = GladosApp(config_path='src2/configs/glados2_gem_config.yaml')

    # Verify configuration
    provider_info = app._llm_manager.get_provider_info()
    print(f"\n✓ Provider: {provider_info['provider']}")
    print(f"✓ Model: {provider_info['model']}")
    print(f"✓ API Key configured: {provider_info['has_api_key']}")
    print(f"✓ Endpoint: {provider_info['completion_url']}")

    # Test simple API call
    print("\n🚀 Sending test message to Gemini...")
    print("   Message: 'Hello, GLaDOS! Tell me a short joke about testing.'")

    try:
        response = await app._llm_manager.send_message(
            "Hello, GLaDOS! Tell me a short joke about testing.",
            streaming=True
        )

        if response:
            print(f"\n✅ SUCCESS! Received response from Gemini:")
            print(f"\n{'-' * 60}")
            print(response)
            print(f"{'-' * 60}\n")

            # Verify it's not a mock response
            if "fascinating. And by fascinating" in response.lower():
                print("⚠️  WARNING: This looks like a mock response!")
            else:
                print("✓ Response appears to be from real Gemini API")

        else:
            print("\n❌ FAILED: No response received")

    except Exception as e:
        print(f"\n❌ FAILED with error: {e}")
        import traceback
        traceback.print_exc()

    print("\n🏁 Test completed!")


def main():
    """Main entry point."""
    try:
        asyncio.run(test_gemini_api())
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
