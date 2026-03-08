#!/bin/bash
# Start the ElevenLabs-compatible TTS API server
# Usage: ./run_tts_server.sh [--host HOST] [--port PORT] [--config CONFIG]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Add src2/ to PYTHONPATH for both tts_server and glados2 packages
export PYTHONPATH="$SCRIPT_DIR/src2:$SCRIPT_DIR:$PYTHONPATH"

cd "$SCRIPT_DIR"
uv run python -m tts_server --config configs/tts_server_config.yaml "$@"
