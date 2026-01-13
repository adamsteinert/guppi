#!/bin/bash
# Wrapper script to run GLaDOS 2.0 with proper Python path

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PYTHONPATH="$SCRIPT_DIR/src2:$PYTHONPATH"

# Default to Gemini config if no arguments provided
if [ $# -eq 0 ]; then
    uv run python -m glados2.main --config src2/configs/glados2_gem_config.yaml
else
    uv run python -m glados2.main "$@"
fi
