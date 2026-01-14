#!/bin/bash
# Run GLaDOS 2.0 tests

set -e

echo "🧪 Running GLaDOS 2.0 Tests"
echo "=============================="

# Change to script directory
cd "$(dirname "$0")"

# Run unit tests (fast)
echo ""
echo "📋 Running unit tests..."
pytest tests/ -m "not integration and not slow" "$@"

# Run integration tests if requested
if [[ "$1" == "--integration" ]] || [[ "$1" == "--all" ]]; then
    echo ""
    echo "🔗 Running integration tests..."
    pytest tests/ -m "integration" "$@"
fi

# Run all tests if requested
if [[ "$1" == "--all" ]]; then
    echo ""
    echo "🎯 Running ALL tests..."
    pytest tests/ "$@"
fi

echo ""
echo "✅ Tests completed!"
