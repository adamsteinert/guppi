"""Entry point for the TTS API server: python -m tts_server"""

import argparse

import uvicorn

from .app import create_app
from .config import load_config


def main():
    parser = argparse.ArgumentParser(description="GLaDOS TTS API Server (ElevenLabs-compatible)")
    parser.add_argument("--config", default=None, help="Path to YAML config file")
    parser.add_argument("--host", default=None, help="Host to bind to (overrides config)")
    parser.add_argument("--port", type=int, default=None, help="Port to bind to (overrides config)")
    args = parser.parse_args()

    config = load_config(args.config)
    host = args.host or config.host
    port = args.port or config.port

    app = create_app(config)
    uvicorn.run(app, host=host, port=port, log_level=config.log_level.lower())


if __name__ == "__main__":
    main()
