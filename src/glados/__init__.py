"""GLaDOS - Voice Assistant using ONNX models for speech synthesis and recognition."""
import asyncio
import engine
#from .engine import Glados, GladosConfig

def main():
    """Main entry point for the package."""
    engine.start()

__version__ = "0.1.0"
__all__ = ["Glados", "GladosConfig"]
