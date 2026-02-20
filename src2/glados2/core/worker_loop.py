"""Persistent background event loop for async operations.

Provides a stable event loop for operations that create loop-bound
resources (like MCP subprocess connections) and need those resources
to remain usable across multiple async operations.
"""

import asyncio
import threading
from concurrent.futures import Future
from typing import Any, Coroutine

from loguru import logger


class WorkerLoop:
    """
    A persistent background event loop running in a dedicated thread.

    All coroutines scheduled via ``run()`` execute on the same event loop,
    ensuring that loop-bound resources (MCP stdio sessions, HTTP streams)
    created in one call remain valid for subsequent calls.

    Usage::

        worker = WorkerLoop()
        worker.start()

        future = worker.run(some_coroutine())
        result = future.result(timeout=30)

        # Or from another async context:
        result = await worker.run_await(some_coroutine())

        worker.stop()
    """

    def __init__(self, name: str = "worker"):
        self._name = name
        self._loop: asyncio.AbstractEventLoop = asyncio.new_event_loop()
        self._thread: threading.Thread | None = None
        self._started = False

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        """The underlying event loop."""
        return self._loop

    @property
    def is_running(self) -> bool:
        return self._started and self._loop.is_running()

    def start(self) -> None:
        """Start the background event loop thread."""
        if self._started:
            return
        # Create the ready event before starting the thread so it
        # exists by the time call_soon_threadsafe fires.
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name=f"{self._name}-loop",
            daemon=True,
        )
        self._thread.start()
        # Wait until the loop is actually running before returning,
        # so that callers can immediately schedule work.
        self._loop.call_soon_threadsafe(self._ready.set)
        self._ready.wait(timeout=5.0)
        self._started = True
        logger.info(f"Worker loop '{self._name}' started")

    def _run(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def stop(self) -> None:
        """Stop the background event loop and wait for thread to exit."""
        if not self._started:
            return
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread:
            self._thread.join(timeout=5.0)
        self._started = False
        logger.info(f"Worker loop '{self._name}' stopped")

    def run(self, coro: Coroutine) -> Future:
        """Schedule a coroutine and return a ``concurrent.futures.Future``."""
        if not self._started:
            raise RuntimeError(f"Worker loop '{self._name}' not started")
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def run_await(self, coro: Coroutine) -> Any:
        """Schedule a coroutine on the worker loop and await from the caller's loop."""
        future = self.run(coro)
        return await asyncio.wrap_future(future)
