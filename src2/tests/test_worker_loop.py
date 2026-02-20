"""Tests for the WorkerLoop persistent background event loop."""

import asyncio

import pytest

from glados2.core.worker_loop import WorkerLoop


class TestWorkerLoop:
    """Core lifecycle and execution tests for WorkerLoop."""

    def test_start_and_stop(self):
        worker = WorkerLoop(name="test")
        worker.start()
        assert worker.is_running
        worker.stop()
        assert not worker.is_running

    def test_double_start_is_idempotent(self):
        worker = WorkerLoop(name="test")
        worker.start()
        worker.start()  # should not raise
        assert worker.is_running
        worker.stop()

    def test_stop_before_start_is_safe(self):
        worker = WorkerLoop(name="test")
        worker.stop()  # should not raise

    def test_run_coroutine(self):
        worker = WorkerLoop(name="test")
        worker.start()
        try:
            async def add(a, b):
                return a + b

            future = worker.run(add(2, 3))
            assert future.result(timeout=5) == 5
        finally:
            worker.stop()

    def test_same_loop_across_calls(self):
        """All coroutines run on the same event loop — the key property."""
        worker = WorkerLoop(name="test")
        worker.start()
        try:
            loops = []

            async def capture_loop():
                loops.append(asyncio.get_running_loop())

            worker.run(capture_loop()).result(timeout=5)
            worker.run(capture_loop()).result(timeout=5)
            worker.run(capture_loop()).result(timeout=5)

            assert len(loops) == 3
            assert loops[0] is loops[1] is loops[2]
            assert loops[0] is worker.loop
        finally:
            worker.stop()

    @pytest.mark.asyncio
    async def test_run_await_from_async_context(self):
        worker = WorkerLoop(name="test")
        worker.start()
        try:
            async def greet(name):
                return f"Hello, {name}"

            result = await worker.run_await(greet("GLaDOS"))
            assert result == "Hello, GLaDOS"
        finally:
            worker.stop()

    def test_run_before_start_raises(self):
        worker = WorkerLoop(name="test")
        with pytest.raises(RuntimeError, match="not started"):
            worker.run(asyncio.sleep(0))

    def test_exception_propagates(self):
        """Exceptions in coroutines propagate through the future."""
        worker = WorkerLoop(name="test")
        worker.start()
        try:
            async def boom():
                raise ValueError("kaboom")

            future = worker.run(boom())
            with pytest.raises(ValueError, match="kaboom"):
                future.result(timeout=5)
        finally:
            worker.stop()

    def test_concurrent_coroutines(self):
        """Multiple coroutines can run concurrently on the worker loop."""
        worker = WorkerLoop(name="test")
        worker.start()
        try:
            async def delayed_value(val, delay):
                await asyncio.sleep(delay)
                return val

            f1 = worker.run(delayed_value("a", 0.1))
            f2 = worker.run(delayed_value("b", 0.1))
            f3 = worker.run(delayed_value("c", 0.1))

            assert f1.result(timeout=5) == "a"
            assert f2.result(timeout=5) == "b"
            assert f3.result(timeout=5) == "c"
        finally:
            worker.stop()
