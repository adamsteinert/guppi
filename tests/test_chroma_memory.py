import pytest
from loguru import logger

from src.glados.Extensions.vectors.Memory import Memory


class TestMemory:
    def setup_method(self):
        self.memory = Memory()

    def test_simple(self):
        m = Memory()
        m.test_simple()

    def test_query_memory(self):
        self.memory.store_command_memories("/foo")
        query = "What do I need for salinity with a current value of thirteen"
        result = self.memory.query_memory(query)
        assert result is not None
        assert isinstance(result, list)
        assert len(result) > 0



