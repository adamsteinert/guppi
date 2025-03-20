import os, time
import pytest
from loguru import logger

from src.glados.Extensions.Commands.store_memory_command import StoreMemoryCommand
from src.glados.Extensions.Commands.command_type import CommandType
from src.glados.Extensions.Commands.null_command import NullCommand


@pytest.mark.parametrize("text, expected", [
    ("store this text", True),
    ("record this text", True),
    ("do nothing", False),
    ("", False)
])
def test_store_memory_handle(text, expected):
    store_memory_command = StoreMemoryCommand("/Users/adams/Documents/Yahara-Sync/_CoreKB/Memento")
    assert (store_memory_command.handle_command(text) == expected)


def test_store_memory_get_response():
    store_memory_command = StoreMemoryCommand("/Users/adams/Documents/Yahara-Sync/_CoreKB/Memento")
    response = store_memory_command.get_response()
    assert response in store_memory_command.response


def test_store_memory_execute():
    dir = "/Users/adams/Documents/Yahara-Sync/_CoreKB/Memento"
    store_memory_command = StoreMemoryCommand(dir)
    result = store_memory_command.execute_command("TESTXZY store this text")

    # Check that a file exists in result.system
    logger.success(f"result.system: {result.system}")

    # Check that the file at result.system was written
    assert os.path.exists(result.context)

    assert CommandType.EXPLICIT_RESPONSE == result.commandType

    # Clean up. delete the file at path result.system if it was written
    if os.path.exists(result.system):
        os.remove(result.system)


def test_null_command_execute():
    null_command = NullCommand()
    result = null_command.execute_command("do nothing")

    assert null_command.handle_command("do nothing")
    assert CommandType.PASS_TO_LLM == result.commandType
    assert "NullCommand" == result.commandName
    assert "do nothing" == result.text



