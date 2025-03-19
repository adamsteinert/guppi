import os, time
import pytest
from loguru import logger

from src.glados.Extensions.Commands.StoreMemoryCommand import StoreMemoryCommand
from src.glados.Extensions.Commands.commandtype import CommandType
from src.glados.Extensions.Commands.command_manager import command_manager
from src.glados.Extensions.Commands.NullCommand import NullCommand


#    def test_CommandManager_ExecuteStoreMemory(self):
#        dir = "/Users/adams/Documents/Yahara-Sync/_CoreKB/Memento"
#        cm = command_manager()
#        cm.add_command(StoreMemoryCommand(dir))
#        result = cm.process_commands("store this text")
#        self.assertTrue(CommandType.EXPLICIT_RESPONSE, result.commandType)
#        self.assertTrue("StoreMemoryCommand", result.commandName)

#        result = cm.process_commands("do nothing")
#        self.assertTrue(CommandType.PASS_TO_LLM, result.commandType)
#        self.assertTrue("StoreMemoryCommand", result.commandName)

@pytest.mark.parametrize("text, expected", [
    ("store this text", True),
    ("record this text", True),
    ("do nothing", False),
    ("", False)
])
def test_StoreMemory_handle(text, expected):
    store_memory_command = StoreMemoryCommand("/Users/adams/Documents/Yahara-Sync/_CoreKB/Memento")
    assert (store_memory_command.handle_command(text) == expected)

def test_StoreMemory_get_response():
    store_memory_command = StoreMemoryCommand("/Users/adams/Documents/Yahara-Sync/_CoreKB/Memento")
    response = store_memory_command.get_response()
    assert response in store_memory_command.response


def test_StoreMemory_execute():
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


def test_NullCommand_execute():
    null_command = NullCommand()
    result = null_command.execute_command("do nothing")

    assert null_command.handle_command("do nothing")
    assert CommandType.PASS_TO_LLM == result.commandType
    assert "NullCommand" == result.commandName
    assert "do nothing" == result.text



