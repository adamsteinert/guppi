import os, time
import unittest
from loguru import logger

from src.glados.Extensions.Commands.StoreMemoryCommand import StoreMemoryCommand
from src.glados.Extensions.Commands.commandtype import CommandType
from src.glados.Extensions.command_manager import command_manager
from src.glados.Extensions.Commands.NullCommand import NullCommand


class TestCommandManager(unittest.TestCase):

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


    def test_StoreMemory_handle(self):
        store_memory_command = StoreMemoryCommand("/Users/adams/Documents/Yahara-Sync/_CoreKB/Memento")
        self.assertTrue(store_memory_command.handle_command("store this text"))


    def test_StoreMemory_execute(self):
        dir = "/Users/adams/Documents/Yahara-Sync/_CoreKB/Memento"
        store_memory_command = StoreMemoryCommand(dir)
        result = store_memory_command.execute_command("TESTXZY store this text")

        # Check that a file exists in result.system
        logger.success(f"result.system: {result.system}")

        # Check that the file at result.system was written
        self.assertTrue(os.path.exists(result.context))

        self.assertTrue(CommandType.EXPLICIT_RESPONSE, result.commandType)

        # Clean up. delete the file at path result.system if it was written
        if os.path.exists(result.system):
            os.remove(result.system)


    def test_NullCommand_execute(self):
        null_command = NullCommand()
        result = null_command.execute_command("do nothing")

        self.assertTrue(null_command.handle_command("do nothing"))
        self.assertTrue(CommandType.PASS_TO_LLM, result.commandType)
        self.assertTrue("NullCommand", result.commandName)
        self.assertTrue("do nothing", result.text)


if __name__ == '__main__':
    unittest.main()



