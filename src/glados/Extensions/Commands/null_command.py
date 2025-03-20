from loguru import logger

from .command_response import CommandResponse
from .command_type import CommandType


class NullCommand:
    def handle_command(self, text, context=""):
        return True

    def execute_command(self, text, context=""):
        return CommandResponse(CommandType.PASS_TO_LLM,
                               "NullCommand",
                               text,
                               "",
                               context)


