from loguru import logger

from .commandresponse import CommandResponse
from .commandtype import CommandType


class NullCommand:
    def handle_command(self, text, context=""):
        return True

    def execute_command(self, text, context=""):
        return CommandResponse(CommandType.PASS_TO_LLM,
                               "NullCommand",
                               text,
                               "",
                               context)


