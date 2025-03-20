from loguru import logger

from .command_type import CommandType


class CommandStore:
    START_WINDOW_INDEX = 20

    def handle_command(self, text: str, context: str = ""):
        # check if the text contains the word "store" near the beginning
        index = text.find("store")
        return index != -1 and index < self.START_WINDOW_INDEX

    def execute_command(self, text: str, context: str = ""):
        logger.success("cmd_store: execute_command")
        #return command_type.EXPLICIT_RESPONSE, "Sir, I am afraid, I don't know how to store data yet."
        return CommandType.PASS_TO_LLM, "You have been asked to " + text + ". Respond by stating that you are unable to complete this task because you lack the proper training."
