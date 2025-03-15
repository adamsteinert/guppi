import os
from datetime import datetime
from loguru import logger

from .commandresponse import CommandResponse
from .commandtype import CommandType


class StoreMemoryCommand:
    START_WINDOW_INDEX = 20

    def __init__(self, directory: str):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def handle_command(self, text: str, context: str = ""):
        # check if the text contains the word "store" near the beginning
        index = text.find("store")
        return index != -1 and index < self.START_WINDOW_INDEX

    def execute_command(self, text: str, context: str = ""):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(self.directory, f"memento_{timestamp}.txt")

        with open(filename, 'w') as file:
            file.write(f"{timestamp}\n\n{text}")

        logger.success(f"StoreMemoryCommand: execute_command - Command stored in {filename}")
        return CommandResponse(
            CommandType.EXPLICIT_RESPONSE,
            "StoreMemoryCommand",
            "I have stored that information for you.",
            "",
            filename)
