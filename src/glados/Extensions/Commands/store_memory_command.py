import os
import re
import random
from datetime import datetime
from loguru import logger
from .command_response import CommandResponse
from .command_type import CommandType


class StoreMemoryCommand:
    START_WINDOW_INDEX = 20
    response = ["I have stored that information for you.",
                "Memory recorded, sir.",
                "Memory stored.",
                "Memory saved.",
                "Got it, sir.",
                "Understood."
                "Recorded."]

    def __init__(self, directory: str):
        self.directory = directory
        if not os.path.exists(directory):
            os.makedirs(directory)

    def handle_command(self, text: str, context: str = ""):
        # check if the text contains the word "store" near the beginning
        match = re.search(r'\b(store|record)\b', text)
        index = -1

        if match != None:
            index = match.start()

        return index != -1 and index < self.START_WINDOW_INDEX

    def get_response(self):
        return self.response[random.randint(0, len(self.response))]

    def execute_command(self, text: str, context: str = ""):
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(self.directory, f"memento_{timestamp}.md")

        with open(filename, 'w') as file:
            file.write(f"{timestamp}\n\n{text}")

        logger.success(f"StoreMemoryCommand: execute_command - Command stored in {filename}")
        return CommandResponse(
            CommandType.EXPLICIT_RESPONSE,
            "StoreMemoryCommand",
            # return a random value from the repsonse list
            self.get_response(),
            "",
            filename)
