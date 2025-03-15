from glados.Extensions.Commands.NullCommand import NullCommand
from glados.Extensions.Commands.StoreMemoryCommand import StoreMemoryCommand


class command_manager:

    def __init__(self):
        self.commands = []

    def add_command(self, command):
        self.commands.append(command)

    def load_commands(self):
        # preload all active commands
        dir = "/Users/adams/Documents/Yahara-Sync/_CoreKB/Memento"
        self.commands.append(StoreMemoryCommand(dir))

    def process_commands(self, text: str, context: str = ""):
        for x in self.commands:
            if x.handle_command(text, context):
                return x.execute_command(text, context)

        return NullCommand().execute_command(text, context)
