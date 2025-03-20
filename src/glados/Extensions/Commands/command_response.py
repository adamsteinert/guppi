

class CommandResponse:
    def __init__(self, commandType: str, commandName: str, text: str, system: str = "", context: str = ""):
        self.commandName = commandName
        self.commandType = commandType
        self.text = text
        self.system = system
        self.context = context