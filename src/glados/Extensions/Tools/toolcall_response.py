
class ToolCallResponse:
    """A class that encapsulates the response of a tool call,
       including function name, arguments, and results or an error if one occurred"""

    def __init__(self, function_name: str, arguments: dict, result: object, error: Exception):
        self.function_name = function_name
        self.arguments = arguments
        self.result = result
        self.error = error

    def __str__(self):
        return f"Function {self.function_name} output: {self.result}" if self.result else f"Error executing {self.function_name}: {self.error}"