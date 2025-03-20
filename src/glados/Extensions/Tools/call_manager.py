import ollama

from src.glados.Extensions.Tools.toolcall_response import ToolCallResponse
from src.glados.Extensions.Tools.tools import *

# Tool calling/python/ollama: https://www.cohorte.co/blog/using-ollama-with-python-step-by-step-guide
# https://toolworks.dev/docs/Guides/Advanced/call-functions-ollama-python.md
# Change a model file: https://www.restack.io/p/ollama-answer-change-model-file-cat-ai
# Dolphin3 toolcalling bug: https://github.com/ollama/ollama/issues/8329

available_functions = {
    'calculate_area': calculate_area,
    'get_salinity': get_salinity
}

tool_system_prompt = """You are an AI assistant used for tool calling. Choose the best tool from those provided.
queries involving salt or salinity should always use the tool get_salinity.
queries involving area should always use the tool calculate_area.
If one does not exist, please reply there was no tool call found matching the provided parameters.
"""

tool_sentiment_prompt = """
You are an AI assistant analyzing a user request to understand if a tool call is available that can 
fulfill that request. Respond with only a true or false value based on if the request indicates a tool
call is appropriate. The available tools are listed below:
1. There is a tool for calculating the area of a rectangle (and no other shapes)
2. There is a tool for calculating the amount of salt to add to a solution to reach a salinity of 26 
"""


def handle_tool_calls(response, available_functions):
    """Process tool calls from model response"""
    for tool in response.message.tool_calls or []:
        function_name = tool.function.name
        function_to_call = available_functions.get(function_name)

        if function_to_call:
            try:
                result = function_to_call(**tool.function.arguments)
                return ToolCallResponse(function_name, tool.function.arguments, result, None)

            except Exception as e:
                return ToolCallResponse(function_name, tool.function.arguments, None, e)

        else:
            return ToolCallResponse("FNF", None, None, Exception(f"Function not found: {function_name}"))

    return ToolCallResponse("FNF", None, None, Exception("No tool calls found"))


def process_tool_call_response(text: str, modelName: str = "llama3.2", context: str = ""):
    ###Answer a user request with a tool call###
    response = ollama.chat(
        modelName,
        messages=[{'role': 'system', 'content': tool_system_prompt},
                  {'role': 'user', 'content': text}],
        tools=[get_salinity, calculate_area]
    )

    return handle_tool_calls(response, available_functions)


def analyze_request_for_tools(text: str, modelName: str = "llama3.2"):
    """Determine if a tool call is appropriate based on the user request and tools present in the application"""
    response = ollama.chat(
        modelName,
        messages=[
            {'role': 'system', 'content': tool_system_prompt},
            {'role': 'user', 'content': text}],
    )

    return response
