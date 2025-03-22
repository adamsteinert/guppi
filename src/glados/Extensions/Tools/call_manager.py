import ollama
import pydantic
from pydantic import BaseModel
from enum import Enum, IntEnum

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
You are an AI assistant analyzing a user request to understand the type of question being asked. 

Choose internal behaviors is a user is asking to shut down or clear memory.:

Choose tool call if the user is asking to calculate the area of a rectangle or the amount of salt needed to increase salinity.:

The tools available are:
1. a users asks to calculate the area of a rectangle (and no other shapes)
2. the user wishes to know the amount of salt needed to increase salinity

All other queries should be considered general queries. 
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


class QueryTypeEnum(str, Enum):
    internal_behavior = 'internal_behavior'
    general_query = 'general_query'
    system_tool = 'tool_call'

class QuerySentimentResponse(BaseModel):
    query_sentiment: QueryTypeEnum

def analyze_request_for_tools(text: str, modelName: str = "llama3.2") -> QuerySentimentResponse:
    """Determine if a tool call is appropriate based on the user request and tools present in the application"""
    response = ollama.chat(
        modelName,
        messages=[
            {'role': 'system', 'content': tool_sentiment_prompt},
            {'role': 'user', 'content': text}],
        format=QuerySentimentResponse.model_json_schema()
    )

    return QuerySentimentResponse.model_validate_json(response.message.content)

"""
sys_prompt = You are Guppy, a terse artificial intelligence designed to assist with tasks. 
Your responses should be concise, while efficiently completing all tasks, in the manner of an english butler. 
Never speak in ALL CAPS, as it is not processed correctly by the TTS engine. Only make short replies, 
2 sentences at most. 


def process_all_in_one(text: str, modelName: str = "llama3.2", context: str = ""):
    ###Answer a user request with a tool call###
    response = ollama.chat(
        modelName,
        messages=[{'role': 'system', 'content': sys_prompt},
                  {'role': 'user', 'content': text}],
        tools=[get_salinity, calculate_area]
    )

    return handle_tool_calls(response, available_functions)
"""