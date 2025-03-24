from venv import logger
import ollama
import pydantic
from pydantic import BaseModel
from enum import Enum, IntEnum
from .toolcall_response import ToolCallResponse
from .tools import *

# Tool calling/python/ollama: https://www.cohorte.co/blog/using-ollama-with-python-step-by-step-guide
# https://toolworks.dev/docs/Guides/Advanced/call-functions-ollama-python.md
# Change a model file: https://www.restack.io/p/ollama-answer-change-model-file-cat-ai
# Dolphin3 toolcalling bug: https://github.com/ollama/ollama/issues/8329

available_functions = {
    'calculate_area': calculate_area,
    'get_salinity': get_salinity,
    'everything_else': get_everything_else,
    'transcribe_notes_from_obs_meeting': transcribe_notes_from_obs_meeting
}

tool_system_prompt = """You are an AI assistant used for tool calling. Choose the best tool from those provided.
queries involving salt or salinity should always use the tool get_salinity.
queries involving area should always use the tool calculate_area.
If one does not exist, please reply there was no tool call found matching the provided parameters.
"""

tool_sentiment_prompt = """
You are an AI assistant analyzing user requests to determine if they warrant using one of your available tools. 
Respond with "true" if the request clearly indicates a need for the salt calculator or area calculator. return "false"
otherwise, even if a tool is appropriate but not explicitly mentioned.


Listed tools:
1. Salt Calculator - For calculating the amount of salt to add to a body of water to reach a desired salinity level
2. Area Calculations - For calculating the area of a rectangle (and only a rectangle)
3. OBS Transcription - For transcribing an audio recording in OBS given the name of the file to be created and optionally
the original filename of the recording.

Examples:
- "What is the area of a rectangle with length 5 and width 3?" → true (requires area calculations tool)
- "What is the area of a circle with diameter 3?" → false (No available tool)
- "What do I need for salinity with a current value of thirteen?" → true (requires salt calculator tool)
- "What's the weather in Paris today?" → false (that tool is not availalbe)
- "Help me understand quantum computing." → false (can be answered through conversation)
- "Calculate the compound interest on $10,000 at 5% for 10 years." → false (that tool is not available)
- "What are your thoughts on AI ethics?" → false (can be answered through conversation)
- "Transcribe the last OBS recording, call it meeting notes" -> true (requires OBS Transcription tool, no source fine identified, output file should be called 'meeting notes')
- "Transcribe the OBS recording with the name meeting with jason banks -> true (requires OBS transcription, output file name is 'jason banks'
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


def process_tool_call_response(text: str, modelName: str = "llama3.1", context: str = ""):
    ###Answer a user request with a tool call###
    fnlist = list(available_functions.values())
    response = ollama.chat(
        modelName,
        messages=[{'role': 'system', 'content': tool_system_prompt},
                  {'role': 'user', 'content': text}],
        # get an array of the values form available_functions

        tools=fnlist
    )

    return handle_tool_calls(response, available_functions)


def analyze_request_for_tools(text: str, modelName: str = "llama3.1") -> bool:
    """Determine if a tool call is appropriate based on the user request and tools present in the application"""
    response = ollama.chat(
        modelName,
        messages=[
            {'role': 'assistant', 'content': tool_sentiment_prompt},
            {'role': 'user', 'content': text}],
    )
    return response.message.content.lower().startswith('true')









ts2_prompt = """
You are an AI assistant analyzing user requests to categorize them into types. The categories are 
internal behavior, general query, and tool call.
Respond with "tool_call" if the request clearly indicates a need for the salt calculator or area calculator. 
Respond with "internal_behavior" if the request clearly asks to shut down or clear memory"
For anything else, respond "general_query"

tool calls:
1. Salt Calculator - For calculating the amount of salt to add to a body of water to reach a desired salinity level
2. Area Calculations - For calculating the area of a rectangle (and only a rectangle)

internal behaviors:
1. Shut Down - For shutting down the system
2. Clear Memory - For clearing the system's memory

Examples:
- "What is the area of a rectangle with length 5 and width 3?" → tool_call (requires area calculations tool)
- "What is the area of a circle with diameter 3?" → general_query (No available tool)
- "What do I need for salinity with a current value of thirteen?" → tool_call (requires salt calculator tool)
- "please shut down" → internal_behavior
- "What's the weather in Paris today?" → general_query (that tool is not availalbe)
- "Help me understand quantum computing." → general_query (can be answered through conversation)
- "Calculate the compound interest on $10,000 at 5% for 10 years." → general_query (that tool is not available)
- "What are your thoughts on AI ethics?" → general_query (can be answered through conversation)
"""



class QueryTypeEnum(str, Enum):
    internal_behavior = 'internal_behavior'
    general_query = 'general_query'
    system_tool = 'tool_call'

class QuerySentimentResponse(BaseModel):
    query_sentiment: QueryTypeEnum


def categorize_request(text: str, modelName: str = "llama3.1") -> QuerySentimentResponse:
    """Determine if a tool call is appropriate based on the user request and tools present in the application"""
    response = ollama.chat(
        modelName,
        messages=[
            {'role': 'assistant', 'content': ts2_prompt},
            {'role': 'user', 'content': text}],
        format=QuerySentimentResponse.model_json_schema()
    )

    return QuerySentimentResponse.model_validate_json(response.message.content)
