import asyncio
import os
import json
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageToolCall
from .OllamaToolManager import OllamaToolManager

#from google import genai
#from google.genai import types

DEFAULT_PROMPT = """
You are a helpful assistant ready to answer a wide variety of user questions. You are able to use the tools available 
to you in order to enhance your knowledge and capabilities. If the available tools are not sufficient to answer a question,
you may try to answer the question to the best of your ability.
"""

class GeminiAgent:
    def __init__(self,
                 tool_manager: OllamaToolManager,
                 default_prompt=DEFAULT_PROMPT) -> None:
        self.default_prompt = default_prompt
        self.client = OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.model = os.getenv("GEMINI_MODEL")
        self.messages = []
        self.tool_manager = tool_manager
        self.messages.append({
            'role': 'system',
            'content': default_prompt
        })

    async def get_response(self, content:str):
        self.messages.append({
            'role': 'user',
            'content': content
        })

        tools = self.tool_manager.get_tools_openai()
        query = self.client.chat.completions.create(
            model=self.model,
            messages=self.messages,
            tools=tools,
            tool_choice="auto"
        )

        result = await self.handle_response(query.choices[0])
        if result is None:
            result = "Tool results not available."
        else:
            self.messages.append({
                "role": "assistant",
                "content": result
            })

        return result

    async def handle_response(self, response):
        try:

            tool_calls = response.message.tool_calls

            if tool_calls:
                tool_calls[0].function.arguments = json.loads(tool_calls[0].function.arguments)
                tool_payload = {
                    "function": tool_calls[0].function
                }

                result = await self.tool_manager.execute_tool(tool_payload)

                tool_response = []
                for content in result.content:
                    tool_response.append(content.text)

                return "".join(tool_response)
            else:
                return response.message.content

        except Exception as e:
            print(e)