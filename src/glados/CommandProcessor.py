# src/glados/command_processor.py
import asyncio
from functools import partial
from queue import Empty
from loguru import logger
from .Extensions.llm.LanguageAgent import GeminiAgent, MCPClient, OllamaToolManager
from mcp import StdioServerParameters
from .llm_context import LlmContext

class CommandProcessor:
    def __init__(self, glados: "Glados"):
        self.glados = glados
        self.tts_queue = glados.tts_queue
        self.llm_queue = glados.llm_queue
        self.shutdown_event = glados.shutdown_event

    async def process_item(self, agent: GeminiAgent, queryContext: LlmContext) -> str:
        task = asyncio.create_task(agent.get_response(queryContext.text))
        while not task.done():
            await asyncio.sleep(0.25)
        return task.result()

    def completion_callback(self, task: asyncio.Task, original_item: LlmContext):
        try:
            result = task.result()
            self.process_llm_response_stream(result)
        except Exception as e:
            logger.error(f"Callback: Task failed for '{original_item}': {e}")

    def process_llm_response_stream(self, detected_text):
        if detected_text:
            self._process_sentence(detected_text)

    def _process_sentence(self, current_sentence: list[str]) -> None:
        sentence = "".join(current_sentence)
        logger.debug(f"@@@ Queueing sentence: {sentence}")
        self.glados.currently_speaking.set()
        self.tts_queue.put(sentence)
        self.tts_queue.put("<EOS>")

    async def process_llm(self) -> None:
        agent = GeminiAgent(OllamaToolManager())
        server_params = StdioServerParameters(
            command="uv",
            args=["--directory", "/Users/adams/source/production/mcp-guppi", "run", "mcp-guppi"],
            env={"PROJECTS_FILE": "/Users/adams/source/production/mcp-guppi/data/projects.md"}
        )
        async with MCPClient(server_params) as mcpclient:
            _, tools_list = await mcpclient.get_available_tools()
            for tool in tools_list:
                agent.tool_manager.register_tool(
                    name=tool.name,
                    function=mcpclient.call_tool,
                    description=tool.description,
                    inputSchema=tool.inputSchema
                )
            while not self.shutdown_event.is_set():
                try:
                    if not self.llm_queue.empty():
                        queryContext = self.llm_queue.get_nowait()
                        task = asyncio.create_task(self.process_item(agent, queryContext))
                        callback = partial(self.completion_callback, original_item=queryContext)
                        task.add_done_callback(callback)
                        await asyncio.sleep(0.25)
                    else:
                        await asyncio.sleep(0.25)
                except Empty:
                    await asyncio.sleep(0.25)
