
from mcp import StdioServerParameters
from .MCPClient import MCPClient
from .OllamaToolManager import OllamaToolManager
from .GeminiAgent import GeminiAgent
from dotenv import load_dotenv

load_dotenv()  # load environment variables from .env

class LanguageAgent:
    def __init__(self):
        self.toolManager = []
        self.agent = GeminiAgent(OllamaToolManager())
        git_server_params = StdioServerParameters(
            command="uv",
            args=["--directory", "/Users/adams/source/production/mcp-guppi", "run", "mcp-guppi"],
            env={"PROJECTS_FILE": "/Users/adams/source/production/mcp-guppi/data/projects.md"}
        )
        self.mcpclient = MCPClient(git_server_params)
        self.agent = GeminiAgent(OllamaToolManager())

    def is_ready(self):
        """Check if the agent is ready to process requests."""
        return self.mcpclient.is_ready()

    async def configure(self):
        await self.mcpclient.initialize()
        await self.mcpclient.connect()
        _ , self.tools_list = await self.mcpclient.get_available_tools()
        for tool in self.tools_list:
            self.agent.tool_manager.register_tool(
                name=tool.name,
                function=self.mcpclient.call_tool, # Passing the function reference here
                description=tool.description,
                inputSchema=tool.inputSchema
            )
        print(f"Agent {self.agent.model} configured and registered with the following tools {self.tools_list}")


    async def cleanup(self):
        """Clean up resources."""
        await self.mcpclient.cleanup()


    async def get_response(self, user_prompt: str):
        return await self.agent.get_response(user_prompt)



# print("Fetching available tools from the MCP server")
#     async with MCPClient(git_server_params) as mcpclient:
#         _ ,tools_list = await mcpclient.get_available_tools()
#         console.clear()
#         console.print(Panel.fit("🚀 Welcome to Ollama MCP Client 🚀", padding=(1, 4)))
#         console.status("Registering Tools", spinner="dots")
#         for tool in tools_list:
#             agent.tool_manager.register_tool(
#                 name=tool.name,
#                 function=mcpclient.call_tool, # Passing the function reference here
#                 description=tool.description,
#                 inputSchema=tool.inputSchema
#             )
#
#         while True:
#             try:
#                 print("-" * 40)
#                 user_prompt = input("How can I help you?\n")
#                 print("-" * 40)
#                 if user_prompt.lower() in ['quit', 'exit', 'q']:
#                     break
#                 print()
#                 #with console.status("[bold green]Finding the right tool for the job...[/bold green]", spinner="dots"):
#                 result = await agent.get_response(user_prompt)
#                 #    console.print("\n[bold magenta]Result:[/bold magenta]")
#                 #    console.print(Panel.fit(result, style="green"))
#                 print(result)
#
#             except KeyboardInterrupt:
#                 print("\nExiting...")
#                 break
#             except Exception as e:
#                 print(f"\nError occurred: {e}")