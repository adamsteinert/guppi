import asyncio
from MCPClient import MCPClient
from rich.console import Console
from rich.panel import Panel
from dotenv import load_dotenv
from llm.LanguageAgent import LanguageAgent

load_dotenv()  # load environment variables from .env

async def main():
    console = Console()
    agent = LanguageAgent()
    await agent.configure()

    while True:
        try:
            print("-" * 40)
            user_prompt = input("How can I help you?\n")
            print("-" * 40)
            if user_prompt.lower() in ['quit', 'exit', 'q']:
                break
            print()
            # with console.status("[bold green]Finding the right tool for the job...[/bold green]", spinner="dots"):
            result = await agent.get_response(user_prompt)
            #    console.print("\n[bold magenta]Result:[/bold magenta]")
            #    console.print(Panel.fit(result, style="green"))
            print(result)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")

#            console.print(f"\n[bold red]Error:[/bold red] {e}", style="red")
    await agent.cleanup()
if __name__ == "__main__":
    asyncio.run(main())