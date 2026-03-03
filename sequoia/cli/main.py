"""Main CLI application module."""

import asyncio

import typer
from prompt_toolkit import PromptSession
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from sequoia.brain import Brain

from .command_handler import CLICommandHandler
from .dialogue_handler import LLMDialogueHandler, command_completer, style


class SequoiaCLI:
    """Sequoia Interactive CLI Class"""

    def __init__(self):
        self.session: PromptSession | None = None
        self.console = Console()
        self.brain = Brain()
        self.cli_command_handler = CLICommandHandler(self.console, self.brain)
        self.llm_dialogue_handler = LLMDialogueHandler(self.console, self.brain)

    def display_welcome(self):
        """Display welcome message"""
        welcome_text = Text.from_markup(
            "[bold blue]Welcome to Sequoia CLI![/bold blue]\n"
            "[italic]An AI-powered command line interface[/italic]\n"
            "Type [bold]/help[/bold] for commands or [bold]/quit[/bold] to exit.\n"
        )
        self.console.print(Panel(welcome_text, border_style="blue"))

    async def process_command(self, command: str):
        """Process command"""
        command = command.strip().lower()

        if command.startswith("/"):
            await self.cli_command_handler.handle_command(command)
        else:
            await self.llm_dialogue_handler.handle_dialogue(command)

    async def run_interactive(self):
        """Run interactive CLI"""
        self.session = PromptSession(
            completer=command_completer, style=style, complete_while_typing=True
        )

        self.display_welcome()

        while self.cli_command_handler.is_running:
            try:
                user_input = await self.session.prompt_async(
                    "sequoia> ", completer=command_completer, complete_while_typing=True
                )

                if user_input.strip():
                    await self.process_command(user_input)

            except KeyboardInterrupt:
                self.console.print(
                    "\n[bold red]Received keyboard "
                    "interrupt. Use /quit or /bye to "
                    "exit.[/bold red]"
                )
            except EOFError:
                self.console.print("\n[bold red]Goodbye![/bold red]")
                break


app = typer.Typer(add_completion=False)


@app.command()
def main(
    version: bool = typer.Option(
        False, "--version", "-v", help="Show version information"
    ),
    show_help: bool = typer.Option(
        None, "--help", "-h", help="Show help information and exit"
    ),
):
    """
    Sequoia CLI - An AI-powered command line interface
    """
    if version:
        console = Console()
        console.print("Sequoia CLI Version: 0.1.0")
        return

    if show_help:
        console = Console()
        console.print("Usage: python -m sequoia.cli.main [OPTIONS]")
        console.print("Options:")
        console.print("  --version, -v    Show version information")
        console.print("  --help, -h       Show this help information and exit")
        console.print("\nWithout options, starts interactive mode.")
        return

    # Enter interactive mode by default
    cli = SequoiaCLI()
    asyncio.run(cli.run_interactive())


if __name__ == "__main__":
    app()
