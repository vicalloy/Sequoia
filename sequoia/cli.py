import asyncio
import sys

import typer
from prompt_toolkit import PromptSession
from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from sequoia.brain import Brain


class ThinkingAnimation:
    """A class to handle thinking animation during LLM processing using rich spinner"""

    def __init__(self, console: Console):
        self.console = console
        self.spinner = Spinner("dots", text="Thinking...")
        self.live = Live(
            self.spinner, console=console, refresh_per_second=10, transient=True
        )
        self.animation_running = False

    def start(self):
        """Start the animation"""
        if not self.animation_running:
            self.animation_running = True
            self.live.start()

    def stop(self):
        """Stop the animation and clear the line"""
        if self.animation_running:
            self.live.stop()
            self.animation_running = False

    def is_running(self):
        """Check if animation is running"""
        return self.animation_running

    def stop_sync(self):
        """Synchronously stop the animation"""
        if self.animation_running:
            self.live.stop()
            self.animation_running = False


# Create console object for output
console = Console()

# Define command completer
command_completer = WordCompleter(
    ["/help", "/quit", "/bye", "/version"], ignore_case=True
)

# Define styles
style = Style.from_dict(
    {
        "prompt": "#884444",
        "help-text": "#ansigreen",
        "error": "#ansired",
        "info": "#ansiyellow",
    }
)

app = typer.Typer(add_completion=False)


class SequoiaCLI:
    """Sequoia Interactive CLI Class"""

    def __init__(self):
        self.session: PromptSession | None = None
        self.console = Console()
        self.running = True
        self.brain = Brain()

    def display_welcome(self):
        """Display welcome message"""
        welcome_text = Text.from_markup(
            "[bold blue]Welcome to Sequoia CLI![/bold blue]\n"
            "[italic]An AI-powered command line interface[/italic]\n"
            "Type [bold]/help[/bold] for commands or [bold]/quit[/bold] to exit.\n"
        )
        self.console.print(Panel(welcome_text, border_style="blue"))

    def display_help(self):
        """Display help information"""
        help_text = Text.from_markup(
            "[bold]Available Commands:[/bold]\n"
            "  [green]/help[/green]     - Show this help message\n"
            "  [green]/version[/green]  - Show version information\n"
            "  [green]/quit[/green]     - Quit the application\n"
            "  [green]/bye[/green]      - Quit the application\n\n"
            "[bold]CLI Options:[/bold]\n"
            "  [green]--help, -h[/green]    - Show CLI help and exit\n"
            "  [green]--version, -v[/green] - Show version and exit"
        )
        self.console.print(Panel(help_text, title="Help", border_style="green"))

    def display_version(self):
        """Display version information"""
        version_text = Text.from_markup(
            "[bold]Sequoia CLI[/bold]\n"
            f"Version: [green]0.1.0[/green]\n"
            f"Python: [blue]{sys.version}[/blue]\n"
        )
        self.console.print(Panel(version_text, title="Version", border_style="cyan"))

    async def process_command(self, command: str):
        """Process command"""
        command = command.strip().lower()

        if command.startswith("/"):
            # This is a CLI command
            if command in ["/quit", "/bye"]:
                self.console.print("[bold red]Goodbye![/bold red]")
                self.running = False
            elif command == "/help":
                self.display_help()
            elif command == "/version":
                self.display_version()
            else:
                self.console.print(
                    f"[red]Unknown command: {command}. Type /help for commands.[/red]"
                )
        else:
            # This is a user query to be processed by the brain
            try:
                # Create and start animation
                animation = ThinkingAnimation(self.console)
                animation.start()

                try:
                    # Use streaming output
                    first_chunk = True
                    async for chunk in self.brain.process_input_stream(command):
                        if first_chunk:
                            # Stop animation and clear the line
                            # when first chunk is received
                            animation.stop_sync()
                            first_chunk = False

                        print(chunk, end="", flush=True)

                    # Ensure newline after output
                    if not first_chunk:  # If we had output content
                        print()  # Add newline after streaming output
                    else:
                        # If no content was received,
                        # still clear animation and add newline
                        animation.stop_sync()
                        print()  # Add newline
                except Exception as e:
                    # Ensure animation stops in case of exception
                    animation.stop_sync()
                    raise e

            except Exception as e:
                self.console.print(f"[red]Error processing input: {str(e)}[/red]")

    async def run_interactive(self):
        """Run interactive CLI"""
        self.session = PromptSession(
            completer=command_completer, style=style, complete_while_typing=True
        )

        self.display_welcome()

        while self.running:
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
        console.print("Sequoia CLI Version: 0.1.0")
        return

    if show_help:
        console.print("Usage: python -m sequoia.cli [OPTIONS]")
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
