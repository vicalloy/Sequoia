"""Command handler module for CLI commands."""

import sys

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from sequoia.brain import Brain


class CLICommandHandler:
    """Handles CLI commands that start with /"""

    def __init__(self, console: Console, brain: Brain):
        self.console = console
        self.brain = brain
        self.running = True

    def _quit_command(self):
        """Handle quit/bye commands"""
        self.console.print("[bold red]Goodbye![/bold red]")
        self.running = False

    def _history_command(self):
        """Handle history command"""
        history_summary = self.brain.memory.get_history_summary()
        self.console.print(
            Panel(
                history_summary,
                title="Conversation History",
                border_style="blue",
            )
        )

    def _clear_command(self):
        """Handle clear command"""
        self.brain.memory.clear_history()
        self.console.print("[green]Conversation history cleared.[/green]")

    def display_help(self):
        """Display help information"""
        help_text = Text.from_markup(
            "[bold]Available Commands:[/bold]\n"
            "  [green]/help[/green]     - Show this help message\n"
            "  [green]/version[/green]  - Show version information\n"
            "  [green]/history[/green]  - Show conversation history\n"
            "  [green]/clear[/green]    - Clear conversation history\n"
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
            f"Python: [blue]{self._get_python_version()}[/blue]\n"
        )
        self.console.print(Panel(version_text, title="Version", border_style="cyan"))

        def _get_python_version(self):
            """Get Python version string"""

            return sys.version

    async def handle_command(self, command: str):
        """Handle a CLI command that starts with /"""
        command_handlers = {
            "/quit": self._quit_command,
            "/bye": self._quit_command,
            "/help": self.display_help,
            "/version": self.display_version,
            "/history": self._history_command,
            "/clear": self._clear_command,
        }

        handler = command_handlers.get(command)
        if handler:
            if command in ["/quit", "/bye"]:
                handler()
            else:
                handler()
        else:
            self.console.print(
                f"[red]Unknown command: {command}. Type /help for commands.[/red]"
            )

    @property
    def is_running(self):
        return self.running
