"""LLM dialogue handler module."""

from prompt_toolkit.completion import WordCompleter
from prompt_toolkit.styles import Style
from rich.console import Console
from rich.live import Live
from rich.spinner import Spinner

from sequoia.brain import Brain, OutputDataType


class LLMDialogueHandler:
    """Handles user queries to be processed by the LLM"""

    def __init__(self, console: Console, brain: Brain):
        self.console = console
        self.brain = brain

    async def handle_dialogue(self, command: str):
        """Handle user queries to be processed by the LLM"""
        spinner = Spinner("dots", text="Thinking...")
        live = Live(
            spinner, console=self.console, refresh_per_second=10, transient=True
        )
        try:
            # Create and start spinner animation
            live.start()
            # Use streaming output
            async for data_type, chunk in self.brain.process_input_stream(command):
                style: str | None = None
                text = chunk
                if data_type in [
                    OutputDataType.THINKING,
                    OutputDataType.THINKING_END,
                ]:
                    style = "dim"
                if data_type == OutputDataType.THINKING_END:
                    text = f"\n{'-' * 50}\n"
                if live.is_started:
                    live.stop()
                self.console.print(text, end="", style=style)

        except Exception as e:
            if live.is_started:
                live.stop()
            self.console.print(f"[red]Error processing input: {str(e)}[/red]")


# Define command completer
command_completer = WordCompleter(
    ["/help", "/quit", "/bye", "/version", "/history", "/clear"], ignore_case=True
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
