from collections.abc import AsyncGenerator

from pydantic_ai import Agent, PartDeltaEvent, PartEndEvent, PartStartEvent

from .tools import get_current_time, get_current_timestamp, get_timezone_list


class Brain:
    """Central scheduler for handling user input, LLM processing, and tool execution."""

    def __init__(self):
        # Register tools with the agent
        self.agent = Agent(
            model="ollama:qwen3:8b",
            tools=[get_current_time, get_current_timestamp, get_timezone_list],
        )

    async def process_input(self, user_input: str) -> str:
        result = await self.agent.run(user_input)
        return result.output

    def process_input_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        async def _process_stream():
            if self.agent is None:
                yield (
                    "Error: Ollama provider not initialized. "
                    "Please make sure Ollama is installed and running."
                )
                return

            # Use run_stream_events method to get streaming events
            try:
                async for event in self.agent.run_stream_events(user_input):
                    # Handle different event types appropriately
                    if isinstance(event, PartStartEvent):
                        # Handle different part types
                        if hasattr(event.part, "content"):
                            yield event.part.content
                        else:
                            continue
                    elif isinstance(event, PartDeltaEvent):
                        # Handle content deltas
                        if hasattr(event.delta, "content_delta"):
                            yield event.delta.content_delta
                        else:
                            # For tool call deltas, we don't yield anything to the user
                            continue
                    elif isinstance(event, PartEndEvent):
                        yield "\n"
            except Exception as e:
                yield f"Error: {str(e)}"

        return _process_stream()
