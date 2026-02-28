from collections.abc import AsyncGenerator

from pydantic_ai import Agent, PartDeltaEvent, PartEndEvent, PartStartEvent


class Brain:
    """Central scheduler for handling user input, LLM processing, and tool execution."""

    def __init__(self):
        self.agent = Agent(model="ollama:qwen3:8b")

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
                    # print("|event:", event, "|")
                    if isinstance(event, PartStartEvent):
                        yield event.part.content
                    elif isinstance(event, PartDeltaEvent):
                        yield event.delta.content_delta
                    elif isinstance(event, PartEndEvent):
                        yield "\n"
            except Exception as e:
                yield f"Error: {str(e)}"

        return _process_stream()
