import os
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator, Callable

from pydantic_ai import Agent, PartDeltaEvent, PartEndEvent, PartStartEvent


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    @abstractmethod
    async def process(self, user_input: str) -> str:
        """Process user input using the LLM"""
        pass

    @abstractmethod
    def process_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """Process user input using the LLM and yield response
        in chunks for streaming"""
        pass


class OllamaProvider(LLMProvider):
    """Ollama LLM provider implementation"""

    def __init__(self, model: str | None = None, base_url: str | None = None):
        # Use model from environment variable or default to "qwen3:8b"
        self.model = model or os.getenv("DEFAULT_OLLAMA_MODEL", "qwen3:8b")
        self.agent: Agent | None = None

        try:
            self.agent = Agent(
                model=f"ollama:{self.model}",
            )
        except Exception as e:
            # Handle initialization error gracefully
            print(f"Warning: Failed to initialize Ollama provider: {e}")
            print(
                "Please make sure Ollama is installed and running, "
                "or set OLLAMA_BASE_URL environment variable"
            )
            self.agent = None

    async def process(self, user_input: str) -> str:
        """Process user input using Ollama LLM"""
        if self.agent is None:
            return (
                "Error: Ollama provider not initialized. "
                "Please make sure Ollama is installed and running."
            )
        result = await self.agent.run(user_input)
        return result.output

    def process_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """Process user input using Ollama LLM
        and yield response in chunks for streaming"""

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


class Brain:
    """Central scheduler for handling user input, LLM processing, and tool execution."""

    def __init__(self, llm_provider: LLMProvider | None = None):
        self.llm_provider = llm_provider or OllamaProvider()
        self.tools: dict[str, Callable] = {}

    async def process_input(self, user_input: str) -> str:
        """
        Process user input through the brain scheduler.

        This method handles:
        1. Direct command processing (if input is a command)
        2. LLM processing (for general queries)
        3. Tool execution (in the future)
        4. Skill invocation (in the future)
        """
        # For now, just send everything to the LLM
        # In the future, we can add logic to determine if it's a command vs a query
        return await self.llm_provider.process(user_input)

    async def process_input_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        """
        Process user input through the brain scheduler and yield response
        in chunks for streaming.

        This method handles:
        1. Direct command processing (if input is a command)
        2. LLM processing (for general queries) with streaming output
        3. Tool execution (in the future)
        4. Skill invocation (in the future)
        """
        # For now, just send everything to the LLM for streaming
        # In the future, we can add logic to determine if it's a command vs a query
        async for chunk in self.llm_provider.process_stream(user_input):
            yield chunk

    def add_tool(self, name: str, tool_func: Callable):
        """Add a tool that can be invoked by the brain"""
        self.tools[name] = tool_func

    def add_skill(self, name: str, skill_func: Callable):
        """Add a skill that can be invoked by the brain"""
        # For future implementation
        pass
