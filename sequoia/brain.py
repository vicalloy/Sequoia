import os
from collections.abc import AsyncGenerator
from enum import Enum

from pydantic_ai import (
    Agent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunContext,
    ThinkingPart,
)
from pydantic_ai.models.openai import Model, OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai_skills import SkillsToolset

from .memory import Memory
from .tools import tools


class OutputDataType(str, Enum):
    """Enumeration for output data types."""

    THINKING = "thinking"
    CONTENT = "content"
    THINKING_END = "thinking-end"
    END = "end"
    ERROR = "error"


skills_toolset = SkillsToolset(directories=["./skills"])


async def add_skills(ctx: RunContext) -> str | None:
    """Add skills instructions to the agent's context."""
    return await skills_toolset.get_instructions(ctx)


def get_ai_model() -> Model | str:
    if os.getenv("OPENAI_API_KEY"):
        provider = OpenAIProvider(
            base_url=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        model = OpenAIChatModel(
            model_name=os.getenv("OPENAI_MODEL", ""), provider=provider
        )
        return model
    if os.getenv("OLLAMA_MODEL"):
        name = os.getenv("OLLAMA_MODEL", "qwen3:8b")
        return f"ollama:{name}"
    return ""


class Brain:
    """Central scheduler for handling user input, LLM processing, and tool execution."""

    def get_memory_message(self):
        return self.memory.get_pydantic_ai_messages()

    def add_memory_message(self, role: str, content: str):
        return self.memory.add_message(role, content)

    def __init__(self, memory: Memory | None = None, agent=None):
        self.memory = memory if memory else Memory(memory_dir="./memory")
        self.memory.clear_history()

        if agent is not None:
            # Use provided agent (for testing purposes)
            self.agent = agent
        else:
            # Register tools with the agent
            self.agent = Agent(
                model=get_ai_model(),
                system_prompt="",
                tools=tools,
                # https://ai.pydantic.dev/mcp/fastmcp-client/#usage
                toolsets=[
                    skills_toolset,
                    # FastMCPToolset('http://localhost:8000/mcp')
                    # FastMCPToolset(
                    #     fastmcp.StdioTransport(
                    #         command='python', args=['mcp_server.py']
                    #     )
                    # )
                ],
            )
            self.agent.instructions(add_skills)

    async def process_input(self, user_input: str) -> str:
        message_history = self.get_memory_message()
        self.add_memory_message("user", user_input)
        result = await self.agent.run(
            user_prompt=user_input, message_history=message_history
        )

        # Add assistant response to memory if available
        self.add_memory_message("assistant", result.output)

        return result.output

    def process_input_stream(
        self, user_input: str
    ) -> AsyncGenerator[tuple[OutputDataType, str], None]:
        async def _process_stream():
            # Use run_stream_events method to get streaming events
            try:
                message_history = self.get_memory_message()
                self.add_memory_message("user", user_input)
                data_type = OutputDataType.CONTENT
                async for event in self.agent.run_stream_events(
                    user_prompt=user_input, message_history=message_history
                ):
                    # Handle different event types appropriately
                    if isinstance(event, PartStartEvent):
                        part = event.part
                        # Handle different part types
                        if isinstance(part, ThinkingPart):
                            data_type = OutputDataType.THINKING
                        else:
                            data_type = OutputDataType.CONTENT
                        if hasattr(event.part, "content"):
                            yield data_type, event.part.content
                        else:
                            continue
                    elif isinstance(event, PartDeltaEvent):
                        # Handle content deltas
                        if hasattr(event.delta, "content_delta"):
                            yield data_type, event.delta.content_delta
                        else:
                            # For tool call deltas, we don't yield anything to the user
                            continue
                    elif isinstance(event, PartEndEvent):
                        # Add assistant response to memory if available
                        if (
                            self.memory
                            and hasattr(event, "part")
                            and hasattr(event.part, "content")
                        ):
                            self.add_memory_message("assistant", event.part.content)
                        if data_type == OutputDataType.THINKING:
                            yield OutputDataType.THINKING_END, "\n"
                        else:
                            yield OutputDataType.END, "\n"
            except Exception as e:
                yield OutputDataType.ERROR, f"Error: {str(e)}\n"

        return _process_stream()
