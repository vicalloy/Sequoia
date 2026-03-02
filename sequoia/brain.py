from collections.abc import AsyncGenerator

from pydantic_ai import (
    Agent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunContext,
)
from pydantic_ai.exceptions import UserError
from pydantic_ai_skills import SkillsToolset

from sequoia.memory import Memory

from .tools import get_current_time, get_current_timestamp, get_timezone_list

skills_toolset = SkillsToolset(directories=["./skills"])


async def add_skills(ctx: RunContext) -> str | None:
    """Add skills instructions to the agent's context."""
    return await skills_toolset.get_instructions(ctx)


class Brain:
    """Central scheduler for handling user input, LLM processing, and tool execution."""

    def get_memory_message(self):
        if not self.memory:
            return []
        return self.memory.get_pydantic_ai_messages()

    def add_memory_message(self, role: str, content: str):
        if not self.memory:
            return None
        return self.memory.add_message(role, content)

    def __init__(self, memory: Memory | None = None):
        self.memory = memory
        try:
            # Register tools with the agent
            self.agent = Agent(
                model="ollama:qwen3:8b",
                tools=[get_current_time, get_current_timestamp, get_timezone_list],
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
        except UserError as e:
            # If Ollama is not available, set agent to None
            print(f"Warning: Could not initialize Ollama agent: {e}")
            print("Please make sure Ollama is installed and running.")
            self.agent = None

    async def process_input(self, user_input: str) -> str:
        if self.agent is None:
            error_msg = "Error: Ollama not initialized. Check if Ollama is running."
            return error_msg

        self.add_memory_message("user", user_input)
        result = await self.agent.run(
            user_prompt=user_input, message_history=self.get_memory_message()
        )

        # Add assistant response to memory if available
        self.add_memory_message("assistant", result.output)

        return result.output

    def process_input_stream(self, user_input: str) -> AsyncGenerator[str, None]:
        async def _process_stream():
            if self.agent is None:
                error_msg = "Error: Ollama not initialized. Check if Ollama is running."
                # Add user message to memory if available
                yield error_msg
                return

            self.add_memory_message("user", user_input)

            # Use run_stream_events method to get streaming events
            try:
                async for event in self.agent.run_stream_events(
                    user_prompt=user_input, message_history=self.get_memory_message()
                ):
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
                        # Add assistant response to memory if available
                        if (
                            self.memory
                            and hasattr(event, "part")
                            and hasattr(event.part, "content")
                        ):
                            self.add_memory_message("assistant", event.part.content)
                        yield "\n"
            except Exception as e:
                yield f"Error: {str(e)}"

        return _process_stream()
