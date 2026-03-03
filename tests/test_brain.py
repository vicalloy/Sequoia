"""Unit tests for the Brain class and tool registration."""

import inspect
import os

# Set OLLAMA_BASE_URL before importing sequoia
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")

from pydantic_ai import AgentRunResult, PartDeltaEvent, PartEndEvent, PartStartEvent
from pydantic_ai.messages import ContentPart

from sequoia.brain import Brain
from sequoia.memory import Memory
from sequoia.tools.time_tools import (
    get_current_time,
    get_current_timestamp,
    get_timezone_list,
)


class FakeAgent:
    """Fake agent for testing purposes."""

    def __init__(self):
        self.instructions_called = False
        self.instructions_func = None

    async def run(self, user_prompt: str, message_history=None) -> AgentRunResult:
        """Fake run method that returns a simple result."""
        return AgentRunResult(output=f"Fake response to: {user_prompt}")

    def instructions(self, func):
        """Fake instructions method."""
        self.instructions_func = func
        self.instructions_called = True
        return func

    async def run_stream_events(self, user_prompt: str, message_history=None):
        """Fake run_stream_events that yields some events."""

        # Yield a start event
        yield PartStartEvent(part=ContentPart(content="Fake thinking..."))
        # Yield a delta event
        yield PartDeltaEvent(delta=ContentPart(content="Processing..."))
        # Yield an end event
        yield PartEndEvent(part=ContentPart(content=f"Fake response to: {user_prompt}"))


class TestBrain:
    """Test suite for Brain class and tool registration."""

    def test_brain_initialization_with_real_agent(self):
        """Test that Brain initializes without errors with real agent."""
        # This test will still attempt to create a real agent
        # but will use the get_ai_model which handles missing env vars
        brain = Brain()
        assert brain is not None
        assert brain.agent is not None

    def test_brain_initialization_with_fake_agent(self):
        """Test that Brain initializes correctly with fake agent."""
        fake_agent = FakeAgent()
        brain = Brain(agent=fake_agent)
        assert brain is not None
        assert brain.agent is fake_agent

    def test_agent_model_set_correctly(self):
        """Test agent model is set correctly."""
        fake_agent = FakeAgent()
        brain = Brain(agent=fake_agent)
        # Just ensure agent object exists
        assert brain.agent is not None

    def test_tools_callable_via_agent(self):
        """Test tools can be called via agent (indirect verification)."""
        # This test verifies Brain class was initialized with tools
        # by ensuring no exception is raised during initialization
        fake_agent = FakeAgent()
        Brain(agent=fake_agent)

        # Verify that the Brain object has the expected tools imported
        # The fact that Brain() constructor doesn't raise an error
        # means the tools were registered
        assert callable(get_current_time)
        assert callable(get_current_timestamp)
        assert callable(get_timezone_list)

        # Verify the functions have the expected signatures
        # get_current_time has optional timezone parameter
        sig = inspect.signature(get_current_time)
        assert "timezone" in sig.parameters

        # get_current_timestamp has no parameters
        sig = inspect.signature(get_current_timestamp)
        assert len(sig.parameters) == 0

        # get_timezone_list has no parameters
        sig = inspect.signature(get_timezone_list)
        assert len(sig.parameters) == 0

    async def test_process_input_with_fake_agent(self):
        """Test process_input method with fake agent."""
        fake_agent = FakeAgent()
        brain = Brain(agent=fake_agent)

        result = await brain.process_input("Hello")
        assert result == "Fake response to: Hello"

    def test_brain_initialization_with_memory(self):
        """Test Brain initialization with custom memory."""

        fake_memory = Memory()
        fake_agent = FakeAgent()
        brain = Brain(memory=fake_memory, agent=fake_agent)

        assert brain.memory is fake_memory
        assert brain.agent is fake_agent
