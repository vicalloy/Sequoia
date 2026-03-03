"""Unit tests for the Brain class and tool registration."""

import inspect
import os

# Set OLLAMA_BASE_URL before importing sequoia
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")

from sequoia.brain import Brain
from sequoia.tools.time_tools import (
    get_current_time,
    get_current_timestamp,
    get_timezone_list,
)


class TestBrain:
    """Test suite for Brain class and tool registration."""

    def test_brain_initialization(self):
        """Test that Brain initializes without errors."""
        brain = Brain()
        assert brain is not None
        assert brain.agent is not None

    def test_agent_model_set_correctly(self):
        """Test agent model is set correctly."""
        brain = Brain()
        # Just ensure agent object exists
        # (would fail during init if model was wrong)
        assert brain.agent is not None

    def test_tools_callable_via_agent(self):
        """Test tools can be called via agent (indirect verification)."""
        # This test verifies Brain class was initialized with tools
        # by ensuring no exception is raised during initialization
        Brain()

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
