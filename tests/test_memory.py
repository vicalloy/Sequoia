"""Test script for memory functionality using pytest."""

import os
import tempfile

import pytest

from sequoia.brain import Brain
from sequoia.memory import Memory


def test_memory_class():
    """Test the Memory class functionality."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as test_dir:
        # Initialize memory
        memory = Memory(memory_dir=test_dir)

        # Test adding messages
        memory.add_message("user", "Hello, how are you?")
        memory.add_message("assistant", "I'm doing well, thank you for asking!")
        memory.add_message("user", "What's the weather like today?")

        # Test getting history
        history = memory.get_history()
        assert len(history) == 3, f"Expected 3 messages, got {len(history)}"

        # Test getting limited history
        limited_history = memory.get_history(limit=2)
        assert len(limited_history) == 2, (
            f"Expected 2 messages, got {len(limited_history)}"
        )

        # Test clearing history
        memory.clear_history()
        cleared_history = memory.get_history()
        assert len(cleared_history) == 0, (
            f"Expected 0 messages after clearing, got {len(cleared_history)}"
        )


@pytest.mark.skip(
    reason="Skipping Brain integration test because it requires Ollama to be running"
)
def test_brain_with_memory():
    """Test Brain integration with Memory."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as test_dir:
        # Initialize memory and brain
        memory = Memory(memory_dir=test_dir)
        Brain(memory=memory)

        # Add a conversation to memory manually
        memory.add_message("user", "Hello, test conversation!")
        memory.add_message("assistant", "Hi there, this is a test response.")

        # Test getting history as Pydantic AI messages
        messages = memory.get_pydantic_ai_messages()
        assert len(messages) == 2, (
            f"Expected 2 Pydantic AI messages, got {len(messages)}"
        )

        # Test that directory was created
        assert os.path.exists(test_dir), "Memory directory was not created"
        assert os.path.exists(os.path.join(test_dir, "history.json")), (
            "History file was not created"
        )
