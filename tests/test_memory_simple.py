"""Test script for memory functionality using pytest."""

import tempfile

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

        # Test getting Pydantic AI messages
        pydantic_messages = memory.get_pydantic_ai_messages()
        assert len(pydantic_messages) == 3, (
            f"Expected 3 Pydantic messages, got {len(pydantic_messages)}"
        )

        # Test clearing history
        memory.clear_history()
        cleared_history = memory.get_history()
        assert len(cleared_history) == 0, (
            f"Expected 0 messages after clearing, got {len(cleared_history)}"
        )
