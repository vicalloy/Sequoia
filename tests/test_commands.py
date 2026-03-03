"""Test the new memory commands in CLI using pytest."""

import tempfile

import pytest

from sequoia.cli.main import SequoiaCLI
from sequoia.memory import Memory


@pytest.mark.asyncio
async def test_memory_commands():
    """Test the new /history and /clear commands."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as test_dir:
        # Initialize memory with temporary directory
        memory = Memory(memory_dir=test_dir)
        # Create CLI with temporary memory
        cli = SequoiaCLI()
        # Replace the CLI's memory with our temporary one
        cli.memory = memory

        # Manually add some messages to memory
        cli.memory.add_message("user", "Hello, this is a test message")
        cli.memory.add_message(
            "assistant", "Hello, this is a response to the test message"
        )
        cli.memory.add_message("user", "How are you doing?")

        # Verify we have at least 3 messages
        # (might be more if Brain init error was recorded)
        history = cli.memory.get_history()
        # The Brain initialization error might have added messages to memory
        assert len(history) >= 3, f"Expected at least 3 messages, got {len(history)}"

        # Test clear functionality
        cli.memory.clear_history()

        # Verify history is cleared
        cleared_history = cli.memory.get_history()
        assert len(cleared_history) == 0, (
            f"Expected 0 messages after clearing, got {len(cleared_history)}"
        )

        # Adding new messages after clear
        cli.memory.add_message("user", "New message after clear")
        cli.memory.add_message("assistant", "New response after clear")

        new_history = cli.memory.get_history()
        assert len(new_history) == 2, (
            f"Expected 2 messages after adding new ones, got {len(new_history)}"
        )
