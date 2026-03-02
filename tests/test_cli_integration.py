"""Integration test for the CLI with memory functionality using pytest."""

import os
import tempfile

import pytest

from sequoia.cli import SequoiaCLI
from sequoia.memory import Memory


@pytest.mark.skip(
    reason="Skipping CLI integration test because it requires Ollama to be running"
)
def test_cli_integration():
    """Test CLI integration with memory functionality."""
    # Create a temporary directory for testing
    with tempfile.TemporaryDirectory() as test_dir:
        # Initialize memory with temporary directory
        memory = Memory(memory_dir=test_dir)
        # Create CLI with temporary memory
        cli = SequoiaCLI()
        # Replace the CLI's memory with our temporary one
        cli.memory = memory

        # Test memory directory was created
        assert os.path.exists(cli.memory.memory_dir), (
            f"Memory directory {cli.memory.memory_dir} was not created"
        )

        # Test adding messages to memory
        cli.memory.add_message("user", "Hello from test")
        cli.memory.add_message("assistant", "Hello back from test")

        # Verify history has 2 messages
        history = cli.memory.get_history()
        assert len(history) == 2, f"Expected 2 messages in history, got {len(history)}"

        # Test clear command functionality
        cli.memory.clear_history()
        cleared_history = cli.memory.get_history()
        assert len(cleared_history) == 0, (
            f"Expected 0 messages after clearing, got {len(cleared_history)}"
        )
