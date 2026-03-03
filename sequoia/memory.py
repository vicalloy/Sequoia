"""Memory module for handling conversation history in Sequoia."""

import json
import os
from datetime import datetime
from typing import Any

from pydantic_ai import SystemPromptPart, TextPart, UserPromptPart
from pydantic_ai.messages import ModelRequest, ModelResponse


def as_pydantic_ai_messages(role: str, content: str):
    if role == "user":
        return ModelRequest(parts=[UserPromptPart(content=content)])
    if role == "system":
        return ModelRequest(parts=[SystemPromptPart(content=content)])
    if role == "assistant":
        return ModelResponse(parts=[TextPart(content=content)])
    return None


class Memory:
    """Memory class for handling conversation history in Sequoia."""

    def __init__(self, memory_dir: str = "./memory", persist: bool = False):
        """
        Initialize the memory system.

        Args:
            memory_dir: Directory to store memory files
            persist: Whether to persist conversation history to JSON file
                             (default: False)
        """
        self.memory_dir = memory_dir
        self.persist = persist
        os.makedirs(memory_dir, exist_ok=True)
        self.history_file = os.path.join(memory_dir, "history.json")
        self.history = self._load_history() if persist else []

    def _load_history(self) -> list[dict[str, Any]]:
        """Load conversation history from file."""
        if self.persist and os.path.exists(self.history_file):
            try:
                with open(self.history_file, encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return []
        return []

    def _save_history(self) -> None:
        """Save conversation history to file."""
        if self.persist:
            with open(self.history_file, "w", encoding="utf-8") as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)

    def add_message(self, role: str, content: str) -> None:
        """
        Add a message to the conversation history.

        Args:
            role: Role of the message ("user" or "assistant")
            content: Content of the message
        """
        timestamp = datetime.now().isoformat()
        message = {"role": role, "content": content, "timestamp": timestamp}
        self.history.append(message)
        self._save_history()

    def add_messages(self, messages: list[dict[str, str]]) -> None:
        """
        Add multiple messages to the conversation history.

        Args:
            messages: List of messages to add, each with 'role' and 'content'
        """
        for message in messages:
            self.add_message(message["role"], message["content"])

    def get_history(self, limit: int | None = 10) -> list[dict[str, Any]]:
        """
        Get conversation history.

        Args:
            limit: Limit on number of messages to return (recent first).
                   Default is 10, use None for all messages.

        Returns:
            List of conversation messages
        """
        if limit is None:
            return self.history.copy()
        return self.history[-limit:]

    def get_pydantic_ai_messages(self, limit: int | None = 10) -> list:
        """
        Get conversation history as Pydantic AI messages.

        Args:
            limit: Limit on number of messages to return (recent first).
                   Default is 10, use None for all messages.

        Returns:
            List of Pydantic AI message objects
        """

        messages = []
        for item in self.get_history(limit):
            if message := as_pydantic_ai_messages(item["role"], item["content"]):
                messages.append(message)
        return messages

    def clear_history(self) -> None:
        """Clear the conversation history."""
        self.history = []
        self._save_history()

    def get_history_summary(self, limit: int | None = 10) -> str:
        """
        Get a formatted summary of recent conversation history.

        Args:
            limit: Number of recent messages to include.
                   Default is 10, use None for all messages.

        Returns:
            Formatted string of conversation history
        """
        history = self.get_history(limit)
        if not history:
            return "No conversation history available."

        summary_lines = ["Recent Conversation History:"]
        for i, msg in enumerate(history, 1):
            role = msg["role"].capitalize()
            timestamp = msg["timestamp"][:19]  # Get just the datetime part
            content = (
                msg["content"][:100] + "..."
                if len(msg["content"]) > 100
                else msg["content"]
            )
            summary_lines.append(f"{i}. [{timestamp}] {role}: {content}")

        return "\n".join(summary_lines)
