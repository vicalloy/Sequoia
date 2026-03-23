"""Novel-related helper tools.

This module exposes utility functions that can be used as LangChain tools
or called directly from Python. The primary helper in this module returns
a JSON structure describing the novel workspace:

- a directory tree focused on Markdown files under the novel output
  directory (`/fs/novel`)
- a structured representation of the outline folder content

The function is intentionally pure (it only reads the filesystem) and
returns a JSON string so it can be passed through tool channels or logged
directly by an LLM agent.
"""

import json
from pathlib import Path

from sequoia.tools.fs import build_markdown_tree, get_directory_tree


def get_novel_outline() -> str:
    """Get the complete outline and directory tree of
    the novel currently being written.

    Returns:
        A JSON-formatted string (str).
    """
    novel_dir = Path("/fs/novel")
    outline_dir = novel_dir / "outline"

    data: dict[str, object] = {
        "Directory Tree": get_directory_tree(novel_dir, ["-P", "*.md"]),
        "Outline Content Tree": build_markdown_tree(outline_dir),
    }

    return json.dumps(data, indent=2, ensure_ascii=False)


tools = [get_novel_outline]
