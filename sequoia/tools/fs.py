import json
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import BaseModel, Field

DATA_PATH = Path("./data").resolve()


def rename_parent_directories(
    path: str | Path, parent: str | Path, new_parent: str | Path
) -> Path:
    str_path = str(path)
    str_parent = str(parent)
    if str_path.startswith(str_parent):
        return Path(new_parent) / str_path[len(str_parent) :].lstrip("/")
    return Path(path)


def virtual_dir_to_real_dir(path: str | Path) -> Path:
    return rename_parent_directories(path, "/fs/", DATA_PATH)


def real_dir_to_virtual_dir(path: str | Path) -> Path:
    return rename_parent_directories(path, DATA_PATH, "/fs/")


def get_directory_tree(
    path: str | Path = ".", extra_args: list[str] | None = None
) -> str:
    path = virtual_dir_to_real_dir(path)
    args = ["tree"]
    if extra_args:
        args.extend(extra_args)
    proc = subprocess.run(
        args, capture_output=True, text=True, check=False, cwd=str(path)
    )

    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout).strip()
        raise RuntimeError(f"'tree' failed (exit {proc.returncode}): {err}")

    return proc.stdout


class TreeCommandInput(BaseModel):
    """Input for tree command."""

    path: str = Field(..., description="Path to the directory to list")
    extra_args: list[str] | None = Field(
        ...,
        description="Optional array of additional arguments to pass to "
        "the 'tree' command (e.g., ['-P', '*.md']).",
    )


class TreeCommandTool(BaseTool):
    name: str = "get_directory_tree"
    description: str = (
        "Return the directory tree for a given path by invoking "
        "the system 'tree' command. Accepts an optional path "
        "(defaults to current directory) and optional extra arguments."
    )
    args_schema: type | None = TreeCommandInput

    def _run(self, path: str = ".", extra_args: list[str] | None = None) -> str:
        """Execute the tree command to get directory structure."""
        try:
            return get_directory_tree(path=path, extra_args=extra_args)
        except Exception as exc:
            return f"Failed to run 'tree': {exc}"


def build_markdown_tree(path: str | Path) -> dict[str, Any]:
    root_path = virtual_dir_to_real_dir(path)

    if not root_path.exists():
        raise FileNotFoundError(f"Root path does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Root path is not a directory: {root_path}")

    def _walk(current: Path) -> dict:
        node: dict[str, Any] = {
            "name": current.name or str(current),
            "path": str(real_dir_to_virtual_dir(current)),
            "dirs": {},
            "files": {},
        }

        try:
            for item in current.iterdir():
                if item.is_dir():
                    node["dirs"][item.name] = _walk(item)
                elif item.is_file() and item.suffix.lower() == ".md":
                    try:
                        content = item.read_text(encoding="utf-8")
                    except (UnicodeDecodeError, OSError):
                        content = ""  # or log
                    node["files"][item.name] = {
                        "name": item.name,
                        "path": str(real_dir_to_virtual_dir(item)),
                        "content": content,
                    }
        except PermissionError:
            pass  # Skip directories without permission

        return node

    result = _walk(root_path)
    return result


class BuildMarkdownTreeInput(BaseModel):
    """Input for build markdown tree command."""

    path: str = Field(
        ..., description="Root path to start building the markdown tree from."
    )


class BuildMarkdownTreeTool(BaseTool):
    name: str = "build_markdown_tree"
    description: str = (
        "Build a hierarchical directory tree containing all .md "
        "file contents from a given root directory. Accepts a root "
        "path.  It returns a JSON-formatted string (str)."
    )
    args_schema: type | None = BuildMarkdownTreeInput

    def _run(self, path: str) -> str:
        """Build a markdown tree from the specified root directory."""
        try:
            data = build_markdown_tree(path)
            return json.dumps(data, indent=2, ensure_ascii=False)
        except Exception as exc:
            return f"Failed to build the markdown tree: {exc}"


class FsToolkit(BaseToolkit):
    """Toolkit for file system"""

    def get_tools(self) -> list[BaseTool]:
        """Return a list of tools for file system."""
        return [
            TreeCommandTool(),
            BuildMarkdownTreeTool(),
        ]
