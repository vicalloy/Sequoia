"""File system tools for the Sequoia AI agent."""

import asyncio
from pathlib import Path

_safe_directory = "./memory"


def _validate_path(path: str | Path) -> Path:
    """
    Validate that the given path is within the memory directory.

    Args:
        path: Path to validate

    Returns:
        Path object if valid
    """
    # Check if the path is within the memory directory
    path = Path(path)
    if not path.resolve().is_relative_to(Path(_safe_directory).resolve()):
        error_msg = f"Path {path} is not within the allowed directory."
        raise ValueError(error_msg)
    return path


def _validate_write_path(path: str | Path) -> Path:
    """
    Validate that the given path is within the memory directory for write operations.

    Args:
        path: Path to validate

    Returns:
        Path object if valid
    """
    path = _validate_path(path)

    # Ensure parent directories exist
    path.parent.mkdir(parents=True, exist_ok=True)

    return path


async def read_file(file_path: str) -> str:
    """
    Read the contents of a file within the memory directory.

    Args:
        file_path: Path to the file to read (relative to memory directory)

    Returns:
        File contents as a string, or an error message
    """
    try:
        path = _validate_path(file_path)
        loop = asyncio.get_event_loop()
        content = await loop.run_in_executor(None, path.read_text, "utf-8")
        return content
    except Exception as e:
        return f"Error reading file: {str(e)}"


async def write_file(file_path: str, content: str) -> str:
    """
    Write content to a file within the memory directory.

    Args:
        file_path: Path to the file to write (relative to memory directory)
        content: Content to write to the file

    Returns:
        Success message or an error message
    """
    try:
        path = _validate_write_path(file_path)
        path.write_text(content, "utf-8")
        return f"Successfully wrote to file: {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


async def create_directory(dir_path: str) -> str:
    """
    Create a directory within the memory directory.

    Args:
        dir_path: Path to the directory to create (relative to memory directory)

    Returns:
        Success message or an error message
    """
    try:
        _validate_write_path(Path(dir_path) / "temp")  # Validate parent path
        Path(dir_path).mkdir(mode=0o755, parents=True, exist_ok=True)
        return f"Successfully created directory: {dir_path}"
    except Exception as e:
        return f"Error creating directory: {str(e)}"


async def delete_file(file_path: str) -> str:
    """
    Delete a file within the memory directory.

    Args:
        file_path: Path to the file to delete (relative to memory directory)

    Returns:
        Success message or an error message
    """
    try:
        path = _validate_path(file_path)
        path.unlink()
        return f"Successfully deleted file: {file_path}"
    except Exception as e:
        return f"Error deleting file: {str(e)}"
