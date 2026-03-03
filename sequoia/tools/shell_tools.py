"""Command execution tools for the Sequoia AI agent."""

import asyncio

# Whitelist of allowed read-only commands
ALLOWED_COMMANDS = ["ls", "grep", "find", "tree"]


async def run_shell_command(command_parts: list[str]) -> str:
    """
    Execute a command using with safety checks. support command: ls/grep/find/tree

    Args:
        command_parts: List of command parts [command, arg1, arg2, ...]

    Returns:
        Command output as a string, or an error message
    """
    if not command_parts:
        return "Error: No command provided"

    command = command_parts[0]

    # Check if command is in whitelist
    if command not in ALLOWED_COMMANDS:
        return f"Error: Command '{command}' is not allowed. "

    try:
        # Create subprocess
        process = await asyncio.create_subprocess_exec(
            *command_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Wait for completion with timeout
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=30.0)

        # Decode output
        stdout_str = stdout.decode() if stdout else ""
        stderr_str = stderr.decode() if stderr else ""

        # Return combined output
        if process.returncode == 0:
            return stdout_str.strip()
        return f"Command failed with return code {process.returncode}: {stderr_str}"

    except TimeoutError:
        return "Error: Command timed out"
    except Exception as e:
        return f"Error executing command: {str(e)}"
