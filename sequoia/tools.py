"""Tools for the Sequoia AI agent."""

import time
from datetime import datetime

import pytz


def get_current_time(timezone: str | None = None) -> str:
    """
    Get the current time, optionally in a specific timezone.

    Args:
        timezone: Optional timezone string (e.g. 'UTC', 'US/Eastern', 'Asia/Shanghai').
                 If not provided, uses system local time.

    Returns:
        Current time in ISO format with timezone information.
    """
    if timezone:
        try:
            tz = pytz.timezone(timezone)
            current_time = datetime.now(tz)
        except pytz.exceptions.UnknownTimeZoneError:
            # If the timezone is invalid, fall back to UTC
            current_time = datetime.now(pytz.UTC)
    else:
        # Use local time if no timezone is specified
        current_time = datetime.now()

    return current_time.isoformat()


def get_current_timestamp() -> str:
    """
    Get the current Unix timestamp.

    Returns:
        Current Unix timestamp as a string.
    """
    return str(int(time.time()))


def get_timezone_list() -> str:
    """
    Get a list of available timezones.

    Returns:
        A string containing all available timezones, comma-separated.
    """
    # Get all available timezones from pytz
    all_timezones = pytz.all_timezones
    # Return a comma-separated string of all timezones
    return ", ".join(sorted(all_timezones))
