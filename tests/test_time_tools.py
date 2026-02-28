"""Unit tests for the time tools in Sequoia AI agent."""

import os
from datetime import datetime

import pytest
import pytz

# Set OLLAMA_BASE_URL before importing sequoia
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")

from sequoia.tools import get_current_time, get_current_timestamp, get_timezone_list


class TestTimeTools:
    """Test suite for time-related tools."""

    def test_get_current_time_without_timezone(self):
        """Test getting current time without specifying a timezone."""
        result = get_current_time()
        # Should return a valid ISO format datetime string
        assert result is not None
        assert isinstance(result, str)
        # Try to parse the result as datetime to ensure it's valid
        datetime.fromisoformat(result.replace("Z", "+00:00"))

    def test_get_current_time_with_valid_timezone(self):
        """Test getting current time with a valid timezone."""
        result = get_current_time("UTC")
        assert result is not None
        assert isinstance(result, str)
        # Should contain timezone info
        assert "Z" in result or "+" in result or "-" in result

        result = get_current_time("Asia/Shanghai")
        assert result is not None
        assert isinstance(result, str)
        # Try to parse the result as datetime to ensure it's valid
        datetime.fromisoformat(result)

    def test_get_current_time_with_invalid_timezone(self):
        """Test getting current time with invalid timezone (fallback to UTC)."""
        result = get_current_time("Invalid/Timezone")
        assert result is not None
        assert isinstance(result, str)
        # Should still return a valid datetime string

    def test_get_current_timestamp(self):
        """Test getting current Unix timestamp."""
        result = get_current_timestamp()
        assert result is not None
        assert isinstance(result, str)
        # Should be able to convert to integer
        timestamp = int(result)
        # Should be a recent timestamp (within last minute)
        current_time = int(datetime.now().timestamp())
        assert abs(timestamp - current_time) < 60

    def test_get_timezone_list(self):
        """Test getting the list of available timezones."""
        result = get_timezone_list()
        assert result is not None
        assert isinstance(result, str)
        # Should contain at least one timezone
        assert len(result) > 0
        # Should contain common timezones
        assert "UTC" in result
        assert "US/Eastern" in result
        assert "Europe/London" in result
        assert "Asia/Shanghai" in result

        # Verify that all returned timezones are valid
        timezones = result.split(", ")
        for tz in timezones[:10]:  # Test first 10 timezones to avoid performance issues
            if tz:  # Skip empty strings
                try:
                    pytz.timezone(tz)
                except pytz.exceptions.UnknownTimeZoneError:
                    pytest.fail(f"Invalid timezone in list: {tz}")

    def test_tool_functions_return_types(self):
        """Test that all tools return the expected types."""
        current_time = get_current_time()
        assert isinstance(current_time, str)

        timestamp = get_current_timestamp()
        assert isinstance(timestamp, str)

        timezones = get_timezone_list()
        assert isinstance(timezones, str)
