"""Test suite for sequoia.config module."""

import json
from pathlib import Path

from sequoia.config import Config, get_config_path, load_config
from sequoia.config.schema import MCPServerConfig, ToolsConfig


class TestConfigSchema:
    """Test the configuration schema classes."""

    def test_mcp_server_config_defaults(self):
        """Test MCPServerConfig default values."""
        config = MCPServerConfig(url="http://example.com")

        assert config.url == "http://example.com"
        assert config.api_key is None
        assert config.enabled is True
        assert config.timeout == 30
        assert config.retry_attempts == 3

    def test_mcp_server_config_custom_values(self):
        """Test MCPServerConfig with custom values."""
        config = MCPServerConfig(
            url="http://api.example.com",
            api_key="test-key",
            enabled=False,
            timeout=60,
            retry_attempts=5,
        )

        assert config.url == "http://api.example.com"
        assert config.api_key == "test-key"
        assert config.enabled is False
        assert config.timeout == 60
        assert config.retry_attempts == 5

    def test_tools_config_defaults(self):
        """Test ToolsConfig default values."""
        config = ToolsConfig()

        assert config.mcp_servers == []

    def test_sequoia_config_defaults(self):
        """Test SequoiaConfig default values."""
        config = Config()

        assert config.tools is not None
        assert isinstance(config.tools, ToolsConfig)
        assert config.tools.mcp_servers == []


class TestConfigFunctions:
    """Test configuration loading and saving functions."""

    def test_get_config_path(self):
        """Test getting default config path."""
        path = get_config_path()

        assert path == Path.home() / ".sequoia" / "config.json"
        assert str(path).endswith(".sequoia/config.json")

    def test_load_config_with_nonexistent_file(self, tmp_path):
        """Test loading config from non-existent file returns default."""
        fake_path = tmp_path / "nonexistent.json"

        config = load_config(fake_path)

        # Should return default config
        assert config.tools is not None
        assert config.tools.mcp_servers == []

    def test_load_config_from_valid_file(self, tmp_path):
        """Test loading config from a valid file."""
        config_file = tmp_path / "test_config.json"

        # Create a config file with some data
        test_data = {
            "tools": {
                "mcp_servers": [
                    {
                        "url": "http://test-server.com",
                        "api_key": "test-key",
                        "enabled": True,
                        "timeout": 45,
                        "retry_attempts": 2,
                    }
                ]
            }
        }

        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        config = load_config(config_file)

        # Verify the loaded config matches the file
        assert len(config.tools.mcp_servers) == 1
        server = config.tools.mcp_servers[0]
        assert server.url == "http://test-server.com"
        assert server.api_key == "test-key"
        assert server.enabled is True
        assert server.timeout == 45
        assert server.retry_attempts == 2

    def test_load_config_from_invalid_file(self, tmp_path, capsys):
        """Test loading config from an invalid file shows warning and uses defaults."""
        config_file = tmp_path / "invalid_config.json"

        # Create an invalid JSON file
        with open(config_file, "w", encoding="utf-8") as f:
            f.write("{ invalid json }")

        config = load_config(config_file)

        # Should return default config
        assert config.tools.mcp_servers == []

        # Check that warning was printed
        captured = capsys.readouterr()
        assert "Warning: Failed to load config" in captured.out
        assert "Using default configuration" in captured.out

    def test_load_config_with_none_path(self, monkeypatch, tmp_path):
        """Test loading config with None path uses default path."""
        # Create a temporary file to simulate the default config path
        temp_config_path = tmp_path / ".sequoia" / "config.json"
        temp_config_path.parent.mkdir(parents=True, exist_ok=True)

        # Create a valid config file at the temporary path
        test_data = {
            "tools": {
                "mcp_servers": [{"url": "http://default-server.com", "enabled": True}]
            }
        }

        with open(temp_config_path, "w", encoding="utf-8") as f:
            json.dump(test_data, f)

        # Mock Path.home() to return tmp_path instead of the real home directory
        def mock_home():
            return tmp_path

        monkeypatch.setattr(Path, "home", mock_home)

        # Load config with None (should use default path)
        config = load_config(None)

        # Verify the config was loaded from the mocked default path
        assert len(config.tools.mcp_servers) == 1
        assert config.tools.mcp_servers[0].url == "http://default-server.com"
