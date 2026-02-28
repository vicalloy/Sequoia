"""Configuration schema for Sequoia."""

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class MCPServerConfig(BaseModel):
    """Configuration for MCP server connections."""

    url: str = Field(..., description="URL of the MCP server")
    api_key: str | None = Field(None, description="API key for authentication")
    enabled: bool = Field(True, description="Whether this server is enabled")
    timeout: int = Field(30, description="Request timeout in seconds")
    retry_attempts: int = Field(3, description="Number of retry attempts")


class ToolsConfig(BaseModel):
    """Configuration for tools."""

    mcp_servers: list[MCPServerConfig] = Field(
        default_factory=list, description="List of MCP server configurations"
    )


class SequoiaConfig(BaseSettings):
    """Main configuration for Sequoia application."""

    # Tools configuration
    tools: ToolsConfig = Field(
        default_factory=ToolsConfig, description="Tools configuration"
    )


# For backward compatibility
Config = SequoiaConfig
