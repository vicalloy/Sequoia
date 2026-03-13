"""
Unit tests for SurrealDB toolkit functionality - basic tests.
"""

from sequoia.tools.surrealdb import SurrealDBToolkit


class TestSurrealDBBasic:
    """Basic test suite for SurrealDBToolkit."""

    def test_surrealdb_toolkit_initialization(self):
        """Test that SurrealDB toolkit initializes correctly
        and returns expected number of tools."""
        # Initialize the toolkit (using memory mode for testing)
        toolkit = SurrealDBToolkit(db_path=":memory:")

        tools = toolkit.get_tools()

        # Verify the number of tools
        assert len(tools) == 5

        # Verify that tool names exist
        tool_names = [tool.name for tool in tools]
        expected_names = [
            "add_graph_node",
            "add_graph_edge",
            "delete_graph_node",
            "update_graph_node",
            "query_graph",
        ]
        for name in expected_names:
            assert name in tool_names
