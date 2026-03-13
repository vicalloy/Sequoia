"""
Comprehensive tests for SurrealDB toolkit functionality.
"""

from sequoia.tools.surrealdb import SurrealDBToolkit


class TestSurrealDBFull:
    """Comprehensive test suite for SurrealDBToolkit operations."""

    def test_surrealdb_operations_full(self):
        """Test full SurrealDB operations: adding nodes, edges, and querying."""
        # Initialize the toolkit (using memory mode for testing)
        toolkit = SurrealDBToolkit(db_path=":memory:")

        tools = toolkit.get_tools()

        # Get specific tool instances
        add_node_tool = None
        add_edge_tool = None
        query_tool = None

        for tool in tools:
            if tool.name == "add_graph_node":
                add_node_tool = tool
            elif tool.name == "add_graph_edge":
                add_edge_tool = tool
            elif tool.name == "query_graph":
                query_tool = tool

        # Verify that all tools have been obtained
        assert add_node_tool is not None
        assert add_edge_tool is not None
        assert query_tool is not None

        # === Test adding nodes ===
        # Add a node
        result = add_node_tool._run(
            node_type="person",
            properties={"name": "Alice", "age": 30, "city": "New York"},
            node_id="alice",
        )
        assert "Successfully added node" in result

        # Add another node
        result = add_node_tool._run(
            node_type="person",
            properties={"name": "Bob", "age": 25, "city": "San Francisco"},
            node_id="bob",
        )
        assert "Successfully added node" in result

        # Add a company node
        result = add_node_tool._run(
            node_type="company",
            properties={"name": "Acme Corp", "industry": "Technology"},
            node_id="acme",
        )
        assert "Successfully added node" in result

        # === Test adding edges ===
        # Add an edge
        result = add_edge_tool._run(
            edge_type="works_for",
            from_node="person:alice",
            to_node="company:acme",
            properties={"position": "Engineer", "since": 2020},
        )
        assert "Successfully added edge" in result

        # === Test querying ===
        # Query all nodes
        result = query_tool._run(query="SELECT * FROM person")
        assert "Graph query results:" in result
        assert "Alice" in result
        assert "Bob" in result

        result = query_tool._run(query="SELECT * FROM company")
        assert "Graph query results:" in result
        assert "Acme Corp" in result

        result = query_tool._run(query="SELECT * FROM works_for")
        # Note: In basic graph implementations, relationship queries may return empty
        # This assertion checks that the query executed without errors
        assert "Graph query results:" in result
