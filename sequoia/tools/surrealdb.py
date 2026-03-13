from typing import Any

from langchain_core.tools import BaseTool, BaseToolkit
from pydantic import BaseModel, Field, PrivateAttr
from surrealdb import RecordID, Surreal


def _parse_record_id(record_str: str) -> RecordID:
    """Parse a record string like 'person:tobie' into a RecordID object."""
    if ":" in record_str:
        table, id_part = record_str.split(":", 1)
        return RecordID(table, id_part)
    raise ValueError(
        f"Invalid record format: {record_str}. Expected format: 'table:id'"
    )


class AddGraphNodeInput(BaseModel):
    """Input for adding a node to SurrealDB graph."""

    node_type: str = Field(
        ..., description="The type of the node to add (e.g., 'person', 'company')"
    )
    properties: dict[str, Any] = Field(..., description="Properties of the node")
    node_id: str | None = Field(
        None,
        description="Optional node ID, if not provided, SurrealDB will generate one",
    )


class AddGraphNodeTool(BaseTool):
    """Tool for adding nodes to SurrealDB graph database."""

    name: str = "add_graph_node"
    description: str = "Add a node to the SurrealDB graph database"
    args_schema: type | None = AddGraphNodeInput
    _db: Surreal = PrivateAttr()

    def __init__(self, db: Surreal, **kwargs):
        super().__init__(**kwargs)
        self._db = db

    def _run(
        self, node_type: str, properties: dict[str, Any], node_id: str | None = None
    ) -> str:
        """Add a node to the SurrealDB graph database."""
        try:
            # Validate input
            if not node_type or not node_type.strip():
                return "Error: Node type cannot be empty."

            # Use the proper CRUD method based on whether node_id is provided
            if node_id:
                # Create node with specific ID using RecordID
                record_id = RecordID(node_type, node_id)
                result = self._db.create(record_id, properties)
            else:
                # Create node without specific ID
                result = self._db.create(node_type, properties)

            # Process the result from the create operation
            if result and isinstance(result, list) and len(result) > 0:
                node_info = result[
                    0
                ]  # For create operations, result is typically a list with one item
                node_id = node_info.get("id", "unknown")
                return f"Successfully added node to SurrealDB graph with id: {node_id}"
            if result and isinstance(result, dict) and "id" in result:
                node_id = result["id"]
                return f"Successfully added node to SurrealDB graph with id: {node_id}"
            return "Successfully added node to SurrealDB graph"

        except Exception as e:
            return f"Error adding node to SurrealDB graph: {str(e)}"


class AddGraphEdgeInput(BaseModel):
    """Input for adding an edge to SurrealDB graph."""

    edge_type: str = Field(
        ..., description="The type of the edge to add (e.g., 'knows', 'works_for')"
    )
    from_node: str = Field(
        ..., description="The ID of the source node (e.g., 'person:tobie')"
    )
    to_node: str = Field(
        ..., description="The ID of the target node (e.g., 'company:surrealdb')"
    )
    properties: dict[str, Any] = Field(..., description="Properties of the edge")
    edge_id: str | None = Field(
        None,
        description="Optional edge ID, if not provided, SurrealDB will generate one",
    )


class AddGraphEdgeTool(BaseTool):
    """Tool for adding edges to SurrealDB graph database."""

    name: str = "add_graph_edge"
    description: str = "Add an edge between two nodes in the SurrealDB graph database"
    args_schema: type | None = AddGraphEdgeInput
    _db: Surreal = PrivateAttr()

    def __init__(self, db: Surreal, **kwargs):
        super().__init__(**kwargs)
        self._db = db

    def _run(
        self,
        edge_type: str,
        from_node: str,
        to_node: str,
        properties: dict[str, Any],
        edge_id: str | None = None,
    ) -> str:
        """Add an edge between two nodes in the SurrealDB graph database."""
        try:
            # Validate input
            if not edge_type or not edge_type.strip():
                return "Error: Edge type cannot be empty."
            if not from_node or not from_node.strip():
                return "Error: Source node ID cannot be empty."
            if not to_node or not to_node.strip():
                return "Error: Target node ID cannot be empty."

            # Parse the from_node and to_node to get the RecordID objects
            from_record = _parse_record_id(from_node)
            to_record = _parse_record_id(to_node)

            # Add the from and to relationships to the properties
            edge_properties = {"in": from_record, "out": to_record, **properties}

            # Use proper CRUD method for creating a relation
            if edge_id:
                record_id = RecordID(edge_type, edge_id)
                result = self._db.create(record_id, edge_properties)
            else:
                # Create edge/relationship using the relation table
                result = self._db.create(edge_type, edge_properties)

            # Process the result
            message = "Successfully added edge to SurrealDB graph"
            if result and isinstance(result, list) and len(result) > 0:
                edge_info = result[0]
                edge_id = edge_info.get("id", "unknown")
                message = f"Successfully added edge to SurrealDB with id: {edge_id}"
            elif result and isinstance(result, dict) and "id" in result:
                edge_id = result["id"]
                message = f"Successfully added edge to SurrealDB with id: {edge_id}"

            return message

        except Exception as e:
            return f"Error adding edge to SurrealDB graph: {str(e)}"


class DeleteGraphNodeInput(BaseModel):
    """Input for deleting a node from SurrealDB graph."""

    node_id: str = Field(
        ..., description="The ID of the node to delete (e.g., 'person:tobie')"
    )


class DeleteGraphNodeTool(BaseTool):
    """Tool for deleting nodes from SurrealDB graph database."""

    name: str = "delete_graph_node"
    description: str = "Delete a node from the SurrealDB graph database using its ID"
    args_schema: type | None = DeleteGraphNodeInput
    _db: Surreal = PrivateAttr()

    def __init__(self, db: Surreal, **kwargs):
        super().__init__(**kwargs)
        self._db = db

    def _run(self, node_id: str) -> str:
        """Delete a node from the SurrealDB graph database by ID."""
        try:
            # Validate input
            if not node_id or not node_id.strip():
                return "Error: Node ID cannot be empty."

            # Parse the node_id to get the RecordID object
            record_id = _parse_record_id(node_id)

            # Check if node exists first by trying to select it
            existing_node = self._db.select(record_id)

            if not existing_node:
                return f"Node with ID '{node_id}' does not exist in SurrealDB graph."

            # Delete the node using the proper delete method
            result = self._db.delete(record_id)

            if (
                result is not None
            ):  # If delete was successful, it returns the deleted record
                return (
                    f"Successfully deleted node with ID: {node_id} from SurrealDB graph"
                )
            return f"Failed to delete node with ID: {node_id} from SurrealDB graph"

        except Exception as e:
            return f"Error deleting node from SurrealDB graph: {str(e)}"


class UpdateGraphNodeInput(BaseModel):
    """Input for updating a node in SurrealDB graph."""

    node_id: str = Field(
        ..., description="The ID of the node to update (e.g., 'person:tobie')"
    )
    properties: dict[str, Any] = Field(..., description="New properties for the node")


class UpdateGraphNodeTool(BaseTool):
    """Tool for updating nodes in SurrealDB graph database."""

    name: str = "update_graph_node"
    description: str = "Update an existing node in the SurrealDB graph database"
    args_schema: type | None = UpdateGraphNodeInput
    _db: Surreal = PrivateAttr()

    def __init__(self, db: Surreal, **kwargs):
        super().__init__(**kwargs)
        self._db = db

    def _run(self, node_id: str, properties: dict[str, Any]) -> str:
        """Update an existing node in the SurrealDB graph database."""
        # Validate input
        if not node_id or not node_id.strip():
            return "Error: Node ID cannot be empty."
        if not properties:
            return "Error: Properties cannot be empty."

        try:
            # Parse the node_id to get the RecordID object
            record_id = _parse_record_id(node_id)

            # Check if node exists first by trying to select it
            existing_node = self._db.select(record_id)

            if not existing_node:
                return f"Node with ID '{node_id}' does not exist in SurrealDB graph."

            # Update the node using the proper merge method to preserve existing fields
            result = self._db.merge(record_id, properties)

            # Process update result
            message = "Successfully updated node in SurrealDB graph"
            if result and isinstance(result, list) and len(result) > 0:
                node_info = result[0]
                node_id = node_info.get("id", "unknown")
                message = (
                    f"Successfully updated node with ID: {node_id} in SurrealDB graph"
                )
            elif result and isinstance(result, dict) and "id" in result:
                node_id = result["id"]
                message = (
                    f"Successfully updated node with ID: {node_id} in SurrealDB graph"
                )

            return message

        except Exception as e:
            return f"Error updating node in SurrealDB graph: {str(e)}"


class QueryGraphInput(BaseModel):
    """Input for querying nodes from SurrealDB graph."""

    query: str = Field(
        ..., description="The query text to search in the graph database"
    )
    params: dict[str, Any] | None = Field(
        None, description="Optional parameters for the query"
    )


class QueryGraphTool(BaseTool):
    """Tool for querying nodes from SurrealDB graph database."""

    name: str = "query_graph"
    description: str = "Query nodes and relationships from the SurrealDB graph database"
    args_schema: type | None = QueryGraphInput
    _db: Surreal = PrivateAttr()

    def __init__(self, db: Surreal, **kwargs):
        super().__init__(**kwargs)
        self._db = db

    def _run(self, query: str, params: dict[str, Any] | None = None) -> str:
        """Query nodes and relationships from the SurrealDB graph database."""
        try:
            # Validate input
            if not query or not query.strip():
                return "Error: Query cannot be empty."

            # Execute the query using the proper query method
            result = self._db.query(query, params or {})

            if result and isinstance(result, list) and len(result) > 0:
                # Format the results properly
                formatted_results = []
                for res in result:
                    if isinstance(res, dict) and "result" in res:
                        # If the result is wrapped in a "result" field from query method
                        result_data = res["result"]
                    else:
                        result_data = res

                    if isinstance(result_data, list):
                        for item in result_data:
                            formatted_results.append(str(item))
                    else:
                        formatted_results.append(str(result_data))

                return "Graph query results:\n" + "\n".join(formatted_results)
            return "No results found."
        except Exception as e:
            return f"Error querying graph from SurrealDB: {str(e)}"


class SurrealDBToolkit(BaseToolkit):
    """Toolkit for working with SurrealDB graph database."""

    _db: Surreal = PrivateAttr()

    def get_tools(self) -> list[BaseTool]:
        """Return a list of tools for interacting with SurrealDB graph database."""
        # ref: https://surrealdb.com/docs/sdk/python/api/core/surreal
        return [
            AddGraphNodeTool(db=self._db),
            AddGraphEdgeTool(db=self._db),
            DeleteGraphNodeTool(db=self._db),
            UpdateGraphNodeTool(db=self._db),
            QueryGraphTool(db=self._db),
        ]

    def __init__(self, db_path: str = "./data/sequoia-db"):
        """Initialize the SurrealDB graph database toolkit with embedded mode."""
        super().__init__()

        # Initialize SurrealDB client with proper URL format
        if db_path in [":memory:", "memory", "mem://"]:
            db_url = "mem://"
        # Ensure the path uses the file:// protocol for SurrealDB
        elif not db_path.startswith(("file://", "mem://", "http://", "https://")):
            if db_path.startswith("./"):
                db_path = db_path[2:]  # Remove ./
            db_url = f"file://{db_path}"
        else:
            db_url = db_path

        self._db = Surreal(db_url)

        # Connect to the database
        try:
            self._db.connect()
            self._db.use("sequoia", "graph")
            # self._db.signin({"user": "root", "pass": "root"})
        except Exception as e:
            raise ConnectionError(f"Failed to connect to SurrealDB: {str(e)}") from e
