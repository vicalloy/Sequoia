from typing import Any

from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, BaseToolkit
from langchain_ollama import OllamaEmbeddings
from pydantic import BaseModel, Field, PrivateAttr


class AddDocumentInput(BaseModel):
    """Input for adding a document to Chroma vector store."""

    content: str = Field(..., description="The content of the document to add")
    doc_id: str | None = Field(
        None,
        description="Optional document ID, if not provided, Chroma will generate one",
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Optional metadata for the document"
    )


class AddDocumentTool(BaseTool):
    """Tool for adding documents to Chroma vector store."""

    name: str = "add_document"
    description: str = "Add a document to the Chroma vector store"
    args_schema: type | None = AddDocumentInput
    _db: Chroma = PrivateAttr()

    def __init__(self, db: Chroma, **kwargs):
        super().__init__(**kwargs)
        self._db = db

    def _run(
        self,
        content: str,
        doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Add a document to the Chroma vector store."""
        try:
            # Validate input
            if not content or not content.strip():
                return "Error: Content cannot be empty."

            # Prepare metadata if not provided
            if metadata is None:
                metadata = {}

            # Add the document to the vector store
            if doc_id:
                # Add with a specific ID
                ids = [doc_id]
                documents = [content]
                metadatas = [metadata]
                self._db.add_texts(texts=documents, ids=ids, metadatas=metadatas)
            else:
                # Add without a specific ID (let Chroma generate one)
                self._db.add_texts(texts=[content])

            return (
                f"Successfully added document to Chroma vector store with id: "
                f"{doc_id or 'auto-generated'}"
            )
        except Exception as e:
            return f"Error adding document to Chroma: {str(e)}"


class DeleteDocumentInput(BaseModel):
    """Input for deleting a document from Chroma vector store."""

    doc_id: str = Field(..., description="The ID of the document to delete")


class DeleteDocumentTool(BaseTool):
    """Tool for deleting documents from Chroma vector store."""

    name: str = "delete_document"
    description: str = "Delete a document from the Chroma vector store using its ID"
    args_schema: type | None = DeleteDocumentInput
    _db: Chroma = PrivateAttr()

    def __init__(self, db: Chroma, **kwargs):
        super().__init__(**kwargs)
        self._db = db

    def _run(self, doc_id: str) -> str:
        """Delete a document from the Chroma vector store by ID."""
        try:
            # Validate input
            if not doc_id or not doc_id.strip():
                return "Error: Document ID cannot be empty."

            # Check if document exists before deletion
            existing_docs = self._db.get(ids=[doc_id])
            if not existing_docs or len(existing_docs.get("ids", [])) == 0:
                return (
                    f"Document with ID '{doc_id}' does not exist in Chroma "
                    "vector store."
                )

            # Delete the document by ID
            self._db.delete(ids=[doc_id])
            return (
                f"Successfully deleted document with ID: {doc_id} "
                f"from Chroma vector store"
            )
        except Exception as e:
            return f"Error deleting document from Chroma: {str(e)}"


class UpdateDocumentInput(BaseModel):
    """Input for updating a document in Chroma vector store."""

    doc_id: str = Field(..., description="The ID of the document to update")
    content: str = Field(..., description="The new content of the document")
    metadata: dict[str, Any] | None = Field(
        None, description="Optional new metadata for the document"
    )


class UpdateDocumentTool(BaseTool):
    """Tool for updating documents in Chroma vector store."""

    name: str = "update_document"
    description: str = "Update an existing document in the Chroma vector store"
    args_schema: type | None = UpdateDocumentInput
    _db: Chroma = PrivateAttr()

    def __init__(self, db: Chroma, **kwargs):
        super().__init__(**kwargs)
        self._db = db

    def _run(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> str:
        """Update an existing document in the Chroma vector store."""
        try:
            # Validate input
            if not doc_id or not doc_id.strip():
                return "Error: Document ID cannot be empty."
            if not content or not content.strip():
                return "Error: Content cannot be empty."

            # Check if document exists
            existing_docs = self._db.get(ids=[doc_id])
            if not existing_docs or len(existing_docs.get("ids", [])) == 0:
                return (
                    f"Document with ID '{doc_id}' does not exist in "
                    "Chroma vector store."
                )

            # Update the document by deleting and re-adding with the same ID
            # Note: ChromaDB doesn't have a update method, so we delete and re-add
            self._db.delete(ids=[doc_id])

            # Prepare metadata if not provided
            if metadata is None:
                metadata = {}

            # Add the updated document with the same ID
            self._db.add_texts(texts=[content], ids=[doc_id], metadatas=[metadata])

            return (
                f"Successfully updated document with ID: {doc_id} "
                f"in Chroma vector store"
            )
        except Exception as e:
            return f"Error updating document in Chroma: {str(e)}"


class QueryDocumentInput(BaseModel):
    """Input for querying documents from Chroma vector store."""

    query: str = Field(..., description="The query text to search in the vector store")
    k: int | None = Field(5, description="Number of similar documents to retrieve")
    offset: int | None = Field(
        0, description="Offset to start from when retrieving documents"
    )


class QueryDocumentTool(BaseTool):
    """Tool for querying documents from Chroma vector store."""

    name: str = "query_document"
    description: str = "Query similar documents from the Chroma vector store"
    args_schema: type | None = QueryDocumentInput
    _db: Chroma = PrivateAttr()

    def __init__(self, db: Chroma, **kwargs):
        super().__init__(**kwargs)
        self._db = db

    def _run(self, query: str, k: int = 5, offset: int = 0) -> str:
        """Query similar documents from the Chroma vector store."""
        try:
            # Validate input
            if not query or not query.strip():
                return "Error: Query cannot be empty."

            # Ensure k is positive
            if k <= 0:
                return "Error: Number of documents to retrieve must be positive."

            # Ensure offset is non-negative
            if offset < 0:
                return "Error: Offset must be non-negative."

            # Perform similarity search with a larger k to account for offset
            total_results_needed = k + offset
            all_results = self._db.similarity_search(query, k=total_results_needed)

            # Apply offset and limit to results
            results = (
                all_results[offset : offset + k] if len(all_results) > offset else []
            )

            if not results:
                return "No similar documents found."

            # Format the results without content length limitation
            formatted_results = []
            for i, doc in enumerate(results):
                formatted_doc = (
                    f"Document {i + 1 + offset}:\n  Content: {doc.page_content}\n"
                )
                if doc.metadata:
                    formatted_doc += f"  Metadata: {doc.metadata}\n"
                formatted_results.append(formatted_doc)

            return "Similar documents found:\n" + "\n".join(formatted_results)
        except Exception as e:
            return f"Error querying documents from Chroma: {str(e)}"


class ChromaToolkit(BaseToolkit):
    """Toolkit for working with Chroma vector store."""

    _db: Chroma = PrivateAttr()
    _embeddings: OllamaEmbeddings = PrivateAttr()

    def get_tools(self) -> list[BaseTool]:
        """Return a list of tools for interacting with Chroma vector store."""
        return [
            AddDocumentTool(db=self._db),
            DeleteDocumentTool(db=self._db),
            UpdateDocumentTool(db=self._db),
            QueryDocumentTool(db=self._db),
        ]

    def __init__(self, db_path: str = "./data/"):
        """Initialize the Chroma database toolkit."""
        super().__init__()

        # Use Ollama embeddings with qwen3-embedding:0.6b model
        self._embeddings = OllamaEmbeddings(model="qwen3-embedding:0.6b")

        self._db = Chroma(
            collection_name="example_collection",
            embedding_function=self._embeddings,
            persist_directory=db_path,
        )
