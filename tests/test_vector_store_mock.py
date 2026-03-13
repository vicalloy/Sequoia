"""Unit tests for Chroma vector store tools using mocks to avoid model downloads."""

import tempfile
from unittest.mock import Mock, patch

from sequoia.tools.vector_store import (
    AddDocumentTool,
    ChromaToolkit,
    DeleteDocumentTool,
    QueryDocumentTool,
    UpdateDocumentTool,
)


class TestAddDocumentToolWithMocks:
    """Test cases for AddDocumentTool using mocks."""

    def test_add_document_success(self):
        """Test successful addition of a document."""
        mock_db = Mock()
        tool = AddDocumentTool(db=mock_db)

        result = tool._run(
            content="Test content", doc_id="test_id", metadata={"source": "test"}
        )

        assert "Successfully added document" in result
        assert "test_id" in result

        # Verify the database method was called correctly
        mock_db.add_texts.assert_called_once_with(
            texts=["Test content"], ids=["test_id"], metadatas=[{"source": "test"}]
        )

    def test_add_document_without_id(self):
        """Test adding a document without specifying an ID."""
        mock_db = Mock()
        tool = AddDocumentTool(db=mock_db)

        result = tool._run(content="Test content without ID")

        assert "Successfully added document" in result
        assert "auto-generated" in result

        # Verify the database method was called correctly
        mock_db.add_texts.assert_called_once_with(texts=["Test content without ID"])

    def test_add_document_empty_content(self):
        """Test adding a document with empty content."""
        mock_db = Mock()
        tool = AddDocumentTool(db=mock_db)

        result = tool._run(content="")

        assert "Error: Content cannot be empty." in result

    def test_add_document_whitespace_content(self):
        """Test adding a document with only whitespace content."""
        mock_db = Mock()
        tool = AddDocumentTool(db=mock_db)

        result = tool._run(content="   ")

        assert "Error: Content cannot be empty." in result


class TestDeleteDocumentToolWithMocks:
    """Test cases for DeleteDocumentTool using mocks."""

    def test_delete_document_success(self):
        """Test successful deletion of a document."""
        mock_db = Mock()
        # Mock the get method to return existing document
        mock_db.get.return_value = {"ids": ["test_id"]}

        tool = DeleteDocumentTool(db=mock_db)
        result = tool._run(doc_id="test_id")

        assert "Successfully deleted document" in result
        assert "test_id" in result

        # Verify the database method was called correctly
        mock_db.delete.assert_called_once_with(ids=["test_id"])

    def test_delete_document_nonexistent(self):
        """Test deleting a non-existent document."""
        mock_db = Mock()
        # Mock the get method to return no documents
        mock_db.get.return_value = {"ids": []}

        tool = DeleteDocumentTool(db=mock_db)
        result = tool._run(doc_id="nonexistent_id")

        assert "does not exist in Chroma vector store" in result

    def test_delete_document_empty_id(self):
        """Test deleting a document with empty ID."""
        mock_db = Mock()
        tool = DeleteDocumentTool(db=mock_db)

        result = tool._run(doc_id="")

        assert "Error: Document ID cannot be empty." in result

    def test_delete_document_whitespace_id(self):
        """Test deleting a document with whitespace-only ID."""
        mock_db = Mock()
        tool = DeleteDocumentTool(db=mock_db)

        result = tool._run(doc_id="   ")

        assert "Error: Document ID cannot be empty." in result


class TestUpdateDocumentToolWithMocks:
    """Test cases for UpdateDocumentTool using mocks."""

    def test_update_document_success(self):
        """Test successful update of a document."""
        mock_db = Mock()
        # Mock the get method to return existing document
        mock_db.get.return_value = {"ids": ["test_id"]}

        tool = UpdateDocumentTool(db=mock_db)
        result = tool._run(
            doc_id="test_id", content="Updated content", metadata={"source": "updated"}
        )

        assert "Successfully updated document" in result
        assert "test_id" in result

        # Verify the database methods were called correctly (delete then add)
        assert mock_db.delete.call_count == 1
        mock_db.delete.assert_called_with(ids=["test_id"])
        mock_db.add_texts.assert_called_once_with(
            texts=["Updated content"],
            ids=["test_id"],
            metadatas=[{"source": "updated"}],
        )

    def test_update_document_nonexistent(self):
        """Test updating a non-existent document."""
        mock_db = Mock()
        # Mock the get method to return no documents
        mock_db.get.return_value = {"ids": []}

        tool = UpdateDocumentTool(db=mock_db)
        result = tool._run(doc_id="nonexistent_id", content="New content")

        assert "does not exist in Chroma vector store" in result

    def test_update_document_empty_id(self):
        """Test updating a document with empty ID."""
        mock_db = Mock()
        tool = UpdateDocumentTool(db=mock_db)

        result = tool._run(doc_id="", content="New content")

        assert "Error: Document ID cannot be empty." in result

    def test_update_document_empty_content(self):
        """Test updating a document with empty content."""
        mock_db = Mock()
        tool = UpdateDocumentTool(db=mock_db)

        result = tool._run(doc_id="test_id", content="")

        assert "Error: Content cannot be empty." in result


class TestQueryDocumentToolWithMocks:
    """Test cases for QueryDocumentTool using mocks."""

    def test_query_document_success(self):
        """Test successful querying of documents."""
        mock_db = Mock()

        # Mock the similarity_search method
        mock_doc = Mock()
        mock_doc.page_content = "Test document content for query"
        mock_doc.metadata = {"source": "test"}

        mock_db.similarity_search.return_value = [mock_doc]

        tool = QueryDocumentTool(db=mock_db)
        result = tool._run(query="test query", k=2)

        assert "Similar documents found:" in result
        assert "Test document content for query" in result

        # Verify the database method was called correctly
        mock_db.similarity_search.assert_called_once_with(query="test query", k=2)

    def test_query_document_empty_query(self):
        """Test querying with empty query."""
        mock_db = Mock()
        tool = QueryDocumentTool(db=mock_db)

        result = tool._run(query="", k=2)

        assert "Error: Query cannot be empty." in result

    def test_query_document_invalid_k(self):
        """Test querying with invalid k value."""
        mock_db = Mock()
        tool = QueryDocumentTool(db=mock_db)

        result = tool._run(query="test", k=0)

        assert "Error: Number of documents to retrieve must be positive." in result


class TestChromaToolkitWithMocks:
    """Test cases for ChromaToolkit using mocks."""

    def test_get_tools(self):
        """Test that all expected tools are returned."""
        with patch(
            "sequoia.tools.vector_store.HuggingFaceEmbeddings"
        ) as mock_embeddings_class:
            mock_embeddings = Mock()
            mock_embeddings_class.return_value = mock_embeddings

            with tempfile.TemporaryDirectory() as temp_dir:
                toolkit = ChromaToolkit(db_path=temp_dir)

                tools = toolkit.get_tools()
                tool_names = {tool.name for tool in tools}

                expected_tools = {
                    "add_document",
                    "delete_document",
                    "update_document",
                    "query_document",
                }
                assert expected_tools == tool_names

                # Verify each tool has the correct database instance
                for tool in tools:
                    assert hasattr(tool, "_db")
                    assert tool._db is toolkit.db

    def test_initialization_with_openai_embeddings_mocked(self):
        """Test toolkit initialization with OpenAI embeddings (mocked)."""
        # Mock OpenAI embeddings to avoid requiring API keys
        with patch(
            "sequoia.tools.vector_store.HuggingFaceEmbeddings"
        ) as mock_hf_embeddings_class:
            with patch("sequoia.tools.vector_store.OpenAIEmbeddings") as mock_openai:
                mock_embeddings = Mock()
                mock_hf_embeddings_class.return_value = mock_embeddings
                mock_openai.return_value = mock_embeddings

                with tempfile.TemporaryDirectory() as temp_dir:
                    toolkit = ChromaToolkit(db_path=temp_dir, use_openai=True)

                    # Verify OpenAI embeddings were used
                    mock_openai.assert_called_once()
                    assert toolkit.db is not None
