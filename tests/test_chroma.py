"""Unit tests for Chroma vector store tools."""

import tempfile
from unittest.mock import Mock, patch

import pytest
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from sequoia.tools.chroma import (
    AddDocumentTool,
    ChromaToolkit,
    DeleteDocumentTool,
    QueryDocumentTool,
    UpdateDocumentTool,
)


class TestAddDocumentTool:
    """Test cases for AddDocumentTool."""

    @pytest.fixture
    def chroma_db(self):
        """Create a temporary Chroma database for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            db = Chroma(
                collection_name="test_collection",
                embedding_function=embeddings,
                persist_directory=temp_dir,
            )
            yield db

    def test_add_document_success(self, chroma_db):
        """Test successful addition of a document."""
        tool = AddDocumentTool(db=chroma_db)
        result = tool._run(
            content="Test content", doc_id="test_id", metadata={"source": "test"}
        )

        assert "Successfully added document" in result
        assert "test_id" in result

        # Verify document was added
        results = chroma_db.get(ids=["test_id"])
        assert len(results["ids"]) == 1
        assert results["documents"][0] == "Test content"

    def test_add_document_without_id(self, chroma_db):
        """Test adding a document without specifying an ID."""
        tool = AddDocumentTool(db=chroma_db)
        result = tool._run(content="Test content without ID")

        assert "Successfully added document" in result
        assert "auto-generated" in result

    def test_add_document_empty_content(self, chroma_db):
        """Test adding a document with empty content."""
        tool = AddDocumentTool(db=chroma_db)
        result = tool._run(content="")

        assert "Error: Content cannot be empty." in result

    def test_add_document_whitespace_content(self, chroma_db):
        """Test adding a document with only whitespace content."""
        tool = AddDocumentTool(db=chroma_db)
        result = tool._run(content="   ")

        assert "Error: Content cannot be empty." in result


class TestDeleteDocumentTool:
    """Test cases for DeleteDocumentTool."""

    @pytest.fixture
    def chroma_db_with_doc(self):
        """Create a temporary Chroma database with a document."""
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            db = Chroma(
                collection_name="test_collection",
                embedding_function=embeddings,
                persist_directory=temp_dir,
            )
            # Add a test document
            db.add_texts(
                texts=["Test content"], ids=["test_id"], metadatas=[{"source": "test"}]
            )
            yield db

    def test_delete_document_success(self, chroma_db_with_doc):
        """Test successful deletion of a document."""
        tool = DeleteDocumentTool(db=chroma_db_with_doc)
        result = tool._run(doc_id="test_id")

        assert "Successfully deleted document" in result
        assert "test_id" in result

        # Verify document was deleted
        results = chroma_db_with_doc.get(ids=["test_id"])
        assert len(results["ids"]) == 0

    def test_delete_document_nonexistent(self, chroma_db_with_doc):
        """Test deleting a non-existent document."""
        tool = DeleteDocumentTool(db=chroma_db_with_doc)
        result = tool._run(doc_id="nonexistent_id")

        assert "does not exist in Chroma vector store" in result

    def test_delete_document_empty_id(self, chroma_db_with_doc):
        """Test deleting a document with empty ID."""
        tool = DeleteDocumentTool(db=chroma_db_with_doc)
        result = tool._run(doc_id="")

        assert "Error: Document ID cannot be empty." in result

    def test_delete_document_whitespace_id(self, chroma_db_with_doc):
        """Test deleting a document with whitespace-only ID."""
        tool = DeleteDocumentTool(db=chroma_db_with_doc)
        result = tool._run(doc_id="   ")

        assert "Error: Document ID cannot be empty." in result


class TestUpdateDocumentTool:
    """Test cases for UpdateDocumentTool."""

    @pytest.fixture
    def chroma_db_with_doc(self):
        """Create a temporary Chroma database with a document."""
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            db = Chroma(
                collection_name="test_collection",
                embedding_function=embeddings,
                persist_directory=temp_dir,
            )
            # Add a test document
            db.add_texts(
                texts=["Original content"],
                ids=["test_id"],
                metadatas=[{"source": "original"}],
            )
            yield db

    def test_update_document_success(self, chroma_db_with_doc):
        """Test successful update of a document."""
        tool = UpdateDocumentTool(db=chroma_db_with_doc)
        result = tool._run(
            doc_id="test_id", content="Updated content", metadata={"source": "updated"}
        )

        assert "Successfully updated document" in result
        assert "test_id" in result

        # Verify document was updated
        results = chroma_db_with_doc.get(ids=["test_id"])
        assert len(results["ids"]) == 1
        assert results["documents"][0] == "Updated content"
        assert results["metadatas"][0]["source"] == "updated"

    def test_update_document_nonexistent(self, chroma_db_with_doc):
        """Test updating a non-existent document."""
        tool = UpdateDocumentTool(db=chroma_db_with_doc)
        result = tool._run(doc_id="nonexistent_id", content="New content")

        assert "does not exist in Chroma vector store" in result

    def test_update_document_empty_id(self, chroma_db_with_doc):
        """Test updating a document with empty ID."""
        tool = UpdateDocumentTool(db=chroma_db_with_doc)
        result = tool._run(doc_id="", content="New content")

        assert "Error: Document ID cannot be empty." in result

    def test_update_document_empty_content(self, chroma_db_with_doc):
        """Test updating a document with empty content."""
        tool = UpdateDocumentTool(db=chroma_db_with_doc)
        result = tool._run(doc_id="test_id", content="")

        assert "Error: Content cannot be empty." in result


class TestQueryDocumentTool:
    """Test cases for QueryDocumentTool."""

    @pytest.fixture
    def chroma_db_with_docs(self):
        """Create a temporary Chroma database with multiple documents."""
        with tempfile.TemporaryDirectory() as temp_dir:
            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
            db = Chroma(
                collection_name="test_collection",
                embedding_function=embeddings,
                persist_directory=temp_dir,
            )
            # Add test documents
            db.add_texts(
                texts=["Python programming", "Machine learning", "Data science"],
                ids=["doc1", "doc2", "doc3"],
                metadatas=[
                    {"topic": "programming"},
                    {"topic": "AI"},
                    {"topic": "data"},
                ],
            )
            yield db

    def test_query_document_success(self, chroma_db_with_docs):
        """Test successful querying of documents."""
        tool = QueryDocumentTool(db=chroma_db_with_docs)
        result = tool._run(query="Python", k=2)

        assert "Similar documents found:" in result
        assert "Python programming" in result

    def test_query_document_empty_query(self, chroma_db_with_docs):
        """Test querying with empty query."""
        tool = QueryDocumentTool(db=chroma_db_with_docs)
        result = tool._run(query="", k=2)

        assert "Error: Query cannot be empty." in result

    def test_query_document_whitespace_query(self, chroma_db_with_docs):
        """Test querying with whitespace-only query."""
        tool = QueryDocumentTool(db=chroma_db_with_docs)
        result = tool._run(query="   ", k=2)

        assert "Error: Query cannot be empty." in result

    def test_query_document_invalid_k(self, chroma_db_with_docs):
        """Test querying with invalid k value."""
        tool = QueryDocumentTool(db=chroma_db_with_docs)
        result = tool._run(query="Python", k=0)

        assert "Error: Number of documents to retrieve must be positive." in result


class TestChromaToolkit:
    """Test cases for ChromaToolkit."""

    def test_get_tools(self):
        """Test that all expected tools are returned."""
        toolkit = ChromaToolkit()

        tools = toolkit.get_tools()
        tool_names = [tool.name for tool in tools]

        assert "add_document" in tool_names
        assert "delete_document" in tool_names
        assert "update_document" in tool_names
        assert "query_document" in tool_names

        # Verify each tool has the correct database instance
        for tool in tools:
            assert hasattr(tool, "db")
            assert tool.db is toolkit.db

    def test_initialization_with_defaults(self):
        """Test toolkit initialization with default parameters."""
        toolkit = ChromaToolkit()

        assert hasattr(toolkit, "db")
        assert toolkit.db is not None

    def test_initialization_with_custom_path(self):
        """Test toolkit initialization with custom path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            toolkit = ChromaToolkit(db_path=temp_dir)

            assert hasattr(toolkit, "db")
            assert toolkit.db is not None

    def test_initialization_with_openai_embeddings(self):
        """Test toolkit initialization with OpenAI embeddings."""
        # Mock OpenAI embeddings to avoid requiring API keys
        with patch("sequoia.tools.chroma.OpenAIEmbeddings") as mock_openai:
            mock_embeddings = Mock()
            mock_openai.return_value = mock_embeddings

            toolkit = ChromaToolkit(use_openai=True)

            # Verify OpenAI embeddings were used
            mock_openai.assert_called_once()
            assert toolkit.db is not None
