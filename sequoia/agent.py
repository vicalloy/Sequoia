import os
from pathlib import Path

from deepagents import SubAgent, create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.chat_models import init_chat_model

from sequoia.tools.chroma import ChromaToolkit
from sequoia.tools.surrealdb import SurrealDBToolkit

system_prompt = "You are a helpful assistant"

DATA_PATH = Path("./data").resolve()


generic_worker = SubAgent(
    name="generic_worker",
    description="A general-purpose executor capable of assuming various "
    "specific roles.",
    system_prompt="The role you are now playing is: {role}. Please follow the specific "
    "instructions below to complete the task: {specific_instructions}. "
    "You may use all provided tools to achieve your objective.",
    tools=[],
)

researcher = SubAgent(
    name="deep_researcher",
    description="Used for performing in-depth information retrieval",
    system_prompt="You are a research expert responsible for finding "
    "comprehensive information.",
    tools=[],
)


# ChromaToolkit subagent
chroma_worker = SubAgent(
    name="chroma_worker",
    description=(
        "Primary subagent for vector-database operations using Chroma."
        " Use this agent for embedding generation, similarity search,"
        " vector storage, index management, and retrieval tasks."
        " Search keywords: vector, embeddings, chroma, similarity, retrieval."
    ),
    system_prompt=(
        "You are a Chroma Vector DB specialist. For embedding generation,"
        " similarity search, vector storage, and index management use the"
        " provided Chroma tools. Return concise, machine-readable outputs"
        " that include embedding IDs, document IDs, similarity scores,"
        " and relevant metadata."
    ),
    tools=ChromaToolkit().get_tools(),
)

# SurrealDBToolkit subagent — focused on graph data operations
surrealdb_worker = SubAgent(
    name="surrealdb_worker",
    description=(
        "Primary subagent for graph-database operations using SurrealDB."
        " Use this agent whenever you need graph storage, graph queries,"
        " relationship management, graph traversal, or schema inspection."
        " Search keywords: graph, graph-db, surrealdb, schema, relationships."
    ),
    system_prompt=(
        "You are a SurrealDB Graph specialist. For any graph-data task —"
        " creating/updating entities, traversing relationships, running"
        " graph queries, or managing schemas — use the provided SurrealDB"
        " tools. Return concise, machine-readable outputs that include"
        " table/schema names and record IDs when applicable."
    ),
    tools=SurrealDBToolkit().get_tools(),
)


def composite_backend(rt):
    return CompositeBackend(
        default=StateBackend(rt),
        routes={
            "/fs/": FilesystemBackend(root_dir=DATA_PATH, virtual_mode=True),
        },
    )


def create_agent():
    model = init_chat_model(model=os.environ["MODEL_NAME"], streaming=True)
    agent = create_deep_agent(
        model=model,
        tools=[],
        # memory=["/fs/memories/AGENTS.md"],
        skills=["/fs/skills/"],
        subagents=[
            generic_worker,
            researcher,
            chroma_worker,
            surrealdb_worker,
        ],
        backend=composite_backend,
        system_prompt=system_prompt,
    )
    return agent
