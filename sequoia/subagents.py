import os

from deepagents import SubAgent

# Environment variable to control which subagents are enabled.
# SEQUOIA_SUBAGENTS: comma-separated list of subagent names (e.g.
# "generic,chroma"). Use "all" or leave unset to enable all.
SEQUOIA_SUBAGENTS_ENV = "SEQUOIA_SUBAGENTS"
subagents_names = [
    e.strip()
    for e in os.environ.get(SEQUOIA_SUBAGENTS_ENV, "").strip().split(",")
    if e.strip()
]

subagents = []


def is_subagent_enabled(name: str):
    return "all" in subagents_names or name in subagents_names


if is_subagent_enabled("generic"):
    subagents.append(
        SubAgent(
            name="generic_worker",
            description="A general-purpose executor capable of assuming various "
            "specific roles.",
            system_prompt="The role you are now playing is: {role}. Please follow the "
            "specific instructions below to complete the task: "
            "{specific_instructions}. You may use all provided tools to achieve "
            "your objective.",
            tools=[],
        )
    )

if is_subagent_enabled("researcher"):
    subagents.append(
        SubAgent(
            name="deep_researcher",
            description="Used for performing in-depth information retrieval",
            system_prompt="You are a research expert responsible for finding "
            "comprehensive information.",
            tools=[],
        )
    )

if is_subagent_enabled("chroma"):
    from sequoia.tools.chroma import ChromaToolkit

    subagents.append(
        SubAgent(
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
    )

if is_subagent_enabled("surrealdb"):
    from sequoia.tools.surrealdb import SurrealDBToolkit

    subagents.append(
        SubAgent(
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
    )
