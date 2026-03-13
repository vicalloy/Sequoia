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
        tools=[
            *ChromaToolkit().get_tools(),
            *SurrealDBToolkit().get_tools(),
        ],
        # memory=["/fs/memories/AGENTS.md"],
        skills=["/fs/skills/"],
        subagents=[generic_worker, researcher],
        backend=composite_backend,
        system_prompt=system_prompt,
    )
    return agent
