import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.chat_models import init_chat_model

from .subagents import subagents

system_prompt = "You are a helpful assistant"

DATA_PATH = Path("./data").resolve()


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
        subagents=subagents,
        backend=composite_backend,
        system_prompt=system_prompt,
    )
    return agent
