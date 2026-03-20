import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

from .subagents import subagents

system_prompt = "You are a helpful assistant"

DATA_PATH = Path("./data").resolve()


rate_limiter: None | InMemoryRateLimiter = None
if "glm-4.7-flash" in os.getenv("MODEL_NAME", ""):
    rate_limiter = InMemoryRateLimiter(
        requests_per_second=0.5, check_every_n_seconds=0.1, max_bucket_size=1
    )


def composite_backend(rt):
    return CompositeBackend(
        default=StateBackend(rt),
        routes={
            "/fs/": FilesystemBackend(root_dir=DATA_PATH, virtual_mode=True),
        },
    )


def create_agent():
    model = init_chat_model(
        model=os.environ["MODEL_NAME"], streaming=True, rate_limiter=rate_limiter
    )
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
