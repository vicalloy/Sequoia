import os
from pathlib import Path

from deepagents import create_deep_agent
from deepagents.backends import CompositeBackend, FilesystemBackend, StateBackend
from langchain.chat_models import init_chat_model
from langchain_core.rate_limiters import InMemoryRateLimiter

from .subagents import subagents
from .tools.fs import FsToolkit
from .tools.novel import tools as novel_tools

# Init data paths
DATA_PATH = Path("./data").resolve()
DATA_PATH.mkdir(parents=True, exist_ok=True)
CONVERSATION_HISTORY_PATH = DATA_PATH / "conversation_history"
CONVERSATION_HISTORY_PATH.mkdir(parents=True, exist_ok=True)
MEMORY_FILE = DATA_PATH / "AGENTS.md"
MEMORY_FILE.touch(exist_ok=True)


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
            "/conversation_history/": FilesystemBackend(
                root_dir=CONVERSATION_HISTORY_PATH, virtual_mode=True
            ),
        },
    )


def create_agent():
    model = init_chat_model(
        model=os.environ["MODEL_NAME"], streaming=True, rate_limiter=rate_limiter
    )
    agent = create_deep_agent(
        model=model,
        tools=[*novel_tools, *FsToolkit().get_tools()],
        memory=["/fs/AGENTS.md"],
        skills=["/fs/skills/"],
        subagents=subagents,
        backend=composite_backend,
    )
    return agent
