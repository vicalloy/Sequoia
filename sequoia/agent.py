import os

from deepagents import SubAgent, create_deep_agent
from deepagents.backends import FilesystemBackend
from langchain.chat_models import init_chat_model

system_prompt = "You are a helpful assistant"

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


def create_agent():
    model = init_chat_model(model=os.environ["MODEL_NAME"], streaming=True)
    agent = create_deep_agent(
        model=model,
        tools=[],
        # memory=["./AGENTS.md"],
        skills=["./skills/"],
        subagents=[generic_worker, researcher],
        backend=FilesystemBackend(root_dir="./data"),
        system_prompt=system_prompt,
    )
    return agent
