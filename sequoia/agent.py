import os

from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model


def create_agent():
    model = init_chat_model(model=os.environ["MODEL_NAME"], streaming=True)
    agent = create_deep_agent(
        model=model,
        tools=[],
    )
    return agent
