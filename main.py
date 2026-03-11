from dotenv import load_dotenv

from sequoia.agent import create_agent

load_dotenv()
app = create_agent()
