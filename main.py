import os

from dotenv import load_dotenv

from sequoia.cli import app

# Load environment variables from .env file
load_dotenv()

# Set default environment variables if not set in .env
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def main():
    app()


if __name__ == "__main__":
    main()
