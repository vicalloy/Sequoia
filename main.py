import os

from sequoia.cli import app

# Set default environment variables
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")


def main():
    app()


if __name__ == "__main__":
    main()
