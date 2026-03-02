from dotenv import load_dotenv

from sequoia.cli import app

# Load environment variables from .env file
load_dotenv()


def main():
    app()


if __name__ == "__main__":
    main()
