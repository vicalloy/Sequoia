from dotenv import load_dotenv

from sequoia.cli.main import app

# Load environment variables from .env file


def _setup_logfire():
    import logfire  # noqa: PLC0415

    logfire.configure()
    logfire.instrument_pydantic_ai()


def setup_logfire():
    try:
        _setup_logfire()
    except Exception as e:
        print(f"setup logfire failed: {e}")


def main():
    load_dotenv()
    # setup_logfire()
    app()


if __name__ == "__main__":
    main()
