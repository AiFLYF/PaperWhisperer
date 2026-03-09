from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"


def load_project_env():
    """Load environment variables from the project-level .env file if present."""
    return load_dotenv(dotenv_path=DEFAULT_ENV_FILE, override=False, encoding="utf-8")
