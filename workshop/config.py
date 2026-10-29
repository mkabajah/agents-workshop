"""
SETTINGS · where things are, and which model to use. Every other file imports this one first.

Where things are, and which model to use. Everything is an environment variable, so the
same scripts run on a laptop with a free model, in class with a paid one, and in tests offline.

    (nothing)    the repo runs on its own: the bundled CV MCP server (cv_server/) and the
                 invented CVs in data/cvs. No other repo needed.
    RAG_REPO     OPTIONAL · bring your own RAG: the path to rag-zero-to-agentic; the agents then
                 call YOUR challenge/mcp_server.py instead of the bundled server
    RAG_PYTHON   the Python that runs that server              default: <RAG_REPO>/.venv/bin/python
    RAG_MCP_URL  OPTIONAL · a remote MCP server (Streamable HTTP), with MCP_TOKEN
    CV_DIR       the CVs the server reads                      default: data/cvs (invented CVs)
    MODEL        the LLM, as "provider:model"                  default: openai:gpt-4o-mini
                 e.g. anthropic:claude-sonnet-4-6 · openai:gpt-4o-mini · ollama:qwen2.5:7b
                 · openai:openai/gpt-oss-20b with OPENAI_BASE_URL=https://api.groq.com/openai/v1
                 · fake  (a scripted model: no key, no network; for tests and dry runs)
    SMALL_MODEL  a cheaper model for summaries and routing     default: same as MODEL

You set these in the .env file (copy .env.example). Reading this file has a side effect on purpose:
importing it loads .env into the environment, so every script sees the same settings.
"""
import os
import pathlib
import sys

# The repo root: this file is workshop/config.py, so its parent folder's parent (parents[1]) is the root.
ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTBOX = ROOT / "outbox"                     # what the action tools "send" lands here, as JSON lines


def _load_dotenv():
    """Read KEY=value lines from .env into os.environ (a tiny version of the python-dotenv package).
    setdefault = a variable already set in your shell wins over the file."""
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()                                   # runs once, the first time any script imports config

DATA = ROOT / "data"
JOBS_DIR = DATA / "jobs"
SECURITY_DIR = DATA / "security"                 # Sam_Tester_CV.pdf, the injection demo
# None unless you set RAG_REPO: then we use YOUR RAG server instead of the bundled one.
RAG_REPO = pathlib.Path(os.environ["RAG_REPO"]).expanduser().resolve() if os.getenv("RAG_REPO") else None
CV_DIR = pathlib.Path(os.getenv("CV_DIR", DATA / "cvs")).expanduser().resolve()


def server_mode() -> str:
    """Which CV server the agents will talk to, as a word for check_setup.py to print.
    Priority: remote (RAG_MCP_URL) > your own RAG (RAG_REPO) > bundled (default)."""
    if os.getenv("RAG_MCP_URL"):
        return "remote"
    return "rag-repo" if RAG_REPO else "bundled"


MODEL = os.getenv("MODEL", "openai:gpt-4o-mini")
SMALL_MODEL = os.getenv("SMALL_MODEL", MODEL)


def rag_python() -> str:
    """The Python program that should start the CV server; returns its path.
    Bundled server: this same Python. Your RAG server: the RAG repo's own virtualenv (it has torch,
    chromadb… which this repo doesn't install). RAG_PYTHON overrides both."""
    if os.getenv("RAG_PYTHON"):
        return os.environ["RAG_PYTHON"]
    if RAG_REPO is None:
        return sys.executable
    for candidate in (RAG_REPO / ".venv" / "bin" / "python", RAG_REPO / ".venv" / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def check():
    """Stop with a clear message if the settings can't work (missing server or CV folder).
    Called before starting the server, so a typo in .env fails here, not deep inside MCP."""
    if RAG_REPO is not None and not (RAG_REPO / "challenge" / "mcp_server.py").exists():
        server = RAG_REPO / "challenge" / "mcp_server.py"
        sys.exit(f"Can't find your RAG MCP server at {server}.\n"
                 f"Fix RAG_REPO in .env, or remove it to use the bundled server.")
    if not CV_DIR.exists():
        sys.exit(f"CV_DIR does not exist: {CV_DIR}")
    # Privacy guard: the RAG course's REAL CVs (challenge/cvs) should only go to a local model.
    if "challenge/cvs" in CV_DIR.as_posix() and not MODEL.startswith(("ollama", "fake")):
        print("⚠️  CV_DIR points at the REAL CVs and MODEL is a cloud API. Real people's CVs should stay\n"
              "   local: use the invented ones (data/cvs/candidates) or MODEL=ollama:qwen2.5:7b.", file=sys.stderr)
