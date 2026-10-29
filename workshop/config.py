"""
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
"""
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUTBOX = ROOT / "outbox"                     # what the action tools "send" lands here, as JSON lines


def _load_dotenv():
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

DATA = ROOT / "data"
JOBS_DIR = DATA / "jobs"
SECURITY_DIR = DATA / "security"                 # Sam_Tester_CV.pdf, the injection demo
RAG_REPO = pathlib.Path(os.environ["RAG_REPO"]).expanduser().resolve() if os.getenv("RAG_REPO") else None
CV_DIR = pathlib.Path(os.getenv("CV_DIR", DATA / "cvs")).expanduser().resolve()


def server_mode() -> str:
    """remote (RAG_MCP_URL) > your own RAG (RAG_REPO) > bundled (default)."""
    if os.getenv("RAG_MCP_URL"):
        return "remote"
    return "rag-repo" if RAG_REPO else "bundled"


MODEL = os.getenv("MODEL", "openai:gpt-4o-mini")
SMALL_MODEL = os.getenv("SMALL_MODEL", MODEL)


def rag_python() -> str:
    """Bundled server: this Python. Your RAG server: the RAG repo's own virtualenv (torch, chromadb…)."""
    if os.getenv("RAG_PYTHON"):
        return os.environ["RAG_PYTHON"]
    if RAG_REPO is None:
        return sys.executable
    for candidate in (RAG_REPO / ".venv" / "bin" / "python", RAG_REPO / ".venv" / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def check():
    """Fail early, with a message a student can act on."""
    if RAG_REPO is not None and not (RAG_REPO / "challenge" / "mcp_server.py").exists():
        server = RAG_REPO / "challenge" / "mcp_server.py"
        sys.exit(f"Can't find your RAG MCP server at {server}.\n"
                 f"Fix RAG_REPO in .env, or remove it to use the bundled server.")
    if not CV_DIR.exists():
        sys.exit(f"CV_DIR does not exist: {CV_DIR}")
    if "challenge/cvs" in CV_DIR.as_posix() and not MODEL.startswith(("ollama", "fake")):
        print("⚠️  CV_DIR points at the REAL CVs and MODEL is a cloud API. Real people's CVs should stay\n"
              "   local: use the invented ones (data/cvs/candidates) or MODEL=ollama:qwen2.5:7b.", file=sys.stderr)
