"""
Where things are, and which model to use. Everything is an environment variable, so the
same scripts run on a laptop with a free model, in class with a paid one, and in tests offline.

    RAG_REPO     path to your rag-zero-to-agentic repo         default: ../rag-zero-to-agentic
    RAG_PYTHON   the Python that runs the RAG MCP server       default: <RAG_REPO>/.venv/bin/python
    CV_DIR       the CVs the server reads                      default: <RAG_REPO>/data/cvs/candidates (invented CVs)
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

RAG_REPO = pathlib.Path(os.getenv("RAG_REPO", ROOT.parent / "rag-zero-to-agentic")).expanduser().resolve()
CV_DIR = pathlib.Path(os.getenv("CV_DIR", RAG_REPO / "data" / "cvs" / "candidates")).expanduser().resolve()
MODEL = os.getenv("MODEL", "openai:gpt-4o-mini")
SMALL_MODEL = os.getenv("SMALL_MODEL", MODEL)


def rag_python() -> str:
    """The RAG server runs in the RAG repo's own virtualenv (it needs torch, chromadb, pypdf…)."""
    if os.getenv("RAG_PYTHON"):
        return os.environ["RAG_PYTHON"]
    for candidate in (RAG_REPO / ".venv" / "bin" / "python", RAG_REPO / ".venv" / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def check():
    """Fail early, with a message a student can act on."""
    server = RAG_REPO / "challenge" / "mcp_server.py"
    if not server.exists():
        sys.exit(f"Can't find your RAG MCP server at {server}.\n"
                 f"Set RAG_REPO in .env to the folder of rag-zero-to-agentic.")
    if not CV_DIR.exists():
        sys.exit(f"CV_DIR does not exist: {CV_DIR}")
    if "challenge/cvs" in CV_DIR.as_posix() and not MODEL.startswith(("ollama", "fake")):
        print("⚠️  CV_DIR points at the REAL CVs and MODEL is a cloud API. Real people's CVs should stay\n"
              "   local: use the invented ones (data/cvs/candidates) or MODEL=ollama:qwen2.5:7b.", file=sys.stderr)
