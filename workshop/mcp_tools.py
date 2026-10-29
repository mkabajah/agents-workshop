"""
Your CV search, reached through an MCP server: the bundled one (cv_server/server.py) by default,
or the one you built in the RAG course (set RAG_REPO), or a remote one (set RAG_MCP_URL).

The agent never imports the RAG code. It starts the server as a child process (stdio), asks it
which tools it has (tools/list) and calls them (tools/call), exactly like Claude Desktop, Claude
Code or Kiro did. The server offers four tools:

    search_cvs(question)            passages from the CVs, with the file name
    get_cv(file_name)               one whole CV, contact details removed
    find_candidates(skill, city)    EVERY CV with that skill and/or city (not top-K)
    list_jobs()                     the open positions with must-haves
"""
import contextlib
import os

from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools

from workshop import config

# Settings the RAG server understands; we pass them through to its process.
PASSTHROUGH = ("RAG_EMBEDDER", "RAG_RERANKER", "RAG_LLM", "INJECTION_GUARD", "HF_HUB_OFFLINE",
               "OPENAI_BASE_URL", "OPENAI_API_KEY", "OPENAI_MODEL", "OLLAMA_MODEL", "PATH", "HOME")


def server_config(cv_dir=None, extra_env=None) -> dict:
    """The same JSON you put in Claude Desktop's config or .mcp.json, as a Python dict.

    Bundled (default): start cv_server/server.py as a child process over stdio.
    Your own RAG: set RAG_REPO to start its challenge/mcp_server.py instead.
    Remote: set RAG_MCP_URL (e.g. http://127.0.0.1:8000/mcp from the RAG repo's
    challenge/mcp_live/step4_remote.py, or an MCP server hosted on AgentCore) and MCP_TOKEN.
    """
    if os.getenv("RAG_MCP_URL"):
        headers = {"Authorization": f"Bearer {os.environ['MCP_TOKEN']}"} if os.getenv("MCP_TOKEN") else {}
        return {"cv": {"transport": "streamable_http", "url": os.environ["RAG_MCP_URL"], "headers": headers}}
    env = {k: os.environ[k] for k in PASSTHROUGH if k in os.environ}
    env["CV_DIR"] = str(cv_dir or config.CV_DIR)
    env.setdefault("FASTMCP_LOG_LEVEL", os.getenv("FASTMCP_LOG_LEVEL", "WARNING"))   # quieter server logs
    env.update(extra_env or {})
    if config.RAG_REPO is None:
        env["JOBS_DIR"] = str(config.JOBS_DIR)
        return {"cv": {"transport": "stdio", "command": config.rag_python(),
                       "args": ["-m", "cv_server.server"], "cwd": str(config.ROOT), "env": env}}
    return {
        "cv": {
            "transport": "stdio",
            "command": config.rag_python(),
            "args": [str(config.RAG_REPO / "challenge" / "mcp_server.py")],
            "cwd": str(config.RAG_REPO),
            "env": env,
        }
    }


def client(cv_dir=None, extra_env=None) -> MultiServerMCPClient:
    if not os.getenv("RAG_MCP_URL"):
        config.check()
    return MultiServerMCPClient(server_config(cv_dir, extra_env))


@contextlib.asynccontextmanager
async def rag_tools(cv_dir=None, extra_env=None):
    """Start the RAG server ONCE and yield its tools as LangChain tools.

        async with rag_tools() as tools:
            agent = create_agent(model, tools)

    Why a session: `await client.get_tools()` alone is simpler, but then every tool call starts a
    new server process, which rebuilds your index each time (seconds per call). One session = one
    process for the whole run.
    """
    async with client(cv_dir, extra_env).session("cv") as session:
        yield await load_mcp_tools(session)
