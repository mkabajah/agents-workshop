"""
Run this before class:   python check_setup.py

  ✓ the CV MCP server (bundled, or yours via RAG_REPO) starts and lists its 4 tools
  ✓ one tool call works (find_candidates kafka)
  ✓ the model in MODEL answers one short message (skipped for MODEL=fake)

Every ✗ line says what to fix. The usual ones: a package not installed (pip install -r requirements.txt),
a missing API key in .env, or RAG_REPO pointing at the wrong folder.
"""
import asyncio
import importlib
import sys
import time

from workshop import config

# Green tick and red cross (ANSI colour codes, shown as colours by the terminal).
OK, BAD = "\033[92m✓\033[0m", "\033[91m✗\033[0m"


async def main():
    """Run the checks in order and stop at the first failure. Returns 0 (all good) or 1 (fix something).
    async because talking to the MCP server and the model means waiting on them (see docs/python-notes.md)."""
    print(f"MCP server {config.server_mode()}\nRAG_REPO   {config.RAG_REPO or '(not set: bundled cv_server)'}\nRAG python {config.rag_python()}\nCV_DIR     {config.CV_DIR}\nMODEL      {config.MODEL}\n")
    for pkg in ("langchain", "langgraph", "langchain_mcp_adapters", "mcp"):
        try:
            mod = importlib.import_module(pkg)                # import by name, to test it is installed
            print(OK, pkg, getattr(mod, "__version__", ""))
        except ImportError:
            print(BAD, pkg, "missing: pip install -r requirements.txt")
            return 1
    config.check()
    from workshop.mcp_tools import rag_tools
    start = time.perf_counter()
    # Start the CV server, list its tools, then stop it again when the block ends.
    async with rag_tools() as tools:
        names = [t.name for t in tools]
        print(OK if len(names) >= 4 else BAD, "MCP server tools:", names, f"({time.perf_counter() - start:.1f}s)")
        find = next(t for t in tools if t.name == "find_candidates")
        result = await find.ainvoke({"skill": "kafka", "city": ""})
        print(OK, "find_candidates('kafka') →", str(result)[:160])
    if config.MODEL != "fake":
        from workshop.models import get_model
        try:
            reply = await get_model().ainvoke("Reply with the single word: ready")
            print(OK, "model answered:", str(reply.content)[:60])
        except Exception as exc:                       # noqa: BLE001
            print(BAD, "model call failed:", exc)
            return 1
    print("\nALL GOOD")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))                     # the exit code tells scripts/CI if setup is OK
