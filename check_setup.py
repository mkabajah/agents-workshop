"""
Run this before class:   python check_setup.py

  ✓ the CV MCP server (bundled, or yours via RAG_REPO) starts and lists its 4 tools
  ✓ one tool call works (find_candidates kafka)
  ✓ the model in MODEL answers one short message (skipped for MODEL=fake)
"""
import asyncio
import importlib
import sys
import time

from workshop import config

OK, BAD = "\033[92m✓\033[0m", "\033[91m✗\033[0m"


async def main():
    print(f"MCP server {config.server_mode()}\nRAG_REPO   {config.RAG_REPO or '(not set: bundled cv_server)'}\nRAG python {config.rag_python()}\nCV_DIR     {config.CV_DIR}\nMODEL      {config.MODEL}\n")
    for pkg in ("langchain", "langgraph", "langchain_mcp_adapters", "mcp"):
        try:
            mod = importlib.import_module(pkg)
            print(OK, pkg, getattr(mod, "__version__", ""))
        except ImportError:
            print(BAD, pkg, "missing: pip install -r requirements.txt")
            return 1
    config.check()
    from workshop.mcp_tools import rag_tools
    start = time.perf_counter()
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
    sys.exit(asyncio.run(main()))
