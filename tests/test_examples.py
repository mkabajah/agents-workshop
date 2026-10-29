"""
Offline tests for examples/: each one runs with MODEL=fake against your RAG MCP server.
Strands, CrewAI and AgentCore examples are built (--dry) or served locally, and skipped when the
optional package is not installed.
"""
import importlib.util
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from workshop import config  # noqa: E402

pytestmark = pytest.mark.skipif(config.RAG_REPO is not None and not (config.RAG_REPO / "challenge" / "mcp_server.py").exists(),
                                reason=f"RAG repo not found at {config.RAG_REPO}")
ENV = {**os.environ, "MODEL": "fake", "SMALL_MODEL": "fake", "PYTHONIOENCODING": "utf-8"}


def run(script, *args, env=None):
    p = subprocess.run([sys.executable, str(ROOT / "examples" / script), *args], capture_output=True,
                       text=True, env=env or ENV, cwd=ROOT, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    return p.stdout


def has(pkg):
    return importlib.util.find_spec(pkg) is not None


def test_01_tool_calling():
    assert "ASKS to call: find_candidates" in run("01_tool_calling_basics.py", "Which candidates know Kafka?")


def test_02_structured_output_needs_real_model():
    assert "needs a real model" in run("02_structured_output.py")


def test_03_streaming():
    assert "[model] calling find_candidates" in run("03_streaming.py")


@pytest.mark.parametrize("question,route", [("Which candidates know Python?", "LIST"),
                                            ("What did cand-004 build?", "SEARCH"),
                                            ("What are the backend job must-haves?", "JOB")])
def test_04_router(question, route):
    assert f"route = {route}" in run("04_router.py", question)


def test_05_plan_and_execute():
    out = run("05_plan_and_execute.py")
    assert "✓ step 3" in out


def test_06_reflection():
    assert "After 2 round(s)" in run("06_reflection.py")


def test_07_fanout():
    out = run("07_parallel_fanout.py", "backend-engineer")
    assert "branches in parallel" in out and "cand-006.md" in out


def test_08_long_term_memory():
    assert "prefer candidates in Haifa" in run("08_long_term_memory.py")


@pytest.mark.skipif(not has("strands"), reason="strands-agents not installed")
def test_09_strands_graph_builds():
    env = {**ENV, "MODEL": "anthropic:claude-sonnet-4-6", "ANTHROPIC_API_KEY": "dummy"}
    assert "nodes: ['sourcer', 'screener', 'writer']" in run("09_strands_graph.py", "--dry", env=env)



@pytest.mark.skipif(not has("bedrock_agentcore"), reason="bedrock-agentcore not installed")
def test_11_agentcore_langgraph_serves_invocations(monkeypatch):
    monkeypatch.setenv("MODEL", "fake")
    monkeypatch.setattr(config, "MODEL", "fake")
    spec = importlib.util.spec_from_file_location("ac11", ROOT / "examples" / "11_agentcore_langgraph.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    from starlette.testclient import TestClient
    client = TestClient(mod.app)
    assert client.get("/ping").status_code == 200
    r = client.post("/invocations", json={"prompt": "Who knows Kafka?", "thread_id": "t1"})
    assert r.status_code == 200 and r.json()["tools_used"] == ["find_candidates"]
