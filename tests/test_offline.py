"""
Offline tests: every script runs end to end with MODEL=fake against YOUR RAG MCP server.

    pytest -q                       # the bundled CV server (or RAG_REPO from .env)

No API key and no network needed. Set RAG_EMBEDDER=tfidf to skip the MiniLM download.
They check the plumbing (MCP, the loop, memory, approvals, graphs, evals), not the model's quality:
that is what session2/11_evals.py with a real MODEL is for.
"""
import json
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


def run(script, *args, stdin=""):
    p = subprocess.run([sys.executable, str(ROOT / script), *args], input=stdin, capture_output=True,
                       text=True, env=ENV, cwd=ROOT, timeout=600)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    return p.stdout


def test_loop_by_hand_calls_mcp():
    out = run("session1/01_loop_by_hand.py", "Who knows Kafka?")
    assert "find_candidates" in out and "cand-006.md" in out


def test_first_agent():
    out = run("session1/02_first_agent.py", "Which candidates know Kafka and Docker? Check the evidence.")
    assert "get_cv" in out


def test_memory_threads():
    out = run("session1/03_memory.py")
    assert "Thread A has 8 messages" in out


def test_hitl_approve_and_reject():
    approved = run("session1/04_middleware_hitl.py", "--auto", "Find a Kafka candidate and book an interview")
    assert "Booked cand-006.md" in approved
    rejected = run("session1/04_middleware_hitl.py", "Find a Kafka candidate and book an interview", stdin="r\nWrong slot\n")
    assert "rejected" in rejected


def test_guardrail_blocks_write_without_evidence():
    out = run("session1/05_guardrail.py")
    assert "Blocked: read the CV first" in out


def test_langgraph_screening():
    out = run("session1/06_langgraph_screening.py", "backend-engineer", "--auto")
    assert "approved=True" in out and "screen --> approve" in out


def test_supervisor_delegates():
    out = run("session2/07_supervisor.py", "--auto")
    for name in ("sourcer(", "screener(", "coordinator("):
        assert name in out


def test_tracing_table():
    out = run("session2/10_tracing.py")
    assert "tool sourcer" in out and "   🔧 tool" in out           # nested steps are indented


def test_evals_scoreboard():
    out = run("session2/11_evals.py")
    assert "Scoreboard" in out and "Passed" in out


def test_injection_guard():
    out = run("session2/12_injection_demo.py")
    line = next(x for x in out.splitlines() if x.startswith("hidden_instruction_reached_model"))
    assert "off=True" in line and "on=False" in line


def test_outbox_is_json_lines():
    for f in (ROOT / "outbox").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            json.loads(line)
