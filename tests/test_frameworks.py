"""
Offline tests for frameworks/strands and frameworks/crewai: every step runs end to end with
MODEL=fake against the bundled CV MCP server. Skipped when the optional package is missing.
"""
import importlib.util
import json
import os
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENV = {**os.environ, "MODEL": "fake", "SMALL_MODEL": "fake", "PYTHONIOENCODING": "utf-8", "CREWAI_DISABLE_TELEMETRY": "true"}
ENV.pop("CREW_LLM", None)


def run(*args, stdin=None):
    p = subprocess.run([sys.executable, str(ROOT / "run.py"), *args], capture_output=True, text=True,
                       env=ENV, cwd=ROOT, timeout=900, input=stdin)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-3000:]
    return p.stdout


def interviews():
    f = ROOT / "outbox" / "interviews.jsonl"
    return len(f.read_text().splitlines()) if f.exists() else 0


FRAMEWORKS = [pytest.param("strands", marks=pytest.mark.skipif(not importlib.util.find_spec("strands"), reason="strands not installed")),
              pytest.param("crewai", marks=pytest.mark.skipif(not importlib.util.find_spec("crewai"), reason="crewai not installed"))]


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_02_first_agent_uses_mcp_tools(fw):
    out = run("02", "Which candidates know Kafka?", "--framework", fw)
    assert 'find_candidates({"skill": "kafka"' in out and "cand-006.md" in out


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_03_memory_keeps_one_conversation_per_thread(fw):
    out = run("03", "--framework", fw)
    assert ("Dana's session" in out) or ("Dana's thread: 2 turns" in out)


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_04_approve_books_and_reject_does_not(fw):
    before = interviews()
    out = run("04", "--framework", fw, "--auto")
    assert "⏸" in out and interviews() == before + 1                 # approved: one booking
    out = run("04", "--framework", fw, stdin="r\nWrong slot\n")
    assert "⏸" in out and interviews() == before + 1                 # rejected: nothing new booked


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_05_guardrail_blocks_booking_without_reading_the_cv(fw):
    out = run("05", "Book an interview for a Kafka candidate for the backend job.", "--framework", fw)
    assert "Blocked calls: 1" in out


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_06_workflow_same_shortlist_as_langgraph(fw):
    out = run("06", "backend-engineer", "--auto", "--framework", fw)
    assert "['cand-006.md', 'cand-018.md', 'cand-007.md']" in out and "approved = True" in out


def test_run_py_lists_all_frameworks():
    out = run("--list")
    assert all(name in out for name in ("langchain", "strands", "crewai"))
