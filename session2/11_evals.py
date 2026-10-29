"""
SESSION 2 · STEP 11 · Evaluation: a small dataset beats a gut feeling.

    python session2/11_evals.py                 # run the agent on data/hiring_eval.jsonl, print a scoreboard
    python session2/11_evals.py --judge         # + LLM-as-judge: is every claim supported by the tool results?
    python session2/11_evals.py --langsmith     # also upload the dataset and the run to LangSmith

Three kinds of checks:
  PATH     (plain code)    did it call the right tool? under the step budget?
  ANSWER   (plain code)    does the answer name the expected CVs? does it refuse when nothing fits?
  JUDGE    (another model) is every claim grounded in what the tools returned?

Run it before and after every change to a prompt, a tool description, the model or the RAG.

New to async/await, "async with" or decorators? Read docs/python-notes.md (5 minutes).
"""
import argparse
import asyncio
import json
import pathlib
import re
import sys
import time

# Let Python find the shared "workshop" package in the repo root (docs/python-notes.md §3).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from pydantic import BaseModel

from workshop import config
from workshop.mcp_tools import rag_tools
from workshop.models import get_model
from workshop.prompts import HIRING_AGENT
from workshop.show import banner, content_text

DATA = pathlib.Path(__file__).resolve().parents[1] / "data" / "hiring_eval.jsonl"
REFUSAL = re.compile(r"\b(no (cv|candidate)|none of the|not (found|in the|mentioned)|don.?t know|no one|nobody|no matching)\b", re.I)


class Verdict(BaseModel):
    grounded: bool
    reason: str


# async def: this function waits on the model and the MCP server; each wait is marked with await.
async def run_agent(agent, question):
    start = time.perf_counter()
    out = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    msgs = out["messages"]
    return {"answer": content_text(msgs[-1]),
            "tools": [tc["name"] for m in msgs if isinstance(m, AIMessage) for tc in m.tool_calls],
            "evidence": "\n".join(content_text(m)[:1500] for m in msgs if isinstance(m, ToolMessage)),
            "seconds": time.perf_counter() - start}


def score(ex, out):
    answer = out["answer"].lower()
    found = [cv for cv in ex["expect_cvs"] if cv in answer]
    named = set(re.findall(r"cand-\d{3}(?:-\d{4})?", answer))          # the practice CVs' ids
    return {
        "right_tool": any(t in out["tools"] for t in ex["must_call"]),
        "in_budget": len(out["tools"]) <= ex.get("max_tool_calls", 5),
        "recall": len(found) / len(ex["expect_cvs"]) if ex["expect_cvs"] else None,
        "precision": len(named & set(ex["expect_cvs"])) / len(named) if named and ex["expect_cvs"] else None,
        "refused_ok": bool(REFUSAL.search(answer)) if ex.get("refuse") else None,
    }


async def judge(ex, out):
    model = get_model(config.SMALL_MODEL).with_structured_output(Verdict)
    return await model.ainvoke(
        "You grade a recruiting assistant. Is EVERY claim about a candidate in the ANSWER supported by the "
        f"TOOL RESULTS? Answer grounded=true/false and a one-line reason.\n\nQUESTION: {ex['question']}\n\n"
        f"TOOL RESULTS:\n{out['evidence'][:6000]}\n\nANSWER:\n{out['answer']}")


async def main(use_judge=False, use_langsmith=False):
    examples = [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]
    rows = []
    # Start the CV MCP server once. "async with" stops it again at the end, even after an error.
    async with rag_tools() as tools:
        agent = create_agent(get_model(), tools, system_prompt=HIRING_AGENT)
        for ex in examples:
            out = await run_agent(agent, ex["question"])
            s = score(ex, out)
            if use_judge:
                v = await judge(ex, out)
                s["grounded"] = v.grounded
            rows.append((ex, out, s))
            print(".", end="", flush=True)

    banner(f"Scoreboard · MODEL={config.MODEL}")
    print(f"{'id':<5}{'type':<12}{'tool':>6}{'budget':>8}{'recall':>8}{'prec.':>7}{'refuse':>8}{'judge':>7}{'calls':>7}{'sec':>7}")
    mark = lambda v: "-" if v is None else ("✓" if v is True else "✗" if v is False else f"{v:.2f}")
    for ex, out, s in rows:
        print(f"{ex['id']:<5}{ex['type']:<12}{mark(s['right_tool']):>6}{mark(s['in_budget']):>8}{mark(s['recall']):>8}{mark(s['precision']):>7}"
              f"{mark(s['refused_ok']):>8}{mark(s.get('grounded')):>7}{len(out['tools']):>7}{out['seconds']:>7.1f}")
    recall = [s["recall"] for _, _, s in rows if s["recall"] is not None]
    passed = sum(all(v in (True, None) or (isinstance(v, float) and v == 1.0) for v in s.values()) for _, _, s in rows)
    print(f"\nPassed {passed}/{len(rows)} · mean recall {sum(recall) / len(recall):.2f} · "
          f"tool calls {sum(len(o['tools']) for _, o, _ in rows)} · {sum(o['seconds'] for _, o, _ in rows):.0f}s")

    if use_langsmith:
        await upload(examples)
    return rows


async def upload(examples):
    """The same eval, tracked in LangSmith: a dataset + an experiment you can compare over time."""
    from langsmith import Client
    client, name = Client(), "hiring-agent-evals"
    if not client.has_dataset(dataset_name=name):
        ds = client.create_dataset(name)
        client.create_examples(dataset_id=ds.id, examples=[
            {"inputs": {"question": e["question"]}, "outputs": e} for e in examples])

    async with rag_tools() as tools:
        agent = create_agent(get_model(), tools, system_prompt=HIRING_AGENT)

        async def target(inputs):
            return await run_agent(agent, inputs["question"])

        def right_tool(outputs, reference_outputs):
            return any(t in outputs["tools"] for t in reference_outputs["must_call"])

        def recall(outputs, reference_outputs):
            exp = reference_outputs["expect_cvs"]
            return 1.0 if not exp else sum(cv in outputs["answer"].lower() for cv in exp) / len(exp)

        await client.aevaluate(target, data=name, evaluators=[right_tool, recall],
                               experiment_prefix=config.MODEL.replace(":", "-"))
    print("Uploaded: open the dataset 'hiring-agent-evals' in LangSmith → Experiments.")


# Runs only when you start this file yourself, not when a test imports it.
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--judge", action="store_true")
    p.add_argument("--langsmith", action="store_true")
    a = p.parse_args()
    # asyncio.run() starts the async function and waits until it is finished.
    asyncio.run(main(a.judge, a.langsmith))
