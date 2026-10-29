# Agents Workshop · from your RAG MCP server to a team of agents

By **Mohammad Kabajah**. The follow-up to [rag-zero-to-agentic](https://github.com/mkabajah/rag-zero-to-agentic), but standalone.

You already built a **RAG over CVs** and exposed it as an **MCP server**
that Claude Desktop, Claude Code or Kiro could call. In Claude Desktop, *Claude* was the agent.
Here you build the agent yourself, then a team of agents: a hiring assistant that searches CVs,
checks evidence, books interviews and asks a human before it acts.

**This repo runs on its own.** It ships a small CV MCP server (`cv_server/`, the same four tools
as the RAG course) and 22 invented CVs (`data/`). Want *your* RAG behind the agents instead? Set
`RAG_REPO` in `.env`: the agents don't change, only the server does. That's the point of MCP.

```
            ┌──────────────── agent side ───────────────┐          ┌──── an MCP server ──────────┐
 question → │  agent loop  ──tools/call──►  MCP client  │ ─stdio─► │ cv_server/   (default)      │
            │     ▲   │                                 │  or HTTP │   or YOUR RAG (RAG_REPO)     │
            │     │   └──► action tools (outbox/)       │          │ search_cvs · get_cv          │
            │  middleware: limits · approval · guards   │          │ find_candidates · list_jobs  │
            └───────────────────────────────────────────┘          └──────────────────────────────┘
```

## Setup (5 minutes)

```bash
git clone https://github.com/mkabajah/agents-workshop.git && cd agents-workshop
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                 # + requirements-optional.txt for Strands, CrewAI, AgentCore
cp .env.example .env                            # set MODEL and its API key
python check_setup.py                           # → ALL GOOD
```

| MCP server | How | Notes |
|---|---|---|
| **bundled** (default) | nothing to set | `cv_server/server.py`: BM25 search, no model download, starts in < 1 s |
| **your RAG** | `RAG_REPO=../rag-zero-to-agentic` | Runs its `challenge/mcp_server.py` in *its* `.venv` (levels 1–3 of `pipeline.py` working) |
| **remote** | `RAG_MCP_URL=…/mcp` + `MCP_TOKEN=…` | Any Streamable HTTP server: the RAG course's `step4_remote.py`, or one on AgentCore |

Claude Code works here too: `.mcp.json` registers the bundled server as `cv-agent`, and
`.claude/skills/` has the two CV skills from the RAG course.

**Which model?** Any model that can call tools. `MODEL=anthropic:claude-sonnet-4-6`, `openai:gpt-4o-mini`,
a free OpenAI-compatible API (`openai:openai/gpt-oss-20b` + `OPENAI_BASE_URL=https://api.groq.com/openai/v1`),
or local and free: `ollama:qwen2.5:7b`. `MODEL=fake` runs everything offline with a scripted stand-in.

**Privacy.** `CV_DIR` defaults to the **invented** CVs in `data/cvs`. Real CVs stay on your laptop:
use them only with a local model (Ollama).

## Same agent, three frameworks

LangChain + LangGraph is the default (what we teach). Every Session 1 step also exists in
**Strands Agents** and **CrewAI**, on the same MCP server: [`frameworks/`](frameworks/README.md).

```bash
python run.py 02 "Which candidates know Kafka?"                      # LangChain (default)
python run.py 02 "Which candidates know Kafka?" --framework strands
python run.py 04 --framework crewai                                  # approval before every write
python run.py --list
```

New to `async`/`await`, `async with` or decorators? Read [`docs/python-notes.md`](docs/python-notes.md) first (5 minutes).

## Session 1 · one agent

| Step | Run | You learn |
|---|---|---|
| 01 | `python session1/01_loop_by_hand.py "Who knows Kafka?"` | The agent loop in ~20 lines: the model **asks**, your code **acts** through MCP |
| 02 | `python session1/02_first_agent.py` | `create_agent` + your MCP server + a local tool |
| 03 | `python session1/03_memory.py` | Checkpointers and threads: "which of **them** lives in Haifa?" |
| 04 | `python session1/04_middleware_hitl.py` | Limits, summarisation, **human approval** before every write; resume after a restart (`--sqlite`) |
| 05 | `python session1/05_guardrail.py` | Your own middleware: *no action without evidence*, an audit log |
| 06 | `python session1/06_langgraph_screening.py backend-engineer` | LangGraph: a workflow you can draw, with an agent in one node and `interrupt()` |

Lab: [`docs/session1-lab.md`](docs/session1-lab.md)

## Session 2 · a team of agents, measured and safe

| Step | Run | You learn |
|---|---|---|
| 07 | `python session2/07_supervisor.py` | Supervisor: a lead whose tools are agents (sourcer, screener, coordinator) |
| 08 | `python session2/08_strands_agents.py [--swarm]` | The same team in **Strands Agents**: agents as tools, or a swarm |
| 09 | `python session2/09_crewai_crew.py backend-engineer` | The same team in **CrewAI**: roles, tasks, `mcps=[...]` |
| 10 | `python session2/10_tracing.py` | Tracing: every model and tool call, nested, with time and tokens (+ LangSmith) |
| 11 | `python session2/11_evals.py [--judge]` | An eval set: right tool, recall, precision, refusal, LLM-as-judge |
| 12 | `python session2/12_injection_demo.py` | Prompt injection through a CV, and the first defence |

Capstone: [`docs/session2-capstone.md`](docs/session2-capstone.md)

## More examples · `examples/`

Eleven short, single-pattern examples on the same MCP server: tool calling, structured output,
streaming, router, plan-and-execute, reflection, parallel fan-out, long-term memory, a Strands Graph,
and the hiring agent behind **AWS Bedrock AgentCore Runtime** (Strands and LangGraph).
See [`examples/README.md`](examples/README.md).

## Docs

| File | What |
|---|---|
| [`docs/python-notes.md`](docs/python-notes.md) | `async`/`await`, `async with`, decorators… in plain words |
| [`frameworks/README.md`](frameworks/README.md) | the three frameworks side by side, and which one to pick |
| [`docs/session1-lab.md`](docs/session1-lab.md) · [`docs/session2-capstone.md`](docs/session2-capstone.md) | the lab and the capstone brief + rubric |
| [`docs/resources.md`](docs/resources.md) | MCP servers, Agent Skills and repos worth reusing |

## Test

```bash
pytest -q        # every script, example and framework version, MODEL=fake, against the bundled MCP server
```

## Folders

```
session1/  session2/   the class scripts, step 01 → 12 (LangChain + LangGraph, the default)
frameworks/            the Session 1 steps again in Strands and CrewAI        run.py  runs any of them
examples/              11 single-pattern agents to read at home
workshop/              shared code: config, models (+ the offline fake), MCP client, prompts, write tools
cv_server/  data/      the CV MCP server · invented CVs, job posts, the eval set, the injection demo CV
docs/  tests/          the guides above · offline end-to-end tests
outbox/                what the agent "did": interviews, emails, shortlists, audit log (created on run, git-ignored)
```
