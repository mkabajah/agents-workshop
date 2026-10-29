# Agents Workshop · from your RAG MCP server to a team of agents

By **Mohammad Kabajah**. Builds on [rag-zero-to-agentic](https://github.com/mkabajah/rag-zero-to-agentic).

Two 2.5-hour sessions. You already built a **RAG over CVs** and exposed it as an **MCP server**
(`rag-zero-to-agentic/challenge/mcp_server.py`) that Claude Desktop, Claude Code or Kiro could call.
In Claude Desktop, *Claude* was the agent. Here you build the agent yourself, then a team of agents:
a hiring assistant that searches CVs, checks evidence, books interviews and asks a human before it acts.

```
            ┌──────────────── this repo ────────────────┐          ┌──── rag-zero-to-agentic ────┐
 question → │  agent loop  ──tools/call──►  MCP client  │ ─stdio─► │ challenge/mcp_server.py      │
            │     ▲   │                                 │          │ search_cvs · get_cv          │
            │     │   └──► action tools (outbox/)       │          │ find_candidates · list_jobs  │
            │  middleware: limits · approval · guards   │          │   → your pipeline + Chroma   │
            └───────────────────────────────────────────┘          └──────────────────────────────┘
```

## Setup (10 minutes)

```bash
cd ~/rag-run                                    # next to rag-zero-to-agentic
cd agents-workshop
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt                 # + requirements-optional.txt for Strands and CrewAI (session 2)
cp .env.example .env                            # set MODEL and its API key
python check_setup.py                           # → ALL GOOD
```

The RAG server runs in the RAG repo's **own** virtualenv (`../rag-zero-to-agentic/.venv`), so this
repo only needs the agent libraries. Your `challenge/pipeline.py` must have levels 1–3 working
(behind? `cp challenge/checkpoints/level3.py challenge/pipeline.py` in the RAG repo).

**Which model?** Any model that can call tools. `MODEL=anthropic:claude-sonnet-4-6`, `openai:gpt-4o-mini`,
a free OpenAI-compatible API (`openai:openai/gpt-oss-20b` + `OPENAI_BASE_URL=https://api.groq.com/openai/v1`),
or local and free: `ollama:qwen2.5:7b`. `MODEL=fake` runs everything offline with a scripted stand-in.

**Privacy.** `CV_DIR` defaults to the **invented** CVs in `rag-zero-to-agentic/data/cvs/candidates`.
The real CVs (`challenge/cvs/`) stay on your laptop: use them only with a local model (Ollama).

## Session 1 · one agent

| Step | Run | You learn |
|---|---|---|
| 01 | `python session1/01_loop_by_hand.py "Who knows Kafka?"` | The agent loop in ~20 lines: the model **asks**, your code **acts** through MCP |
| 02 | `python session1/02_first_agent.py` | `create_agent` + your MCP server + a local tool |
| 03 | `python session1/03_memory.py` | Checkpointers and threads: "which of **them** lives in Haifa?" |
| 04 | `python session1/04_middleware_hitl.py` | Limits, summarisation, **human approval** before every write; resume after a restart (`--sqlite`) |
| 05 | `python session1/05_guardrail.py` | Your own middleware: *no action without evidence*, an audit log |
| 06 | `python session1/06_langgraph_screening.py backend-engineer` | LangGraph: a workflow you can draw, with an agent in one node and `interrupt()` |

Lab: [`labs/SESSION1_LAB.md`](labs/SESSION1_LAB.md)

## Session 2 · a team of agents, measured and safe

| Step | Run | You learn |
|---|---|---|
| 07 | `python session2/07_supervisor.py` | Supervisor: a lead whose tools are agents (sourcer, screener, coordinator) |
| 08 | `python session2/08_strands_agents.py [--swarm]` | The same team in **Strands Agents**: agents as tools, or a swarm |
| 09 | `python session2/09_crewai_crew.py backend-engineer` | The same team in **CrewAI**: roles, tasks, `mcps=[...]` |
| 10 | `python session2/10_tracing.py` | Tracing: every model and tool call, nested, with time and tokens (+ LangSmith) |
| 11 | `python session2/11_evals.py [--judge]` | An eval set: right tool, recall, precision, refusal, LLM-as-judge |
| 12 | `python session2/12_injection_demo.py` | Prompt injection through a CV, and the first defence |

Capstone: [`labs/SESSION2_CAPSTONE.md`](labs/SESSION2_CAPSTONE.md)

## More examples · `examples/`

Twelve short, single-pattern examples on the same MCP server: tool calling, structured output,
streaming, router, plan-and-execute, reflection, parallel fan-out, long-term memory, a Strands Graph,
a CrewAI Flow, and the hiring agent behind **AWS Bedrock AgentCore Runtime** (Strands and LangGraph).
See [`examples/README.md`](examples/README.md). Which framework to teach and pick: [`FRAMEWORKS.md`](FRAMEWORKS.md).

**Remote MCP:** set `RAG_MCP_URL=http://127.0.0.1:8000/mcp` and `MCP_TOKEN=…` to use the RAG repo's
`challenge/mcp_live/step4_remote.py` (or an MCP server on AgentCore) instead of a local process.

## Test

```bash
RAG_EMBEDDER=tfidf pytest -q        # 24 tests: every script and example, MODEL=fake, against your MCP server
```

## Files

```
workshop/      config.py (env) · models.py (get_model + the fake) · mcp_tools.py (your RAG over MCP)
               actions.py (mock write tools → outbox/) · prompts.py · show.py (prints the trace)
session1/      01–06          session2/   07–12          examples/  01–12 single patterns
evals/         hiring_eval.jsonl                          FRAMEWORKS.md  which framework, and why
labs/          the lab and the capstone brief + rubric    tests/   offline end-to-end tests
outbox/        what the agent "did": interviews, emails, shortlists, audit log (git-ignored)
```
