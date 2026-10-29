# Agent examples · one pattern per file, all on your CV MCP server

The `session1/` and `session2/` scripts follow the workshop's story. These are a **catalogue**:
one pattern per file, short, runnable on its own. They all use the same RAG MCP server, models
and helpers, so the only thing that changes is the pattern.

| # | File | Pattern | Framework | `MODEL=fake`? |
|---|---|---|---|:-:|
| 01 | `01_tool_calling_basics.py` | One call, see the tool request (no loop) | LangChain | ✓ |
| 02 | `02_structured_output.py` | Agent returns a typed object (Pydantic) | LangChain | real model |
| 03 | `03_streaming.py` | Stream each step as it happens | LangChain | ✓ |
| 04 | `04_router.py` | Classify once, then a fixed branch | LangGraph | ✓ |
| 05 | `05_plan_and_execute.py` | Planner writes steps, executor runs them | LangChain | ✓ |
| 06 | `06_reflection.py` | Writer + critic loop (evaluator–optimizer) | LangGraph | ✓ |
| 07 | `07_parallel_fanout.py` | One branch per must-have in parallel, then merge | LangGraph `Send` | ✓ |
| 08 | `08_long_term_memory.py` | Preferences that survive across threads | LangGraph store | ✓ |
| 09 | `09_strands_graph.py` | Agents in a drawn graph with a condition | Strands | `--dry` |
| 10 | `10_crewai_flow.py` | Steps in code, a crew inside one step | CrewAI Flow | `--dry` |
| 11 | `11_agentcore_strands.py` | A Strands agent behind AgentCore Runtime | AWS + Strands | real model |
| 12 | `12_agentcore_langgraph.py` | The same entrypoint around LangGraph | AWS + LangGraph | ✓ |

Plus, in the sessions: the loop by hand (`session1/01`), memory (`03`), human approval and
resume (`04`), guardrails (`05`), a screening workflow (`06`), a supervisor team (`session2/07`),
Strands agents-as-tools and Swarm (`08`), a CrewAI crew (`09`), tracing (`10`), evals (`11`),
prompt injection (`12`).

## Which pattern when?

```
Is there one question → one action?            → 01 tool calling
Do you know the steps?                         → 04 router · 06 reflection · 07 fan-out · session1/06
Unknown steps, one domain?                     → an agent (session1/02–05) · 03 streaming for UX · 02 for typed output
Long task worth reviewing first?               → 05 plan and execute
Too many tools or too much context?            → a team (session2/07–09)
Needs to remember across conversations?        → 08 long-term memory
Going to production on AWS?                    → 11 / 12 AgentCore Runtime (any framework)
```

## Run them all offline

```bash
RAG_EMBEDDER=tfidf pytest -q tests/test_examples.py
```
