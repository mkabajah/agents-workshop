# Session 2 capstone · a traced, evaluated agent system (35 minutes, teams of 2–4)

Build on your session 1 agent. Pick ONE:

| Option | The process | A good shape |
|---|---|---|
| A · Hiring pipeline | job post → shortlist → interviews booked | LangGraph workflow, an agent in `screen` (start: `session1/06`) |
| B · Hiring team | "fill the backend role" end to end | supervisor + sourcer, screener, coordinator (start: `session2/07`) |
| C · Your startup's process | support triage, sales research, investor update, ops report | your MCP server + the pattern that fits |
| D · Framework swap | the same team in Strands or CrewAI | `session2/08` or `09`, compared with `07` on the same evals |

## Must have

- [ ] your RAG MCP server as a tool source
- [ ] at least two agents, OR a graph with an agent node (and a sentence on why)
- [ ] human approval before every write
- [ ] limits: model calls, tool calls or handoffs
- [ ] a trace you can show (`session2/10_tracing.py` or LangSmith)
- [ ] 5+ eval examples in `evals/`, run once with `session2/11_evals.py`, pass rate written down

## Timeline

| Time | |
|---|---|
| 1:55 | build |
| 2:18 | code freeze |
| 2:20 | demos: 2 minutes each · input → trace → one approval → output |

## Rubric (10 points)

| Pts | Criterion | Full marks when |
|---|---|---|
| 2 | Real problem | A process a team actually runs, with a clear user |
| 2 | Right architecture | The pattern fits; no agent that could be a function |
| 2 | Works end to end | Live demo from input to output, through the RAG MCP server |
| 2 | Safety | Approval before writes, limits set, least-privilege tools, CV text treated as data |
| 1 | Observability | A trace explains one decision the system made |
| 1 | Evaluation | 5+ examples with a pass rate, before and after one change |
