# One hiring agent, three frameworks

The default is **LangChain + LangGraph** (`session1/`): it's what we teach in class. This folder has
the *same* steps written with **Strands Agents** (AWS) and **CrewAI**, on the *same* CV MCP server,
so you can compare them line by line.

```bash
pip install -r requirements-optional.txt          # strands-agents, crewai
python run.py 02 "Which candidates know Kafka?"                       # LangChain (default)
python run.py 02 "Which candidates know Kafka?" --framework strands
python run.py 02 "Which candidates know Kafka?" --framework crewai
python run.py --list
```

Every script also runs with `MODEL=fake`: no key, no network. That's a scripted stand-in, not an LLM.

## The same idea in each framework

| Step | LangChain / LangGraph (`session1/`) | Strands (`frameworks/strands/`) | CrewAI (`frameworks/crewai/`) |
|---|---|---|---|
| 02 · first agent | `create_agent(model, tools, system_prompt)` then `await agent.ainvoke(...)` | `Agent(model, tools, system_prompt)` then `agent("…")` | `Agent(role, goal, backstory, mcps)` + `Task` + `Crew(...).kickoff()` |
| MCP tools | `langchain-mcp-adapters` · `load_mcp_tools(session)` | `MCPClient(...)` · `list_tools_sync()` | `mcps=[MCPServerStdio(...)]` on the Agent |
| 03 · short-term memory | checkpointer + `thread_id` | `FileSessionManager(session_id=…)` | keep the conversation yourself and pass it in (or long-term `Crew(memory=True)`) |
| 04 · limits | `ModelCallLimitMiddleware`, `ToolCallLimitMiddleware` | a hook that counts tool calls and cancels past the budget | `Agent(max_iter=12)` |
| 04 · human approval | `HumanInTheLoopMiddleware` → `interrupt`, resume with `Command(resume=…)` | hook → `event.interrupt(...)`, resume with `agent([{"interruptResponse": …}])` | `@before_tool_call` hook asks, `return False` blocks |
| 05 · your own rule | `@wrap_tool_call` middleware | `BeforeToolCallEvent` hook, `event.cancel_tool = "why"` | `@before_tool_call` / `@after_tool_call` hooks |
| 06 · workflow | `StateGraph`: nodes, edges, `interrupt()` | plain Python around an `Agent` (graphs of agents: `GraphBuilder`, examples/09) | `Flow`: `@start`, `@listen`, `@router`, `self.state` |
| sync or async? | async (`await`), because MCP and many models are async | sync: `agent("…")` | sync: `crew.kickoff()` |

## What to notice

- **The tools never change.** Three frameworks, one MCP server. That's the whole point of MCP.
- **Approvals differ most.** LangGraph saves a paused run to a checkpointer, so a person can approve
  hours later, even after a restart. Strands interrupts are in memory by default (add a session manager
  to keep them). CrewAI's hook asks in the same process.
- **Workflows:** LangGraph and CrewAI Flows both give you a drawable graph with shared state.
  Strands prefers letting the model drive; for a fixed process, plain Python around one agent is fine.

## Which one to pick, and why we teach LangGraph first

Short answer: **teach the patterns deeply in one framework (LangChain + LangGraph), show Strands
as the second, mention CrewAI in one slide, and make Strands the main tool of the AWS session.**

### Why one framework deeply, not three equally

Students need to *build* in two sessions. Three APIs at equal depth means three sets of import
errors and no deep understanding. The ideas (the loop, tools, memory, approval, graphs, multi-agent
patterns, evals, injection) are the same everywhere; one framework is enough to learn them, and
the comparison slides show how they map to the others.

### Why LangChain + LangGraph as the main one, for now

- **The concepts are explicit**: state, nodes, edges, checkpointers, `interrupt()` and middleware are
  visible in the code, which is what you want when teaching.
- **Human approval and durable pauses are first-class**: the session 1 resume-after-restart demo is a
  few lines.
- **Most material and hiring demand**: the largest share of tutorials, examples and job posts
  students will meet today. (Check current numbers before quoting any.)
- **It does not lock anyone out of AWS**: AgentCore Runtime hosts LangGraph, Strands, CrewAI or plain
  Python. `examples/11_agentcore_langgraph.py` wraps the session agent in the AgentCore entrypoint.

### Why Strands as the second, and the AWS session's main tool

- Smallest surface: an agent is a model, tools and a prompt. Good for showing that the loop is the same.
- Built by AWS, first-class with Bedrock and AgentCore (deploy guides, OpenTelemetry tracing).
- Multi-agent built in: agents as tools, Swarm, Graph, Workflow, A2A.
- In the AWS session, switching to Strands costs little: the patterns, the MCP server and the evals carry over.

### CrewAI: one slide

Useful role/task vocabulary and Flows, but a third API at depth costs class time without a new idea.
The CrewAI versions of every Session 1 step are in `frameworks/crewai/`, and the team version in `session2/09`.

### How a team should pick (for the startups)

| If the team… | Pick |
|---|---|
| needs approvals, long-running flows, audit trails | LangGraph |
| lives on AWS / Bedrock, wants the simplest code and AgentCore deploys | Strands |
| builds content or research pipelines with clear roles and deliverables | CrewAI (Flows for production) |
| is on OpenAI or Google stacks | OpenAI Agents SDK / Google ADK |
| isn't sure | the one the team already knows; MCP tools and evals keep the choice reversible |

### Suggested plan

| Session | Main framework | Others |
|---|---|---|
| 1 · one agent | LangChain + LangGraph | — |
| 2 · a team | LangChain + LangGraph | Strands (15 min), CrewAI (1 slide) |
| 3 · AWS | Strands on AgentCore | the LangGraph agent on AgentCore (examples/11) to show portability |
