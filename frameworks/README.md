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
| 03 · memory | checkpointer + `thread_id` | `FileSessionManager(session_id=…)` | keep the conversation yourself and pass it in (or long-term `Crew(memory=True)`) |
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

Which to pick, and why we teach LangGraph first: [`../FRAMEWORKS.md`](../FRAMEWORKS.md).
