# Which framework should we teach, and which should a team pick?

Short answer: **teach the patterns deeply in one framework (LangChain + LangGraph), show Strands
as the second, mention CrewAI in one slide, and make Strands the main tool of the AWS session.**

## Why one framework deeply, not three equally

Students need to *build* in two sessions. Three APIs at equal depth means three sets of import
errors and no deep understanding. The ideas (the loop, tools, memory, approval, graphs, multi-agent
patterns, evals, injection) are the same everywhere; one framework is enough to learn them, and
the comparison slides show how they map to the others.

## Why LangChain + LangGraph as the main one, for now

- **The concepts are explicit**: state, nodes, edges, checkpointers, `interrupt()` and middleware are
  visible in the code, which is what you want when teaching.
- **Human approval and durable pauses are first-class**: the session 1 resume-after-restart demo is a
  few lines.
- **Most material and hiring demand**: the largest share of tutorials, examples and job posts
  students will meet today. (Check current numbers before quoting any.)
- **It does not lock anyone out of AWS**: AgentCore Runtime hosts LangGraph, Strands, CrewAI or plain
  Python. `examples/12_agentcore_langgraph.py` wraps the session agent in the AgentCore entrypoint.

## Why Strands as the second, and the AWS session's main tool

- Smallest surface: an agent is a model, tools and a prompt. Good for showing that the loop is the same.
- Built by AWS, first-class with Bedrock and AgentCore (deploy guides, OpenTelemetry tracing).
- Multi-agent built in: agents as tools, Swarm, Graph, Workflow, A2A.
- In the AWS session, switching to Strands costs little: the patterns, the MCP server and the evals carry over.

## CrewAI: one slide

Useful role/task vocabulary and Flows, but a third API at depth costs class time without a new idea.
Keep `session2/09` and `examples/10` in the repo for teams that want it.

## How a team should pick (for the startups)

| If the team… | Pick |
|---|---|
| needs approvals, long-running flows, audit trails | LangGraph |
| lives on AWS / Bedrock, wants the simplest code and AgentCore deploys | Strands |
| builds content or research pipelines with clear roles and deliverables | CrewAI (Flows for production) |
| is on OpenAI or Google stacks | OpenAI Agents SDK / Google ADK |
| isn't sure | the one the team already knows; MCP tools and evals keep the choice reversible |

## Suggested plan

| Session | Main framework | Others |
|---|---|---|
| 1 · one agent | LangChain + LangGraph | — |
| 2 · a team | LangChain + LangGraph | Strands (15 min), CrewAI (1 slide) |
| 3 · AWS | Strands on AgentCore | the LangGraph agent on AgentCore (examples/12) to show portability |
