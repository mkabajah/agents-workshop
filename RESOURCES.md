# Reuse before you build

Things worth stealing from, sorted by what you need. Checked October 2026; star counts and APIs move fast, so
read each README before you depend on it.

## MCP servers: tools your agent can use today

| What | Where | Use it for |
|---|---|---|
| Official MCP Registry | https://registry.modelcontextprotocol.io | The public catalogue of MCP servers: search before you write one |
| Reference servers | https://github.com/modelcontextprotocol/servers | Filesystem, fetch, git, memory, time… and how a good server is written |
| MCP Inspector | https://github.com/modelcontextprotocol/inspector | `npx @modelcontextprotocol/inspector python -m cv_server.server`: click through your tools before an agent does |
| GitHub MCP server | https://github.com/github/github-mcp-server | Issues, PRs, code search for a dev-team agent |
| Playwright MCP | https://github.com/microsoft/playwright-mcp | A browser as a tool (forms, scraping a careers page) |
| AWS MCP servers | https://github.com/awslabs/mcp | AWS docs, CDK, Bedrock KB retrieval, cost… (useful for the AWS session) |
| Community list | https://github.com/punkpeye/awesome-mcp-servers | Hundreds more; quality varies |

**Safety:** an MCP server runs code with *your* permissions and its tool descriptions go straight into the
model's context. Use servers from publishers you trust, pin the version, and give an agent only the servers
(and the tools) it needs: the same least-privilege rule as session 2.

## Agent Skills: packaged know-how

| What | Where |
|---|---|
| Anthropic's skills repo (docx, xlsx, pdf, pptx, skill-creator, …) | https://github.com/anthropics/skills |
| How a skill works | A folder with a `SKILL.md` (name + when to use + steps) and optional scripts. The agent reads only the description until it needs the skill, so 50 skills cost almost no context |

You wrote skills for the CV tools in the RAG course (`.claude/skills/`): same idea. A skill is *instructions*,
an MCP server is *tools*; good agents use both.

## Reference agents and samples

| Repo | Why look |
|---|---|
| https://github.com/langchain-ai/deepagents | Planning + sub-agents + a file system on LangGraph: the "Claude Code" shape, open source |
| https://github.com/langchain-ai/open_deep_research | A full research agent (supervisor + researchers) with evals |
| https://github.com/langchain-ai/openevals | Ready-made evaluators (LLM-as-judge, trajectory match) |
| https://github.com/strands-agents/samples | Strands agents, multi-agent patterns, deployment |
| https://github.com/awslabs/amazon-bedrock-agentcore-samples | AgentCore Runtime, Gateway, Memory, Identity with LangGraph, Strands, CrewAI |
| https://github.com/crewAIInc/crewAI-examples | Crews and Flows for many business cases |
| https://github.com/microsoft/agent-framework | Microsoft's successor to AutoGen + Semantic Kernel (Python and .NET) |

## Read

- Anthropic, *Building effective agents*: https://www.anthropic.com/engineering/building-effective-agents (workflows vs agents, the patterns in session 2)
- HumanLayer, *12-Factor Agents*: https://github.com/humanlayer/12-factor-agents (own your prompts, own your control flow, small focused agents)
- LangChain Academy, free LangGraph course: https://academy.langchain.com
- AgentCore: any framework on AgentCore Runtime: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/using-any-agent-framework.html
