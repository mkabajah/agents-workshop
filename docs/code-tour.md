# A tour of the code

How the pieces fit together, in the order the computer runs them. Read this once and every
script in the repo will look familiar.

## One run, file by file

What happens when you type:

```bash
python session1/02_first_agent.py "Which candidates know Kafka?"
```

1. **`session1/02_first_agent.py`**: the script you run. It is short on purpose: it builds an agent
   from three ingredients (a model, tools, a system prompt) and asks it your question.
2. **`workshop/config.py`**: imported first. It reads your `.env` file (which model, which CV
   server, which CV folder) so every script sees the same settings.
3. **`workshop/models.py`**: `get_model()` turns `MODEL=anthropic:…` (or `openai:…`, `ollama:…`)
   into a chat model. With `MODEL=fake` you get a scripted stand-in instead (see below).
4. **`workshop/mcp_tools.py`**: `async with rag_tools() as tools` starts the CV server as a child
   process and asks it for its tools (MCP `tools/list`). They come back as normal LangChain tools.
5. **`cv_server/server.py`**: the other side of MCP. A separate process that reads the CVs in
   `data/cvs/`, builds a small keyword index, and answers four tools: `search_cvs`, `get_cv`,
   `find_candidates`, `list_jobs`. Your agent never imports it; it only talks to it.
6. **The loop** (inside `create_agent`): the model reads the messages and either answers or asks
   for a tool; LangChain calls the tool over MCP (`tools/call`) and adds the result; repeat.
7. **`workshop/actions.py`**: the "write" tools (book an interview, send an email, save a shortlist).
   They are pretend: each one adds a line to a file in `outbox/`, so you can see what "happened".
8. **`workshop/show.py`**: prints the run, one line per step (user, calls, tool result, answer).

```
your question
   │
   ▼
02_first_agent.py ──► config (.env) ──► models.get_model()
   │
   ▼
create_agent loop ◄──── tools/list, tools/call ────► cv_server/server.py ──► data/cvs/
   │
   ├──► actions.py ──► outbox/*.jsonl      (only the write tools)
   ▼
show.trace() prints the answer
```

## What each `workshop/` file is for

| File | In one line |
|---|---|
| `config.py` | settings from `.env`: paths, `MODEL`, which CV server |
| `models.py` | `get_model()`, plus the offline fake model and its `decide()` |
| `mcp_tools.py` | connect to the CV server and get its tools (LangChain) |
| `actions.py` | the pretend write tools that log to `outbox/` |
| `prompts.py` | every system prompt, in one place |
| `jobs.py` | read a job post and its must-have skills |
| `show.py` | print a run step by step |
| `strands_kit.py` | the same model / CV server / write tools, for Strands |
| `crewai_kit.py` | the same model / CV server / write tools, for CrewAI |

Rule of thumb: if two scripts need the same thing, it lives in `workshop/`, once.

## The same pieces in three frameworks

`frameworks/strands/` and `frameworks/crewai/` repeat the Session 1 steps with other frameworks.
They do not have their own copies of anything important:

- the **CV server** is the same `cv_server/` (via `strands_kit.cv_client()` / `crewai_kit.cv_server()`);
- the **write tools** wrap the same `workshop/actions.py`, so bookings land in the same `outbox/`;
- the **prompts** come from `workshop/prompts.py`, and the workflows use `workshop/jobs.py`;
- `python run.py 02 --framework strands` just finds the right file and runs it.

So when you compare `session1/04_middleware_hitl.py` with `frameworks/strands/04_approval.py`,
the only differences you see are the framework's own way of doing it.

## How the fake model works

`MODEL=fake` lets every script and test run with no API key and no internet. It is not an LLM.
Its whole "brain" is `decide()` in `workshop/models.py`:

1. It reads your question and the tool results so far.
2. It walks a fixed list of rules in `_plan()`: a skill in the question → `find_candidates`;
   "evidence" or "check" → `get_cv`; "book" or "interview" → `get_free_slots`, then
   `schedule_interview`; and so on.
3. It calls the first tool from that list it hasn't called yet. When nothing is left, it answers
   with the CV names it saw.

The Strands and CrewAI fakes (`FakeStrandsModel`, `FakeCrewLLM`) call the same `decide()` and only
translate its choice into their framework's format, so all three behave the same offline.

## Where to start reading

1. `session1/01_loop_by_hand.py`: the whole agent loop in about 20 lines, no framework.
2. `session1/02_first_agent.py`: the same thing with `create_agent`.
3. `workshop/mcp_tools.py` and `cv_server/server.py`: both ends of MCP.
4. Then follow the session scripts in order, 03 → 06, and compare one of them with its version
   in `frameworks/`.

New to `async`, `await` or `async with`? Read [`python-notes.md`](python-notes.md) first.
