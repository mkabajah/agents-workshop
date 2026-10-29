# The Python you need to read this repo

Seven things appear in almost every script. Each one in plain words, with the line you'll see.

## 1. `async` / `await`: "wait without freezing"

```python
async def run(question):
    result = await agent.ainvoke({...})
```

An agent spends most of its time **waiting**: for the model's reply over the internet, for the MCP server
to answer through a pipe. `await` means "start this, and wait for it, but let Python do other work
meanwhile". A function that uses `await` must be declared `async def`.

Why we need it here: the MCP client library and LangChain's `ainvoke` are async, because waiting for
networks and pipes is what they do. (Strands and CrewAI hide it: `agent("…")` and `crew.kickoff()` are
normal calls that run the async parts for you.)

You start the whole thing once, at the bottom of the file:

```python
asyncio.run(run("Who knows Kafka?"))      # "run this async function until it's done"
```

Rule of thumb: inside `async def`, put `await` in front of anything that talks to a model or a server.
Forget it and you get a "coroutine was never awaited" warning instead of a result.

## 2. `async with` / `with`: "open, use, always close"

```python
async with rag_tools() as tools:          # starts the CV MCP server
    ...                                   # use the tools
                                          # ← here the server is stopped, even after an error
```

`with` is for things that must be cleaned up: files, connections, server processes. The cleanup happens
when the block ends, also if something crashes. `async with` is the same for async resources.

## 3. The `sys.path.insert(...)` line at the top

```python
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
```

The scripts live in folders (`session1/`, `frameworks/strands/`…) but import the shared `workshop`
package from the repo root. This line tells Python "also look for imports in the repo root".
`parents[1]` = one folder up, `parents[2]` = two folders up.

## 4. Decorators: `@tool`, `@wrap_tool_call`, `@start`…

```python
@tool
def get_free_slots() -> list[str]:
    """The interview slots that are still free this week."""
```

A decorator is a label that turns a normal function into something a framework understands. `@tool`
makes it a tool the model can call; the **docstring** (the text in `"""…"""`) becomes the tool's
description, which the model reads, so write it for the model.

## 5. Type hints: `cv: str`, `-> list[str]`

```python
def schedule_interview(cv: str, job: str, slot: str) -> str:
```

Python doesn't need them to run, but tool frameworks read them to build the JSON schema the model sees
("cv is a string"). `TypedDict` (LangGraph state) and Pydantic `BaseModel` (CrewAI Flow state) do the
same for a whole dictionary of fields.

## 6. f-strings: `f"Booked {cv} on {slot}"`

Text with values inserted: whatever is inside `{ }` is evaluated and pasted in.

## 7. `if __name__ == "__main__":`

The code under it runs only when you run the file directly (`python session1/02_first_agent.py`), not
when another file or a test imports it. That's how tests can reuse a script's functions.

---

Still stuck on a line? Paste it into Claude and ask "explain this line to a beginner".
