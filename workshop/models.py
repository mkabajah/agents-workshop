"""
get_model() returns a LangChain chat model from MODEL="provider:model".

    anthropic:claude-sonnet-4-6   needs ANTHROPIC_API_KEY  (pip install langchain-anthropic)
    openai:gpt-4o-mini            needs OPENAI_API_KEY     (pip install langchain-openai)
    openai:<model> + OPENAI_BASE_URL   any OpenAI-compatible API: Groq, OpenRouter, Gemini, Cerebras…
    ollama:qwen2.5:7b             local, free              (pip install langchain-ollama)
    fake                          ScriptedChatModel below: no key, no network

The fake model is NOT an LLM. It follows simple rules (spot a skill in the question → call
find_candidates, …) so that every script and test runs offline, and so you can see the agent
loop's mechanics without paying for tokens. Its answers are stitched from the tool results.
"""
import json
import re
import uuid

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from workshop import config

SKILLS = ["power bi", "spring boot", "kubernetes", "terraform", "pytorch", "fastapi", "tableau", "docker",
          "kafka", "python", "java", "react", "sql", "aws", "n8n", "rag", "llm", "go", "node", "excel"]
CITIES = ["haifa", "tel aviv", "nazareth", "jerusalem", "beersheba", "ramallah", "umm al-fahm"]
CV_NAME = re.compile(r"[\w\-.]+\.(?:md|pdf|docx|txt)\b")      # the fake model only sees practice CVs
ONCE = {"list_jobs", "search_cvs", "get_cv", "get_free_slots", "schedule_interview", "send_email",
        "save_shortlist", "sourcer", "screener", "coordinator"}


def get_model(name: str | None = None) -> BaseChatModel:
    name = name or config.MODEL
    if name == "fake":
        return ScriptedChatModel()
    return init_chat_model(name)


def _job(q: str) -> str:
    if "backend" in q or "java" in q:
        return "backend-engineer"
    if "analyst" in q or " bi" in f" {q}":
        return "bi-analyst"
    return "ai-engineer" if " ai" in f" {q}" else "backend-engineer"


class ScriptedChatModel(BaseChatModel):
    """A rule-following stand-in for an LLM that can call tools."""

    tool_names: list[str] = []
    tool_args: dict[str, list[str]] = {}
    max_calls: int = 8

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        def name(t):
            if isinstance(t, dict):
                return t.get("name") or t.get("function", {}).get("name")
            return getattr(t, "name", None) or t.__name__
        def args(t):
            if isinstance(t, dict):
                return list((t.get("function", t).get("parameters") or {}).get("properties", {}))
            return list(getattr(t, "args", {}) or {})
        names = [name(t) for t in tools]
        return self.model_copy(update={"tool_names": names, "tool_args": {name(t): args(t) for t in tools}})

    # ------------------------------------------------------------------ one model call
    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _next(self, messages) -> AIMessage:
        last_h = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=-1)
        question = str(messages[last_h].content) if last_h >= 0 else str(messages[-1].content)
        turn = messages[last_h + 1:]
        done = [(tc["name"], tc["args"]) for m in turn if isinstance(m, AIMessage) for tc in m.tool_calls]
        outputs = " ".join(str(m.content) for m in turn if isinstance(m, ToolMessage))
        kind, name, args = decide(question, self.tool_names, self.tool_args, done, outputs, self.max_calls)
        if kind == "call":
            return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": f"call_{uuid.uuid4().hex[:8]}"}])
        return AIMessage(content=name)

def _plan(tool_names, tool_args, q, question, found):
    skills = [s for s in SKILLS if re.search(rf"(?<![a-z]){re.escape(s)}(?![a-z])", q)]
    city = next((c for c in CITIES if c in q), "")
    job = _job(q)
    if "job" in q or "shortlist" in q:
        yield "list_jobs", {}
    if "sourcer" in tool_names:
        yield "sourcer", {"task": question}
    if found:
        yield "screener", {"task": f"Check the evidence in {', '.join(found[:3])} for: {question}"}
    for s in skills[:3]:
        yield "find_candidates", {"skill": s, "city": city}
    if city and not skills:
        yield "find_candidates", {"skill": "", "city": city}
    if not skills and not city:
        yield "search_cvs", {"question": question}
    if found and any(w in q for w in ("evidence", "check", "shortlist", "strongest", "best", "screen")):
        yield "get_cv", {"file_name": found[0]}
    if found and any(w in q for w in ("interview", "schedule", "book")):
        yield "get_free_slots", {}
        yield "schedule_interview", {"cv": found[0], "job": job, "slot": "Mon 11:00"}
        yield "coordinator", {"task": f"Book an interview for {found[0]} ({job})."}
    if found and "email" in q:
        yield "send_email", {"cv": found[0], "subject": f"Your application: {job}",
                             "body": "Thank you for applying. We'd like to talk next week."}
    for name in tool_names:                               # any other tool named in the question
        verb = name.split("_")[0]                              # e.g. "remember" in remember_preference
        if name not in ONCE and name != "find_candidates" and len(verb) > 3 and verb in q:
            yield name, {a: question for a in tool_args.get(name, [])}
    if found and "save" in q:
        yield "save_shortlist", {"job": job, "candidates": found[:3], "reasons": ["matches the must-haves"] * len(found[:3])}


def decide(question, tool_names, tool_args, done, outputs, max_calls=8):
    """The fake model's whole brain, shared by the LangChain, Strands and CrewAI stand-ins.

    question  the user's latest message           tool_names/tool_args  the tools it may call
    done      [(tool, args)] already called       outputs               the tool results so far, as text
    Returns ("call", tool, args) or ("answer", text, None).
    """
    if not tool_names:                                             # e.g. the summariser: no tools
        return "answer", f"(fake model) Summary of the conversation so far: {question[:200]}", None
    done_keys = [(n, json.dumps(a, sort_keys=True)) for n, a in done]
    found = list(dict.fromkeys(n for n in CV_NAME.findall(question + " " + outputs) if "/" not in n))
    if len(done) < max_calls:
        for name, args in _plan(tool_names, tool_args, question.lower(), question, found):
            key = (name, json.dumps(args, sort_keys=True))
            if name in tool_names and key not in done_keys and not (name in ONCE and any(d[0] == name for d in done_keys)):
                return "call", name, args
    names = ", ".join(found[:5]) if found else "no matching CVs"
    return ("answer", f"(fake model) From the tool results: {names}. "
                      f"A real model would now reason over the evidence and write the answer.", None)
