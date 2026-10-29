"""
PRINTING · show what the agent did, step by step: the best way to understand (and debug) an agent.

Used by the LangChain scripts after a run: trace(result["messages"]) prints one line per message,
banner("title") prints a coloured separator. Nothing here changes what the agent does.
"""
import json

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

# ANSI escape codes: special characters that most terminals show as colours (END resets the colour).
BLUE, ORANGE, GREEN, GREY, BOLD, END = "\033[94m", "\033[33m", "\033[92m", "\033[90m", "\033[1m", "\033[0m"


def short(x, n=300):
    """Any value as one line of text, cut to n characters, so long tool results don't flood the screen."""
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    s = " ".join(s.split())
    return s if len(s) <= n else s[:n] + " …"


def content_text(m) -> str:
    """The plain text of a message. Some models return a list of content blocks instead of a string."""
    if isinstance(m.content, str):
        return m.content
    return " ".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in m.content)


def trace(messages, start=0):
    """One line per step: who spoke, which tool was called with which arguments, what came back."""
    for m in messages[start:]:
        if isinstance(m, SystemMessage):
            continue
        if isinstance(m, HumanMessage):
            print(f"\n{BOLD}👤 user{END}   {short(content_text(m), 500)}")
        elif isinstance(m, AIMessage) and m.tool_calls:
            for tc in m.tool_calls:
                print(f"{ORANGE}🤖 calls{END}  {tc['name']}({short(tc['args'], 200)})")
        elif isinstance(m, AIMessage):
            print(f"{GREEN}🤖 answer{END} {content_text(m)}")
        elif isinstance(m, ToolMessage):
            print(f"{GREY}🔧 {m.name} →{END} {GREY}{short(content_text(m))}{END}")


def banner(title):
    """Print a title between two coloured lines, to separate runs in the output."""
    print(f"\n{BLUE}{BOLD}{'━' * 70}\n{title}\n{'━' * 70}{END}")
