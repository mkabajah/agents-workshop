"""
ACTION TOOLS · the things an agent can DO, not just read.

The RAG tools only read. A hiring agent also needs to act: book an interview, email a
candidate, save a shortlist. These are mocks: they write one JSON line to outbox/ instead of
touching a real calendar or mailbox, so you can test safely and look at what "happened".

Reads run freely. Writes (schedule_interview, send_email) are the ones we put behind
human approval in session1/04_middleware_hitl.py.

Note: the docstrings of the @tool functions below are read by the MODEL, not just by you: they are
the tool descriptions it uses to decide when and how to call each tool. Edit them with care.
"""
import datetime as dt
import json

from langchain.tools import tool

from workshop.config import OUTBOX

# The pretend calendar. A real version would ask Google Calendar or Outlook.
FREE_SLOTS = ["Sun 10:00", "Sun 14:00", "Mon 11:00", "Tue 09:30", "Wed 15:00"]


def _write(kind: str, record: dict) -> dict:
    """Append one record (plus a timestamp) to outbox/<kind>.jsonl and return it.
    JSON Lines = one JSON object per line: easy to append to and to read back."""
    OUTBOX.mkdir(exist_ok=True)
    record = {"at": dt.datetime.now().isoformat(timespec="seconds"), **record}
    with open(OUTBOX / f"{kind}.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record


@tool
def get_free_slots() -> list[str]:
    """The interviewer's free interview slots this week. Call it before schedule_interview."""
    return FREE_SLOTS


@tool
def schedule_interview(cv: str, job: str, slot: str) -> str:
    """Book an interview for ONE candidate. cv: the CV file name (e.g. 'cand-006.md').
    job: the job id (e.g. 'backend-engineer'). slot: one of get_free_slots().
    Writes to the shared calendar: needs a reason from the CVs first."""
    if slot not in FREE_SLOTS:
        return f"Error: '{slot}' is not free. Free slots: {', '.join(FREE_SLOTS)}"
    _write("interviews", {"cv": cv, "job": job, "slot": slot})
    return f"Booked {cv} for {job} on {slot}."


@tool
def send_email(cv: str, subject: str, body: str) -> str:
    """Email ONE candidate (the address is looked up from the CV file name, never typed by you).
    Use only after a human asked to contact this candidate. Keep it short and polite."""
    _write("emails", {"to_cv": cv, "subject": subject, "body": body})
    return f"Email to {cv} queued: '{subject}'."


@tool
def save_shortlist(job: str, candidates: list[str], reasons: list[str]) -> str:
    """Save the final shortlist for a job: CV file names and one evidence-based reason each."""
    _write("shortlists", {"job": job, "candidates": candidates, "reasons": reasons})
    return f"Saved a shortlist of {len(candidates)} for {job}."


# Ready-made lists so scripts can give an agent "just the reads" or "everything".
READ_TOOLS = [get_free_slots]
WRITE_TOOLS = [schedule_interview, send_email, save_shortlist]
ALL = READ_TOOLS + WRITE_TOOLS
