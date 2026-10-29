# Session 1 lab · your first real hiring agent (30 minutes, pairs)

Start from `session1/04_middleware_hitl.py` (copy it to `session1/my_agent.py`).

## Steps

1. **Run it as is** with your MODEL. Ask: *"Find a Kafka candidate and book an interview for the backend job."*
   Approve once, reject once with a reason. Open `outbox/` and see what happened.
2. **Add one tool your recruiter needs**, in `workshop/actions.py` or in the RAG MCP server:
   e.g. `reject_candidate(cv, reason)`, `compare_candidates(cv_a, cv_b)`, `notes_for(cv)`.
   Write the docstring for the model: what it is for, what it is NOT for, the argument formats.
3. **Decide which tools need approval** and put them in `interrupt_on`.
4. **Add one guardrail** of your own with `@wrap_tool_call` (start from `session1/05_guardrail.py`),
   e.g. at most 3 interviews per run, no email on Saturdays, block `get_cv` on a CV not returned by a search.
5. **Test five prompts** and write down what happened:

| # | Prompt | Expected |
|---|---|---|
| 1 | Which candidates know Kubernetes? | `find_candidates`, 4 CVs |
| 2 | Who has the strongest Kafka experience? Check the evidence. | `find_candidates` then `get_cv` |
| 3 | Book an interview for the best Kafka candidate. | pauses for approval |
| 4 | And which of them lives in Haifa? *(same thread)* | uses the earlier answer |
| 5 | Email every candidate that we rejected them. | refuses or asks first |

## Done when

- [ ] answers come from your CVs, with file names
- [ ] every write pauses; approve, edit and reject all work
- [ ] a follow-up question uses the thread's memory
- [ ] your guardrail blocks something, and the agent explains why

## Stretch

- `--sqlite checkpoints.db`: press Ctrl+C at the approval prompt, then `--resume <thread_id>`.
- Change `MODEL` to a small local model (`ollama:qwen2.5:7b`) and run the same five prompts. What breaks first?
