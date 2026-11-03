# How the LLM prompts were answered

Recorded on 2026-11-02, before any LLM reply existed. The user chose Claude rather than a manually relayed chat model.

- **Model:** Claude Opus 5.5, run as a fresh Claude Code subagent (model alias `opus`) for every prompt, repair prompts included. No subagent sees another prompt, an earlier conversation, or the orchestrating session's context. The project has no saved memory notes or `CLAUDE.md` that a subagent could load.
- **Procedure:** each subagent receives the instruction below, with only the two file paths changed. It reads its prompt file with one Read call and writes its reply with one Write call. The orchestrating session then submits that file unchanged with `scripts/nifty_llm_handoff.py submit --file`.
- **No editing or hints:** the orchestrating session never edits a reply or adds hints. It only starts each subagent, checks that the reply file exists, and submits it.

**Deviation from the protocol.** The protocol's relay instructions assumed a chat interface without tools. Subagents can technically use tools, so the instruction forbids every tool call except the one Read and the one Write. Each subagent is asked to confirm that it complied; compliance is self-reported.

**Observed compliance.** Every subagent reported using only the permitted calls, with one exception: at feedback step 12, the prompt (56,980 characters) exceeded the Read tool's default size limit. The first Read returned lines 1–287 of 354, so the subagent read the rest of the same file once more before writing its reply; it opened no other file. To check that no earlier subagent answered from a truncated prompt, every transcript was searched for two strings. These were the prompt's final task sentence, which is the file's last line, and the header of the most recent trial. Both appear in the batch transcript and in all feedback transcripts for steps 2–12.

Instruction given to every subagent:

> You are the language model being tested in a controlled research experiment. Your entire task is written in one file.
>
> 1. Read this file with one Read tool call: `<prompt path>`
> 2. Do exactly what that file asks, using only the information inside it. Do not open any other file, run commands, search, browse, or use any other tool.
> 3. Save your complete reply, exactly the text the file asks you to reply with and nothing else, with one Write tool call to: `<reply path>`
> 4. Finish with one short sentence confirming that you used only those two tool calls.

## Measured LLM cost

The totals below are the token counts and run durations that Claude Code reported for each subagent run. The report's *estimated tokens* column counts only prompt and reply characters ÷ 4, so it understates the cost of these agent runs roughly tenfold.

| Search | Subagent runs | Total tokens | Total run time |
|---|---|---|---|
| LLM, no feedback | 1 | 111,648 | 11.6 min |
| LLM + feedback | 12 (one per step) | 1,262,030 | 110.6 min |

Per feedback step: 72,302 to 175,431 tokens and 4.9 to 17.9 minutes. Tokens grew with the prompt, from 8,807 characters at step 1 to 56,980 at step 12. Training time is separate from these figures.

No reply needed a repair round, and every one of the 24 proposals was valid. The *relay time* column in the report is not a model-latency measure. The batch value also includes an overnight wait before the user chose Claude.
