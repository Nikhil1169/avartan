import json

import openai

from tracing import NullSpan, to_json

BASE_MAX_TOKENS = 8000
LENGTH_RETRY_MAX_TOKENS = 12000
MAX_LENGTH_ATTEMPTS = 5
TOOL_OUTPUT_MAX_CHARS = 15000
REPEAT_CALL_THRESHOLD = 3
CONVERGENCE_NUDGE_FRACTION = 0.85
TOOL_ALIASES = {
    "read": "read_file",
    "write": "write_file",
    "edit": "edit_file",
    "shell": "bash",
    "search": "grep",
}


def _nudge_if_repeating(recent_calls, name, arguments, messages):
    recent_calls.append((name, arguments))
    del recent_calls[:-REPEAT_CALL_THRESHOLD]
    if len(recent_calls) == REPEAT_CALL_THRESHOLD and len(set(recent_calls)) == 1:
        recent_calls.clear()
        messages.append(
            {
                "role": "user",
                "content": (
                    "You've repeated the exact same tool call several times in a "
                    "row without new results. Stop and try a fundamentally "
                    "different approach."
                ),
            }
        )


def _compact_messages(messages):
    system = next((m for m in messages if m.get("role") == "system"), None)
    first_user = next((m for m in messages if m.get("role") == "user"), None)
    tool_call_count = sum(1 for m in messages if m.get("role") == "tool")

    compacted = []
    if system:
        compacted.append(system)
    if first_user:
        compacted.append(first_user)
    compacted.append(
        {
            "role": "user",
            "content": (
                "[Context condensed after repeated truncation — "
                f"{tool_call_count} tool result(s) from earlier in this conversation "
                "were dropped to free up space. Continue the task above; re-run any "
                "tool calls whose results you still need, and keep your response concise.]"
            ),
        }
    )
    return compacted


def run_turn(
    client,
    model,
    messages,
    tools_by_name,
    openai_tools,
    auto_approve=False,
    plan_mode=False,
    max_iterations=None,
    tracer=None,
    trace_id=None,
    parent_span_id="",
    agent_span=None,
):
    content = ""
    iterations = 0
    length_streak = 0
    traced_message_count = 0
    current_max_tokens = BASE_MAX_TOKENS
    recent_tool_calls = []
    convergence_nudged = False
    convergence_nudge_at = (
        int(max_iterations * CONVERGENCE_NUDGE_FRACTION) if max_iterations else None
    )

    while True:
        iterations += 1
        if max_iterations is not None and iterations > max_iterations:
            if agent_span is not None:
                agent_span.error = f"hit max_iterations ({max_iterations}) without finishing"
            return content

        if (
            convergence_nudge_at is not None
            and not convergence_nudged
            and iterations >= convergence_nudge_at
        ):
            convergence_nudged = True
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "You're nearing the iteration budget for this task. Stop "
                        "exploring alternative approaches and commit to finalizing "
                        "your best current solution now."
                    ),
                }
            )

        error = None
        current_tools = openai_tools
        llm_span = (
            tracer.span(f"response.{model}", "LLM", trace_id, parent_span_id)
            if tracer
            else NullSpan()
        )

        with llm_span:
            llm_span.attributes["llm.provider"] = "openrouter"
            llm_span.attributes["llm.model_name"] = model
            llm_span.attributes["llm.input_messages"] = to_json(
                messages[traced_message_count:]
            )
            traced_message_count = len(messages)

            for attempt in range(2):
                content = ""
                tool_calls = {}
                finish_reason = None

                try:
                    stream = client.chat.completions.create(
                        model=model,
                        messages=messages,
                        tools=current_tools,
                        stream=True,
                        max_tokens=current_max_tokens,
                    )

                    for chunk in stream:
                        choice = chunk.choices[0]
                        delta = choice.delta

                        print(delta.content or "", end="", flush=True)
                        content += delta.content or ""

                        if delta.tool_calls:
                            for tc in delta.tool_calls:
                                call = tool_calls.setdefault(
                                    tc.index, {"id": "", "name": "", "arguments": ""}
                                )
                                if tc.id:
                                    call["id"] += tc.id
                                if tc.function:
                                    if tc.function.name:
                                        call["name"] += tc.function.name
                                    if tc.function.arguments:
                                        call["arguments"] += tc.function.arguments

                        if choice.finish_reason:
                            finish_reason = choice.finish_reason

                    if finish_reason == "length":
                        print(
                            f"[finish_reason] {finish_reason!r} — "
                            "response was truncated, continuing"
                        )
                    else:
                        print(f"[finish_reason] {finish_reason!r}")

                    error = None
                    break
                except openai.APIError as e:
                    error = e
                    if "tool calls cutoff by max_tokens" in str(e).lower():
                        print(
                            "[retry] tool calls cutoff by max_tokens — "
                            "retrying with a reduced tool set"
                        )
                        current_tools = [
                            t
                            for t in openai_tools
                            if tools_by_name.get(t["function"]["name"])
                            and tools_by_name[t["function"]["name"]].is_read_only
                        ]
                        messages.append(
                            {
                                "role": "user",
                                "content": (
                                    "Your previous write_file call was too large and got "
                                    "cut off. Retry with a smaller payload — split it into "
                                    "multiple smaller edit_file calls, or reduce tokens per "
                                    "call."
                                ),
                            }
                        )
                except Exception as e:
                    error = e

            print()
            llm_span.attributes["llm.finish_reason"] = finish_reason

            if error is not None:
                llm_span.error = f"{type(error).__name__}: {error}"
                print(f"error: {type(error).__name__}: {error}")
                if agent_span is not None:
                    agent_span.error = llm_span.error
                return content

            if finish_reason == "tool_calls":
                length_streak = 0
                current_max_tokens = BASE_MAX_TOKENS
                assistant_message = {
                    "role": "assistant",
                    "content": content or None,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": call["arguments"],
                            },
                        }
                        for call in tool_calls.values()
                    ],
                }
                llm_span.attributes["llm.output_messages"] = to_json([assistant_message])
                messages.append(assistant_message)

                for call in tool_calls.values():
                    print(f"[tool_call] name={call['name']!r} arguments={call['arguments']!r}")

                    tool_span = (
                        tracer.span(f"function.{call['name']}", "TOOL", trace_id, parent_span_id)
                        if tracer
                        else NullSpan()
                    )
                    with tool_span:
                        tool_span.attributes["tool.name"] = call["name"]
                        tool_span.attributes["input.value"] = call["arguments"]

                        try:
                            args = json.loads(call["arguments"])
                        except json.JSONDecodeError as e:
                            print(
                                f"[tool_call] json.loads failed on: "
                                f"{call['arguments']!r} ({e})"
                            )
                            result = (
                                f"error: could not parse arguments as JSON: "
                                f"{call['arguments']!r}"
                            )
                            tool_span.attributes["output.value"] = result
                            tool_span.error = result
                            messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": call["id"],
                                    "content": result,
                                }
                            )
                            _nudge_if_repeating(
                                recent_tool_calls, call["name"], call["arguments"], messages
                            )
                            continue

                        tool_name = TOOL_ALIASES.get(call["name"], call["name"])
                        tool = tools_by_name.get(tool_name)

                        if tool is None:
                            result = (
                                f"error: no tool named '{call['name']}' — "
                                "check available tools and try again"
                            )
                        elif not tool.is_read_only and plan_mode:
                            result = "denied: plan mode active — read-only tools only"
                        elif not tool.is_read_only and not auto_approve:
                            print(f"Run {tool.name} with {args}? [y/n]")
                            if input("> ").strip().lower() != "y":
                                result = "user denied"
                            else:
                                result = tool.execute(args)
                        else:
                            result = tool.execute(args)

                        if isinstance(result, str) and len(result) > TOOL_OUTPUT_MAX_CHARS:
                            result = (
                                result[:TOOL_OUTPUT_MAX_CHARS]
                                + f"\n[Truncated: {len(result)} chars total. "
                                "Re-run with a narrower command or save full output "
                                "to a file.]"
                            )

                        tool_span.attributes["output.value"] = result
                        if isinstance(result, str) and result.startswith("error:"):
                            tool_span.error = result

                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": result,
                            }
                        )
                        _nudge_if_repeating(
                            recent_tool_calls, call["name"], call["arguments"], messages
                        )
                continue

            if finish_reason == "length":
                length_streak += 1
                assistant_message = {"role": "assistant", "content": content or None}
                llm_span.attributes["llm.output_messages"] = to_json([assistant_message])
                messages.append(assistant_message)

                if length_streak >= MAX_LENGTH_ATTEMPTS:
                    message = (
                        "model kept hitting the length limit after "
                        f"{length_streak} attempts, giving up"
                    )
                    print(f"error: {message}")
                    if agent_span is not None:
                        agent_span.error = message
                    return content

                if length_streak == 1:
                    current_max_tokens = LENGTH_RETRY_MAX_TOKENS
                    print(
                        f"[length-retry] bumping max_tokens to {current_max_tokens} and retrying"
                    )
                elif length_streak == 2:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your previous response was cut off for being too long. "
                                "Please continue, but be more concise."
                            ),
                        }
                    )
                    print("[length-retry] asked for a more concise response")
                elif length_streak == 3:
                    messages[:] = _compact_messages(messages)
                    traced_message_count = 0
                    current_max_tokens = BASE_MAX_TOKENS
                    print(
                        "[length-retry] summarized context and continuing with a fresh window"
                    )
                else:
                    current_max_tokens = LENGTH_RETRY_MAX_TOKENS
                    print(
                        f"[length-retry] bumping max_tokens to {current_max_tokens} "
                        "again and retrying"
                    )

                continue

            length_streak = 0
            current_max_tokens = BASE_MAX_TOKENS
            assistant_message = {"role": "assistant", "content": content}
            llm_span.attributes["llm.output_messages"] = to_json([assistant_message])
            messages.append(assistant_message)
            return content
