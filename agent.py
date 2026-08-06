import json

import openai


def run_turn(
    client,
    model,
    messages,
    tools_by_name,
    openai_tools,
    auto_approve=False,
    plan_mode=False,
    max_iterations=None,
):
    content = ""
    iterations = 0
    length_streak = 0

    while True:
        iterations += 1
        if max_iterations is not None and iterations > max_iterations:
            return content

        error = None

        for attempt in range(2):
            content = ""
            tool_calls = {}
            finish_reason = None

            try:
                stream = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=openai_tools,
                    stream=True,
                    max_tokens=8000,
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
            except Exception as e:
                error = e

        print()

        if error is not None:
            print(f"error: {type(error).__name__}: {error}")
            return content

        if finish_reason == "tool_calls":
            length_streak = 0
            messages.append(
                {
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
            )

            for call in tool_calls.values():
                print(f"[tool_call] name={call['name']!r} arguments={call['arguments']!r}")

                try:
                    args = json.loads(call["arguments"])
                except json.JSONDecodeError as e:
                    print(f"[tool_call] json.loads failed on: {call['arguments']!r} ({e})")
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": (
                                f"error: could not parse arguments as JSON: "
                                f"{call['arguments']!r}"
                            ),
                        }
                    )
                    continue

                tool = tools_by_name.get(call["name"])

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

                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": result,
                    }
                )
            continue

        if finish_reason == "length":
            length_streak += 1
            messages.append({"role": "assistant", "content": content or None})
            if length_streak >= 3:
                print(
                    "error: model kept hitting the length limit after "
                    f"{length_streak} attempts, giving up"
                )
                return content
            continue

        messages.append({"role": "assistant", "content": content})
        return content
