import json


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

    while True:
        iterations += 1
        if max_iterations is not None and iterations > max_iterations:
            return content

        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=openai_tools,
            stream=True,
        )

        content = ""
        tool_calls = {}
        finish_reason = None

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

        print()

        if finish_reason == "tool_calls":
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
                args = json.loads(call["arguments"])
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

        messages.append({"role": "assistant", "content": content})
        return content
