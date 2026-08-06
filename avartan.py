import argparse
import os
import platform
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from agent import run_turn
from tracing import tracer_from_env, to_json
from tools import (
    ReadFileTool,
    WriteFileTool,
    EditFileTool,
    GrepTool,
    BashTool,
    TodoWriteTool,
    WebSearchTool,
    SpawnAgentTool,
    to_openai_tool,
)

load_dotenv(dotenv_path=Path(__file__).parent / ".env")


def main():
    parser = argparse.ArgumentParser(prog="avartan")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--task-file", type=str, default=None)
    parser.add_argument("--max-iterations", type=int, default=50)
    args = parser.parse_args()

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    MODEL = "poolside/laguna-s-2.1:free"

    identity = (
        "You are avartan, a terminal coding agent. Be concise. Prefer tools over guessing. "
        "Use todo_write to plan any task with more than a couple of steps. "
        "If plan mode is active, do not attempt edits — research and propose a plan using the todo tool instead."
    )

    environment = (
        "Environment:\n"
        f"cwd: {os.getcwd()}\n"
        f"OS: {platform.system()}\n"
        f"Files: {', '.join(os.listdir('.'))}"
    )

    system_message = identity + "\n\n" + environment

    if os.path.exists("AVARTAN.md"):
        with open("AVARTAN.md") as f:
            system_message += "\n\n# Project instructions\n" + f.read()

    base_tools = [
        ReadFileTool(),
        WriteFileTool(),
        EditFileTool(),
        GrepTool(),
        BashTool(),
        TodoWriteTool(),
        WebSearchTool(),
    ]

    spawn_tool = SpawnAgentTool(
        client=client,
        model=MODEL,
        system_message=system_message,
        tools=base_tools,
    )

    tools = base_tools + [spawn_tool]
    tools_by_name = {tool.name: tool for tool in tools}
    openai_tools = [to_openai_tool(tool) for tool in tools]

    history = [{"role": "system", "content": system_message}]

    if args.task is not None or args.task_file is not None:
        if args.task is not None:
            task_text = args.task
        else:
            with open(args.task_file) as f:
                task_text = f.read()

        history.append({"role": "user", "content": task_text})

        tracer = tracer_from_env(project_id="avartan", service_name="avartan")
        trace_id = tracer.new_id()
        with tracer.span("agent.avartan", "AGENT", trace_id) as agent_span:
            agent_span.attributes["agent.name"] = "avartan"
            agent_span.attributes["agent.tools"] = to_json([t.name for t in tools])
            run_turn(
                client,
                MODEL,
                history,
                tools_by_name,
                openai_tools,
                auto_approve=True,
                max_iterations=args.max_iterations,
                tracer=tracer,
                trace_id=trace_id,
                parent_span_id=agent_span.span_id,
                agent_span=agent_span,
            )
        tracer.close()
        return

    plan_mode = False

    try:
        while True:
            user_input = input("> ")

            if user_input == "/plan":
                plan_mode = not plan_mode
                print(f"plan mode {'on' if plan_mode else 'off'}")
                continue

            history.append({"role": "user", "content": user_input})
            run_turn(client, MODEL, history, tools_by_name, openai_tools, plan_mode=plan_mode)
    except KeyboardInterrupt:
        print("\nExiting.")
        raise SystemExit(0)


if __name__ == "__main__":
    main()
