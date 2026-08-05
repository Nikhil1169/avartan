import os
import re
import subprocess

from firecrawl import Firecrawl

from agent import run_turn


class Tool:
    name: str
    description: str
    parameters: dict
    is_read_only: bool = False

    def execute(self, args: dict) -> str:
        raise NotImplementedError


def to_openai_tool(tool: Tool) -> dict:
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file from disk."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read."},
        },
        "required": ["path"],
    }
    is_read_only = True

    def execute(self, args: dict) -> str:
        with open(args["path"]) as f:
            return f.read()


class WriteFileTool(Tool):
    name = "write_file"
    description = "Write content to a file on disk, overwriting it if it exists."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to write."},
            "content": {"type": "string", "description": "Content to write to the file."},
        },
        "required": ["path", "content"],
    }
    is_read_only = False

    def execute(self, args: dict) -> str:
        with open(args["path"], "w") as f:
            f.write(args["content"])
        return f"Wrote {len(args['content'])} bytes to {args['path']}"


class EditFileTool(Tool):
    name = "edit_file"
    description = "Replace an exact occurrence of old_string with new_string in a file."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to edit."},
            "old_string": {"type": "string", "description": "Exact string to replace."},
            "new_string": {"type": "string", "description": "String to replace it with."},
        },
        "required": ["path", "old_string", "new_string"],
    }
    is_read_only = False

    def execute(self, args: dict) -> str:
        with open(args["path"]) as f:
            content = f.read()

        if args["old_string"] not in content:
            return f"Error: old_string not found in {args['path']}"

        content = content.replace(args["old_string"], args["new_string"], 1)
        with open(args["path"], "w") as f:
            f.write(content)
        return f"Edited {args['path']}"


class GrepTool(Tool):
    name = "grep"
    description = "Search files under a directory for lines matching a regex pattern."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Regex pattern to search for."},
            "path": {"type": "string", "description": "Directory to search in. Defaults to '.'."},
        },
        "required": ["pattern"],
    }
    is_read_only = True

    def execute(self, args: dict) -> str:
        regex = re.compile(args["pattern"])
        root = args.get("path", ".")
        matches = []

        for dirpath, _, filenames in os.walk(root):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                with open(filepath, errors="ignore") as f:
                    for lineno, line in enumerate(f, start=1):
                        if regex.search(line):
                            matches.append(f"{filepath}:{lineno}:{line.rstrip()}")

        return "\n".join(matches) if matches else "No matches found."


class BashTool(Tool):
    name = "bash"
    description = "Run a shell command and return its stdout and stderr."
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command to run."},
        },
        "required": ["command"],
    }
    is_read_only = False

    def execute(self, args: dict) -> str:
        result = subprocess.run(
            args["command"], shell=True, capture_output=True, text=True
        )
        return result.stdout + result.stderr


class TodoWriteTool(Tool):
    name = "todo_write"
    description = "Write the current task todo list, replacing any previous one."
    parameters = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "done"],
                        },
                    },
                    "required": ["content", "status"],
                },
            },
        },
        "required": ["items"],
    }
    is_read_only = True

    def __init__(self):
        self.todos = []

    def execute(self, args: dict) -> str:
        self.todos = args["items"]
        marks = {"pending": " ", "in_progress": "~", "done": "x"}
        return "\n".join(
            f"[{marks[item['status']]}] {item['content']}" for item in self.todos
        )


class WebSearchTool(Tool):
    name = "web_search"
    description = "Search the web and return the top results."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query."},
        },
        "required": ["query"],
    }
    is_read_only = True

    def __init__(self):
        self.firecrawl = Firecrawl(api_key=os.environ["FIRECRAWL_API_KEY"])

    def execute(self, args: dict) -> str:
        result = self.firecrawl.search(args["query"], limit=5)
        return "\n\n".join(
            f"{item.title}\n{item.url}\n{item.description}" for item in result.web
        )


class SpawnAgentTool(Tool):
    name = "task"
    description = "Spawn a fresh sub-agent to carry out a self-contained task and return its final answer."
    parameters = {
        "type": "object",
        "properties": {
            "description": {"type": "string", "description": "Short description of the task."},
            "prompt": {"type": "string", "description": "The task for the sub-agent to perform."},
        },
        "required": ["description", "prompt"],
    }
    is_read_only = False

    def __init__(self, client, model, system_message, tools):
        self.client = client
        self.model = model
        self.system_message = system_message
        self.tools_by_name = {tool.name: tool for tool in tools}
        self.openai_tools = [to_openai_tool(tool) for tool in tools]

    def execute(self, args: dict) -> str:
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": args["prompt"]},
        ]
        return run_turn(
            self.client,
            self.model,
            messages,
            self.tools_by_name,
            self.openai_tools,
            auto_approve=True,
        )
