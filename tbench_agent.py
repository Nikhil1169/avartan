import shlex
from typing import override

from harbor.agents.installed.base import BaseInstalledAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext


class AvartanAgent(BaseInstalledAgent):
    @staticmethod
    @override
    def name() -> str:
        return "avartan"

    @override
    async def install(self, environment: BaseEnvironment) -> None:
        await self.exec_as_root(
            environment,
            command=(
                "if ! command -v python3 >/dev/null 2>&1 || "
                "! command -v pip3 >/dev/null 2>&1 || "
                "! command -v git >/dev/null 2>&1; then "
                "apt-get update && apt-get install -y python3 python3-pip git; "
                "fi"
            ),
            env={"DEBIAN_FRONTEND": "noninteractive"},
        )
        await self.exec_as_agent(
            environment,
            command=(
                "pip install --break-system-packages "
                "git+https://github.com/Nikhil1169/avartan.git@main"
            ),
        )

    @override
    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        env = {
            "OPENROUTER_API_KEY": self._get_env("OPENROUTER_API_KEY"),
            "FIRECRAWL_API_KEY": self._get_env("FIRECRAWL_API_KEY"),
        }
        escaped_instruction = shlex.quote(instruction)
        await self.exec_as_agent(
            environment,
            command=(
                f"avartan --task {escaped_instruction} --max-iterations 30 "
                "2>&1 | stdbuf -oL tee /logs/agent/avartan.txt"
            ),
            env=env,
        )
