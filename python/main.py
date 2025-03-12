import asyncio
import json
from typing import Callable, Sequence

from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.base import ChatAgent, Team, TerminatedException, TerminationCondition
from autogen_agentchat.conditions import ExternalTermination, MaxMessageTermination, TextMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient

from kagent.tools import AgentTool
from kagent.tools.docs import Config, QueryTool
from kagent.tools.helm import GetRelease, ListReleases, Uninstall, Upgrade
from kagent.tools.k8s import ApplyManifest, GetAvailableAPIResources, GetResources

k8s_agent = AssistantAgent(
    name="k8s_agent",
    description="A Kubernetes agent that can manage Kubernetes resources.",
    system_message="""
    You are a Kubernetes agent that can manage Kubernetes resources.
    You can use the following tools to manage Kubernetes resources:
    - GetAvailableAPIResources
    - GetResources
    - ApplyManifest

    If you need more information about a specific Kubernetes resource, you can use the QueryTool to search the Kubernetes documentation.
    """,
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
    tools=[
        GetAvailableAPIResources(),
        GetResources(),
        ApplyManifest(),
        QueryTool(Config(docs_base_path="", docs_download_url="https://doc-sqlite-db.s3.sa-east-1.amazonaws.com")),
    ],
    tool_call_summary_format="\nTool: \n{tool_name}\n\nArguments:\n\n{arguments}\n\nResult: \n{result}\n",
)

helm_agent = AssistantAgent(
    name="helm_agent",
    description="A Kubernetes agent that can manage Helm releases.",
    system_message="""
    You are a Kubernetes agent that can manage Helm releases.
    You can use the following tools to manage Helm releases:
    - ListReleases
    - GetRelease
    - Upgrade
    - Uninstall

    If you need more information about helm, you can use the QueryTool to search the helm documentation.


    For all other requests, you should delegate to the k8s_agent tool.
    """,
    model_client=OpenAIChatCompletionClient(model="gpt-4o-mini"),
    tools=[
        ListReleases(),
        GetRelease(),
        Upgrade(),
        Uninstall(),
        AgentTool(k8s_agent),
    ],
)

team = RoundRobinGroupChat(
    participants=[helm_agent], termination_condition=TextMessageTermination(source=helm_agent.name)
)


# with open("/home/eitanyarmush/src/kagent-dev/kagent/python/agents/helm_agent.json", "r") as f:
#     team_json = f.read()

# team = Team.load_component(json.loads(team_json))

task = "List all gateways in the cluster."
# Use asyncio.run(...) if you are running this in a script.
asyncio.run(Console(team.run_stream(task=task)))
