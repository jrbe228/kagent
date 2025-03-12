import json
import logging
from typing import Annotated, List

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.base._task import TaskResult
from autogen_agentchat.conditions import TextMessageTermination
from autogen_agentchat.messages import (
    AgentEvent,
    ChatMessage,
    HandoffMessage,
    MemoryQueryEvent,
    ModelClientStreamingChunkEvent,
    MultiModalMessage,
    StopMessage,
    TextMessage,
    ThoughtEvent,
    ToolCallExecutionEvent,
    ToolCallRequestEvent,
    ToolCallSummaryMessage,
    UserInputRequestedEvent,
)
from autogen_agentchat.state import TeamState
from autogen_agentchat.teams import RoundRobinGroupChat, TeamRuntimeContext
from autogen_agentchat.teams._group_chat._events import GroupChatMessage
from autogen_core import CancellationToken, Component, ComponentModel
from autogen_core.tools import BaseTool
from pydantic import BaseModel
from typing_extensions import Self


class AssistantAgentState(BaseModel):
    """State for the AssistantAgent."""

    team: TeamState


class AgentToolConfig(BaseModel):
    """Configuration for the AgentTool."""

    agent: ComponentModel


class AgentToolInput(BaseModel):
    """Input for the AgentTool."""

    task: Annotated[str, "The task to be executed by the agent."]


class AgentTool(BaseTool, Component[AgentToolConfig]):
    """AgentTool that can be used to create a tool for an agent."""

    component_type = "tool"
    component_config_schema = AgentToolConfig
    component_provider_override = "kagent.tools.AgentTool"

    def __init__(self, agent: AssistantAgent) -> None:
        self._agent = agent

        super().__init__(
            args_type=AgentToolInput,
            return_type=str,
            name=self._agent.name,
            description=self._agent.description,
        )

    async def run(self, args: AgentToolInput, cancellation_token: CancellationToken) -> str:
        team = RoundRobinGroupChat(
            participants=[self._agent], termination_condition=TextMessageTermination(source=self._agent.name)
        )
        response = None
        async for event in team.run_stream(task=args.task, cancellation_token=cancellation_token):
            if isinstance(event, TaskResult):
                response = event
                break
            # Way too noisy
            if isinstance(event, ModelClientStreamingChunkEvent):
                continue
            # We've already got the events
            elif isinstance(event, ToolCallSummaryMessage):
                continue
            else:
                event.metadata["agent_id"] = f"{TeamRuntimeContext.agent_id()}"
                await TeamRuntimeContext.current_runtime().publish_message(
                    GroupChatMessage(message=event), TeamRuntimeContext.output_channel()
                )

        assert response is not None
        return self._format_response(response)

    def _format_response(self, response: TaskResult) -> str:
        formatted_response: list[str] = []
        for message in response.messages:
            if isinstance(message, TextMessage):
                formatted_response += message.content
            elif isinstance(message, MultiModalMessage):
                raise NotImplementedError("MultiModalMessage is not supported yet.")
            elif isinstance(message, HandoffMessage):
                raise NotImplementedError("HandoffMessage is not supported yet.")
            elif isinstance(message, ToolCallSummaryMessage):
                formatted_response += message.content
            elif isinstance(message, StopMessage):
                continue
            elif isinstance(message, UserInputRequestedEvent):
                continue
            elif isinstance(message, ThoughtEvent):
                formatted_response += message.content
            elif isinstance(message, MemoryQueryEvent):
                raise NotImplementedError("MemoryQueryEvent is not supported yet.")
            elif isinstance(message, ModelClientStreamingChunkEvent):
                continue
            elif isinstance(message, ToolCallRequestEvent):
                continue
            elif isinstance(message, ToolCallExecutionEvent):
                continue

        return "\n".join(formatted_response)

    def _to_config(self) -> AgentToolConfig:
        return AgentToolConfig(agent=self._agent.dump_component())

    @classmethod
    def _from_config(cls, config: AgentToolConfig) -> Self:
        return cls(
            agent=AssistantAgent.load_component(config.agent),
        )
