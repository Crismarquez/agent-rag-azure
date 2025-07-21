
from typing import Literal

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.base import BaseAgent
from config.config import ENV_VARIABLES
from agents.rag.schemas.graph import AgentState, GuardialSchema
from agents.rag.prompts.agent import AGENT_SYSTEM_PROMPT_RRR, AGENT_SYSTEM_PROMPT_PIC
from agents.rag.prompts.guardrails import GUARDRAILS_PROMPT, FRIENDLY_RESPONSE_PROMPT
from agents.rag.utils import format_tool_for_prompt
from agents.rag.tools.base import AVAILABLE_TOOLS
from agents.rag.utils import load_guardrails_examples

from tracing.tracing_config import get_tracer

class RAGAgent(BaseAgent):
    """
    Agent specialized in user workspace.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.llm = AzureChatOpenAI(
                    deployment_name=ENV_VARIABLES["AZURE_OPENAI_CHATGPT4_DEPLOYMENT"],
                    api_key=ENV_VARIABLES["AZURE_OPENAI_API_KEY"],
                    azure_endpoint=f"https://{ENV_VARIABLES['AZURE_OPENAI_SERVICE']}.openai.azure.com/",
                    api_version="2025-04-01-preview",
                    temperature=0.7,
                    seed=42,
                )
        
        self.guardrails_prompt = GUARDRAILS_PROMPT
        self.friendly_response_prompt = FRIENDLY_RESPONSE_PROMPT
        self.guardrails_examples = load_guardrails_examples()

        self.agent_graph = self.create()

    def route_condition(self, state: AgentState):
        """
        Route the agent to the appropriate node based on the state.
        """
        # If len of messages is greater than 12, stop the agent
        if len(state["messages"]) > 12:
            return END
        # If the latest message requires a tool, route to tools
        # Otherwise, provide a direct response
        if hasattr(state["messages"][-1], "tool_calls") and len(state["messages"][-1].tool_calls) > 0:
            return "tools"
        else:
            return END

    def create(self):
        """
        Creates the user workspace agent.
        """
        builder = StateGraph(AgentState)
        builder.add_node("guardrial", self.guardrial_node)
        builder.add_node("friendly_response", self.friendly_response)
        builder.add_node("agent_brain", self.agent_brain)
        builder.add_node("tools", ToolNode(AVAILABLE_TOOLS))

        builder.add_edge(START, "guardrial")
        builder.add_conditional_edges(
            "agent_brain",
            self.route_condition,
        )

        builder.add_edge("tools", "agent_brain")
        #builder.add_edge("guardial", "agent_brain")
        builder.add_edge("friendly_response", END)

        # Compile the graph
        agent_graph = builder.compile()

        # Obtener los bytes de la imagen PNG
        png_bytes = agent_graph.get_graph().draw_mermaid_png()
        # Guardar la imagen en un archivo
        with open("grafo_rag_agent.png", "wb") as f:
            f.write(png_bytes)
        
        return agent_graph
    
    async def guardrial_node(self, state: AgentState) -> Command[Literal["agent_brain", "friendly_response"]]:
        """Process incoming messages through guardrails to determine if they should be accepted.
        
        Args:
            state (AgentState): The current state containing messages and metadata.
            
        Returns:
            Command: A command object specifying the next node to execute ("agent_brain" or "friendly_response")
            and any state updates. If the message passes guardrails, it proceeds to agent_brain;
            otherwise, it goes to friendly_response with a rejection reason.
        """

        prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            self.guardrails_prompt["system"] # .format(examples=self.guardrails_examples)
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human",  self.guardrails_prompt["human"]),
    ]
        ) 
        
        guardials_runnable = prompt | self.llm.with_structured_output(schema=GuardialSchema, method="function_calling", include_raw=False)
        guardials_response = await guardials_runnable.ainvoke({
            "examples": self.guardrails_examples,
            "input_user": state["messages"][-1].content, 
            "history": state["messages"]})

        if guardials_response.classification == "accepted":
            goto = "agent_brain"
            update = {
                "classification": guardials_response.classification,
            }
        else:
            goto = "friendly_response"
            update = {
                "classification": guardials_response.classification,
                "reject_reason": guardials_response.reason
                }
        
        return Command(goto=goto, update=update)


    async def friendly_response(self, state: AgentState):
        """Generate a friendly response when a message fails guardrails.
        
        Args:
            state (AgentState): The current state containing the rejection reason and message history.
            
        Returns:
            dict: A dictionary containing the friendly response message.
        """

        prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            self.friendly_response_prompt["system"]
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human",  self.friendly_response_prompt["human"]),
    ]
        ) 
        
        friendly_response_runnable = prompt | self.llm
        friendly_response_answer = await friendly_response_runnable.ainvoke({"reject_reason": state["reject_reason"], "history": state["messages"]})

        return {
            "messages": friendly_response_answer,
        }

    
    async def agent_brain(self, state: AgentState):
        """Process accepted messages through the main agent logic.
        
        Args:
            state (AgentState): The current state containing messages and metadata.
            
        Returns:
            dict: A dictionary containing the processed messages, conversation ID, and user ID.
            The response is generated using the last 12 messages of context.
        """

        llm_with_tools = self.llm.bind_tools(AVAILABLE_TOOLS)
        textual_description_tool = [format_tool_for_prompt(tool_schema) for tool_schema in AVAILABLE_TOOLS]
        textual_description_tool = "\n\n".join(textual_description_tool)

        sys_msg = SystemMessage(content=AGENT_SYSTEM_PROMPT_PIC.format(textual_description_tool=textual_description_tool))

        response = [await llm_with_tools.ainvoke([sys_msg] + state["messages"])]
        
        # Convert the response back to a dictionary and append to existing messages
        return {
            "messages": response,
            "conversation_id": state["conversation_id"],
            "user_id": state["user_id"]
        }

    async def _run_graph(self, messages: list, metadata: dict):
        """Internal method to run the agent with the given messages and metadata.
        
        Args:
            messages: List of message dictionaries to process.
            metadata: Metadata of the conversation.
            
        Returns:
            dict: The final state after running the agent graph.
        """
        if not self.agent_graph:
            await self.build_graph()
        
        langchain_messages = []
        for msg in messages:
            if msg.role == "user":
                langchain_messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                langchain_messages.append(AIMessage(content=msg.content))
            
        # Initialize the state
        state = {
            "messages": langchain_messages,
            "conversation_id": metadata["conversation_id"],
            "user_id": metadata["user_id"],
            "document": metadata.get("document", "")
        }
        
        #configuration sent to the invoke method
        config = {
        #"callbacks": [get_tracer()],
        "run_name": f"RAGAgent",
        #"recursion_limit": 3
        }
        # Run the agent
        return await self.agent_graph.ainvoke(state, config=config)
    
    async def run(self, history: list, metadata: dict) -> dict:
        """
        Executes the user workspace agent with the provided conversation history.
        """
    
        history_messages = history[-20:]

        final_response = await self._run_graph(history_messages, metadata)

        return final_response

    async def stream_run(self, history: list, metadata: dict):
        """
        Yields tuples (stream_mode, chunk) from the graph.
        """
        # Prepare LangChain messages
        state = {
            "messages": [HumanMessage(content=m["content"]) for m in history],
            "alert_id": metadata["alert_id"],
            "alert": metadata["alert"]
        }

        # LangGraph’s async stream
        async for mode, chunk in self.agent_graph.astream(
            state,
            stream_mode=["messages", "custom"],
        ):
            yield {"type": mode, "data": chunk}
