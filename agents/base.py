from abc import ABC, abstractmethod

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import START, END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agents.rag.prompts.evaluation import EVALUATOR_SYSTEM_PROMPT
from agents.rag.schemas.evaluation import ScoreSchema
from agents.rag.utils import load_json_examples

from config.config import ENV_VARIABLES



class BaseAgent(ABC):
    """
    Abstract base class for all chat agents.
    Defines the common interface that all agents must implement.
    """
    
    def __init__(self, llm_provider: str, retrieval_args: dict):
        """
        Initializes a base agent.
        
        Args:
            llm_provider (str): The LLM provider to use (e.g. "azure").
            retrieval_args (dict): Arguments for information retrieval.
        """
        self.llm_provider = llm_provider
        self.retrieval_args = retrieval_args
    
    @abstractmethod
    async def run(self, history: list, metadata: dict) -> dict:
        """
        Runs the agent with the provided conversation history.
        
        Args:
            history (list): Conversation history.
            metadata (dict): Additional metadata for execution.
            
        Returns:
            dict: The agent's response.
        """
        pass


class EvaluationAgent():
    def __init__(self, llm_provider: str):
        self.llm_provider = llm_provider

        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")

        if self.llm_provider == "azure":
            self.llm = AzureChatOpenAI(
                deployment_name=ENV_VARIABLES["AZURE_OPENAI_CHATGPT4_DEPLOYMENT"],
                azure_endpoint=f"https://{ENV_VARIABLES['AZURE_OPENAI_SERVICE']}.openai.azure.com/",
                api_version="2025-04-01-preview",
            azure_ad_token_provider=token_provider,
            temperature=0.7,
            seed=42,
        )
        else:
            raise ValueError(f"LLM provider {self.llm_provider} not supported")
        
        self.evaluation_examples = load_json_examples("evaluation_examples.json")
        self.evaluation_prompt = EVALUATOR_SYSTEM_PROMPT

    async def run(self, record_dataset: dict) -> dict:
        
        prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            self.evaluation_prompt["system"]
        ),
        ("human",  self.evaluation_prompt["human"]),
        ]
        ) 
        
        eval_runnable = prompt | self.llm.with_structured_output(schema=ScoreSchema, method="function_calling", include_raw=False)
        eval_response = await eval_runnable.ainvoke({
            "examples": self.evaluation_examples,
            "ground_truth": record_dataset["answer"],
            "candidate": record_dataset["result"]["answer"]
            })
        return eval_response
