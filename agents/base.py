from abc import ABC, abstractmethod

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
