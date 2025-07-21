from typing import TypedDict, Annotated, Optional, Literal
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    user_id: str
    conversation_id : str
    messages: Annotated[list[AnyMessage], add_messages]
    reject_reason: Optional[str] = None
    classification: Optional[str] = None

class GuardialSchema(BaseModel):
    classification: Literal["accepted", "rejected"] = Field(
        description="Question classification: 'accepted' if the question is related to who performs certain tasks or who can assist with specific processes, "
        "'rejected'  if the question is about topics outside of this context.",)
    
    reason: str = Field(
        description="Reason for the classification of the question. This should be a short and concise explanation (10-15 words) that the user can understand.",
    )
    