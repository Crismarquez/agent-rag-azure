from pydantic import BaseModel, Field
from typing import List, Literal, Optional, Dict

class MessageItem(BaseModel):
    role: Literal["user", "assistant", "system"] = Field(..., description="Message role: user, assistant or system")
    content: str = Field(..., description="Message content")

class InputChat(BaseModel):
    user_id: str = Field(default="test0001", min_length=1, max_length=10000)
    conversation_id: str = Field(default="convtest0001", min_length=5, max_length=10000)
    history: List[MessageItem] = Field(..., min_items=1, max_items=100, description="Conversation message history")

class ResponseRAG(BaseModel):
    response: str
    agent_state: dict