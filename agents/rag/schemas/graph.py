from typing import TypedDict, Annotated, Optional, Literal
import operator
from pydantic import BaseModel, Field
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

# TODO: add ids to the messages
import uuid
from typing import Optional, List


def combine_ids(
    existing: List[str] | None, 
    update: List[str] | str
) -> List[str]:
    """
    - Si `update` es una cadena "CLEAR", reinicia la lista.
    - Si es str distinto, lo añade.
    - Si es lista, la extiende.
    - Elimina duplicados preservando orden.
    """
    # Arrancamos con lista vacía si no había nada
    entries: List[str] = existing or []

    # Handle special "CLEAR"
    if update == "CLEAR":
        return []

    # Aplanar update
    if isinstance(update, list):
        entries.extend(update)
    else:
        entries.append(update)

    # Quitar duplicados
    seen = set()
    deduped: List[str] = []
    for i in entries:
        if i not in seen:
            seen.add(i)
            deduped.append(i)
    return deduped


class AgentState(TypedDict):
    user_id: str
    conversation_id : str
    messages: Annotated[list[AnyMessage], add_messages]
    reject_reason: Optional[str] = None
    classification: Optional[str] = None
    # should be optional Annotated[list[str], add_ids]
    ids_content: Annotated[List[str], combine_ids]

class GuardialSchema(BaseModel):
    classification: Literal["accepted", "rejected"] = Field(
        description="Question classification: 'accepted' if the question is related to who performs certain tasks or who can assist with specific processes, "
        "'rejected'  if the question is about topics outside of this context.",)
    
    reason: str = Field(
        description="Reason for the classification of the question. This should be a short and concise explanation (10-15 words) that the user can understand.",
    )
    