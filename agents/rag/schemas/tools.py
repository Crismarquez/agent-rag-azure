from pydantic import BaseModel, Field
from typing import Optional, Annotated, Literal
from langgraph.prebuilt import InjectedState
from langchain_core.tools import InjectedToolCallId

class GeneralSearchInput(BaseModel):
    """Búsqueda general de información."""
    query: str = Field(..., description="Frase de búsqueda")
    state: Optional[Annotated[dict, InjectedState]] = None
    tool_call_id: Annotated[str, InjectedToolCallId]

class DomainSearchInput(BaseModel):
    """Búsqueda general de información."""
    query: str = Field(..., description="Frase de búsqueda")
    domain: Literal["manuales", "garantias"] = Field(..., description="Dominio de la búsqueda: consultas relacionadas con manuales de producto debes usar el dominio 'manuales' y para consultas relacionadas con garantías debes usar el dominio 'garantias'")
    state: Optional[Annotated[dict, InjectedState]] = None
    tool_call_id: Annotated[str, InjectedToolCallId]