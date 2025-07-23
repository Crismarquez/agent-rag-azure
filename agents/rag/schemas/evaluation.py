from pydantic import BaseModel, Field
from typing import Optional, Annotated, Literal

class ScoreSchema(BaseModel):
    """Puntuación de la pregunta."""
    analysis: str =  Field(..., description="Análisis de la respuesta generada.")
    evaluation: Literal["none", "few", "most", "all"]
