from pydantic import BaseModel, Field
from typing import Optional, Annotated, Literal

class ScoreSchema(BaseModel):
    """Puntuación de la pregunta."""
    evalutaion: Literal[""]