import json
from pathlib import Path
from typing import List, Dict
from langchain_core.tools import BaseTool

from config.config import BASE_DIR

def format_tool_for_prompt(tool: BaseTool) -> str:
    """
    Toma un objeto BaseTool y devuelve su descripción en texto
    con firma de parámetros y documentación, para pegar en el system prompt.
    """
    name = tool.name
    desc = tool.description.strip()
    # Pydantic v2: accedemos a model_fields
    fields = tool.args_schema.model_fields

    # Construimos la firma, ignorando “state”
    params_signature = []
    for fname, model_field in fields.items():
        if fname == "state" or fname == "tool_call_id":
            continue
        # extraemos el nombre del tipo
        t = getattr(model_field.annotation, "__name__", str(model_field.annotation))
        if model_field.is_required:
            params_signature.append(f"{fname}: {t}")
        else:
            params_signature.append(f"{fname}: {t} = {model_field.default!r}")
    signature = f"{name}({', '.join(params_signature)})"

    # Documentación de cada parámetro, ignorando “state”
    params_docs = []
    for fname, model_field in fields.items():
        if fname == "state":
            continue
        desc_field = model_field.description or ""
        params_docs.append(f"{fname}: {desc_field}")
    params_docs_str = "\n        ".join(params_docs)

    # Asumimos retorno de texto; podrías hacerlo dinámico leyendo tool.return_direct
    return_type = "str"

    prompt_block = f"""{signature} -> {return_type}:
    {desc}

    Args:
        {params_docs_str}

    Returns:
        {return_type} result."""
    return prompt_block

def load_guardrails_examples() -> List[Dict[str, str]]:
    with open(Path(BASE_DIR, "agents", "rag", "prompts", "guardrails_examples.json"), "r") as f:
        return json.load(f)
