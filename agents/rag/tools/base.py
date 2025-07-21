from typing import Annotated, List, Dict

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langgraph.types import Command
from langchain_core.tools import InjectedToolCallId
from langchain_core.messages import ToolMessage

from agents.rag.schemas.tools import GeneralSearchInput, DomainSearchInput
from agents.rag.retriever.cognitivesearch import CognitiveSearch


@tool("general_search", args_schema=GeneralSearchInput)
async def general_search(
    query: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> List[Dict]:
    """
    Utiliza este tool para buscar información en la base de conocimiento.
    """
    include_fields = ["domain", "source", "id_document", "id_content", "@search.score", "@search.reranker_score", "content"]
    #user_id = state['user_id']
    search_client = CognitiveSearch()
    search_results = await search_client.search(
        query,
        top=20,
        use_hybrid=True,
    )
    result_fields: List[Dict] = [
        { field: record.get(field) for field in include_fields }
        for record in search_results
    ]

    # get all ids
    ids_content = [record['id_content'] for record in result_fields]

    return Command(update={
        "ids_content": ids_content,
        "messages": [
            ToolMessage(content=str(result_fields), tool_call_id=tool_call_id)
        ]
        })

@tool("domain_search", args_schema=DomainSearchInput)
async def domain_search(
    query: str,
    domain: str,
    state: Annotated[dict, InjectedState],
    tool_call_id: Annotated[str, InjectedToolCallId]
) -> List[Dict]:
    """
    Utiliza este tool para buscar información en la base de conocimiento filtrada por dominio.
    """

    include_fields = ["domain", "source", "id_document", "id_content", "@search.score", "@search.reranker_score", "content"]

    search_client = CognitiveSearch()
    search_results = await search_client.search(
        query,
        top=20,
        use_hybrid=True,
        filters={"domain": domain}
    )

    result_fields: List[Dict] = [
        { field: record.get(field) for field in include_fields }
        for record in search_results
    ]
    ids_content = [record['id_content'] for record in result_fields]

    return Command(update={
        "ids_content": ids_content,
        "messages": [
            ToolMessage(content=str(result_fields), tool_call_id=tool_call_id)
        ]
        })

AVAILABLE_TOOLS = [general_search, domain_search]
