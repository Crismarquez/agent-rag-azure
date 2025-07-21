from typing import Annotated, List, Dict

from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState

from agents.rag.schemas.tools import GeneralSearchInput, DomainSearchInput
from agents.rag.retriever.cognitivesearch import CognitiveSearch


@tool("general_search", args_schema=GeneralSearchInput)
async def general_search(
    query: str,
    state: Annotated[dict, InjectedState]
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

    return result_fields

@tool("domain_search", args_schema=DomainSearchInput)
async def domain_search(
    query: str,
    domain: str,
    state: Annotated[dict, InjectedState]
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
    return result_fields

AVAILABLE_TOOLS = [general_search, domain_search]
