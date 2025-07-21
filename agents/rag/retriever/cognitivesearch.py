
from azure.search.documents import SearchClient
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.models import VectorizedQuery
from langchain_openai import AzureOpenAIEmbeddings

from config.config import ENV_VARIABLES


class CognitiveSearch:
    def __init__(self) -> None:
        self.embedding_model = AzureOpenAIEmbeddings(
            azure_endpoint=f"https://{ENV_VARIABLES['AZURE_OPENAI_SERVICE']}.openai.azure.com",
            openai_api_key=ENV_VARIABLES["AZURE_OPENAI_API_KEY"],
            azure_deployment=ENV_VARIABLES["AZURE_OPENAI_EMBEDDING"],
            openai_api_version="2023-05-15",
        )

    async def generate_embeddings(self, text):
        embedding = await self.embedding_model.aembed_query(text)
        return embedding
    

    async def search(
        self,
        semantic_query: str,
        top: int = 5,
        use_hybrid: bool = True,
        filters: dict | None = None,
        **kwargs: dict | None,
    ):
        self.search_client = SearchClient(
            endpoint=f"https://{ENV_VARIABLES['AZURE_SEARCH_SERVICE']}.search.windows.net",
            index_name=ENV_VARIABLES["AZURE_SEARCH_INDEX"],
            credential=AzureKeyCredential(ENV_VARIABLES["AZURE_SEARCH_KEY"]),
        )

        vector = await self.generate_embeddings(semantic_query)

        vector_query = VectorizedQuery(
            vector=vector,
            k_nearest_neighbors=top,
            fields=ENV_VARIABLES["MAIN_VECTOR_FIELD"],
        )

        return await self._search(semantic_query, vector_query, top, use_hybrid, filters, **kwargs)

        
    async def _search(
        self,
        semantic_query: str,
        vector_query: VectorizedQuery,
        top: int = 5,
        use_hybrid: bool = True,
        filters: str | None = None,
        **kwargs: dict | None,
    ):


        if filters:
            filters = self._build_filters(filters)

            results = self.search_client.search(
                search_text=semantic_query,
                vector_queries=[vector_query],
                filter=filters,
                top=top,
            )
        else:
            results = self.search_client.search(
                search_text=semantic_query,
                vector_queries=[vector_query],
                top=top,
            )

        result_docs = [record for record in results]
        result_docs = self.delete_duplicates(result_docs)

        return result_docs

    def _build_filters(self, filters: dict) -> str:
        """
        Convierte un diccionario de filtros a una expresión OData para Azure Cognitive Search.

        Ejemplo de entrada:
        {
            "category": "health",
            "region": ["north", "south"]
        }

        Resultado:
        "category eq 'health' and (region eq 'north' or region eq 'south')"
        """
        filter_expressions = []

        for key, value in filters.items():
            if isinstance(value, list):
                if not value:
                    continue  # evitar listas vacías
                # Unir cada valor con "or" para ese campo
                expressions = [f"{key} eq '{v}'" for v in value]
                joined = " or ".join(expressions)
                filter_expressions.append(f"({joined})")
            elif isinstance(value, str):
                filter_expressions.append(f"{key} eq '{value}'")
            else:
                raise ValueError(f"Tipo de filtro no soportado: {key} = {value} ({type(value)})")

        return " and ".join(filter_expressions)
    
    def delete_duplicates(self, documents: list, id_field: str = "id_content"):
        """
        Elimina duplicados de una lista de documentos basándose en el campo id_content.
        """
        seen = set()
        return [doc for doc in documents if not (doc[id_field] in seen or seen.add(doc[id_field]))]
    
