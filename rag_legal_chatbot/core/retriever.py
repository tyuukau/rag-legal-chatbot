from llama_index.core.callbacks.base import CallbackManager
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.retrievers.fusion_retriever import FUSION_MODES
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.llms.llm import LLM
from llama_index.core.schema import (
    BaseNode,
    IndexNode,
    NodeWithScore,
    QueryBundle,
)
from llama_index.core.selectors import LLMSingleSelector
from llama_index.core.tools import RetrieverTool
from llama_index.core.retrievers import (
    BaseRetriever,
    QueryFusionRetriever,
    RouterRetriever,
    VectorIndexRetriever,
)
from llama_index.retrievers.bm25 import BM25Retriever

from .prompts import QueryGenPrompt, SingleSelectPrompt

from ..settings import RAGSettings


class TwoStageRetriever(QueryFusionRetriever):
    """
    A retriever that performs two-stage retrieval using a fusion of multiple retrievers.

    Args:
        retrievers (list[BaseRetriever]): List of retrievers to be used for retrieval.
        setting (RAGSettings | None): RAGSettings object for configuring the retriever (default: None).
        llm (str | None): Language model to be used for retrieval (default: None).
        query_gen_prompt (str | None): Prompt for generating queries (default: None).
        mode (FUSION_MODES): Fusion mode for combining retriever results (default: FUSION_MODES.SIMPLE).
        similarity_top_k (int): Number of top similar documents to consider during retrieval (default: ...).
        num_queries (int): Number of queries to generate for each input query (default: 4).
        use_async (bool): Flag indicating whether to use asynchronous retrieval (default: True).
        verbose (bool): Flag indicating whether to print verbose output (default: False).
        callback_manager (CallbackManager | None): Callback manager for handling asynchronous retrieval (default: None).
        objects (list[IndexNode] | None): List of index nodes for retrieval (default: None).
        object_map (dict | None): Mapping of object IDs to index nodes (default: None).
        retriever_weights (list[float] | None): List of weights for retrievers during fusion (default: None).

    Attributes:
        _setting (RAGSettings): RAGSettings object for configuring the retriever.
        _rerank_model (SentenceTransformerRerank): SentenceTransformerRerank object for reranking retrieved results.

    """

    def __init__(
        self,
        retrievers: list[BaseRetriever],
        setting: RAGSettings | None = None,
        llm: str | None = None,
        query_gen_prompt: str | None = None,
        mode: FUSION_MODES = FUSION_MODES.SIMPLE,
        similarity_top_k: int = ...,
        num_queries: int = 4,
        use_async: bool = True,
        verbose: bool = False,
        callback_manager: CallbackManager | None = None,
        objects: list[IndexNode] | None = None,
        object_map: dict | None = None,
        retriever_weights: list[float] | None = None,
    ) -> None:
        super().__init__(
            retrievers,
            llm,
            query_gen_prompt,
            mode,
            similarity_top_k,
            num_queries,
            use_async,
            verbose,
            callback_manager,
            objects,
            object_map,
            retriever_weights,
        )
        self._setting = setting or RAGSettings()
        self._rerank_model = SentenceTransformerRerank(
            top_n=self._setting.retriever.top_k_rerank,
            model=self._setting.retriever.rerank_llm,
        )

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        """
        Perform retrieval using the two-stage retrieval approach.

        Args:
            query_bundle (QueryBundle): QueryBundle object containing the input query.

        Returns:
            list[NodeWithScore]: List of retrieved nodes with their corresponding scores.

        """
        queries: list[QueryBundle] = [query_bundle]

        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        if self.use_async:
            results = self._run_nested_async_queries(queries)
        else:
            results = self._run_sync_queries(queries)

        results = self._simple_fusion(results)

        return self._rerank_model.postprocess_nodes(results, query_bundle)

    async def _aretrieve(
        self, query_bundle: QueryBundle
    ) -> list[NodeWithScore]:
        """
        Perform asynchronous retrieval using the two-stage retrieval approach.

        Args:
            query_bundle (QueryBundle): QueryBundle object containing the input query.

        Returns:
            list[NodeWithScore]: List of retrieved nodes with their corresponding scores.

        """
        queries: list[QueryBundle] = [query_bundle]
        if self.num_queries > 1:
            queries.extend(self._get_queries(query_bundle.query_str))

        results = await self._run_async_queries(queries)
        results = self._simple_fusion(results)
        return self._rerank_model.postprocess_nodes(results, query_bundle)


class LocalRetrieverFactory:
    """
    This class represents a local retriever used in the RAG chatbot engine.

    Attributes:
        _setting (RAGSettings | None): The RAG settings object.
        _host (str): The host address.

    Methods:
        __init__: Initializes the _LocalRetriever object.
        _get_normal_retriever: Returns a normal retriever.
        _get_hybrid_retriever: Returns a hybrid retriever.
        _get_router_retriever: Returns a router retriever.
        get_retrievers: Returns the appropriate retriever based on the number of nodes.
    """

    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal",
    ):
        super().__init__()
        self._setting = setting or RAGSettings()
        self._host = host

    def _get_normal_retriever(
        self, vector_index: VectorStoreIndex, llm: LLM | None = None
    ):
        """
        Returns a normal retriever.

        Args:
            vector_index (VectorStoreIndex): The vector store index.
            llm (LLM | None): The LLM object.

        Returns:
            VectorIndexRetriever: The normal retriever.
        """
        llm = llm or Settings.llm
        return VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self._setting.retriever.similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=True,
        )

    def _get_hybrid_retriever(
        self,
        vector_index: VectorStoreIndex,
        llm: LLM | None = None,
        language: str = "eng",
        gen_query: bool = True,
    ) -> QueryFusionRetriever:
        """
        Returns a hybrid retriever.

        Args:
            vector_index (VectorStoreIndex): The vector store index.
            llm (LLM | None): The LLM object.
            language (str): The language.
            gen_query (bool): Whether to generate a query or not.

        Returns:
            QueryFusionRetriever or TwoStageRetriever: The hybrid retriever.
        """
        # VECTOR INDEX RETRIEVER
        vector_retriever = VectorIndexRetriever(
            index=vector_index,
            similarity_top_k=self._setting.retriever.similarity_top_k,
            embed_model=Settings.embed_model,
            verbose=True,
        )

        bm25_retriever = BM25Retriever.from_defaults(
            index=vector_index,
            similarity_top_k=self._setting.retriever.similarity_top_k,
            verbose=True,
        )

        # FUSION RETRIEVER
        if gen_query:
            hybrid_retriever = QueryFusionRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                retriever_weights=self._setting.retriever.retriever_weights,
                llm=llm,
                query_gen_prompt=QueryGenPrompt()(language),
                similarity_top_k=self._setting.retriever.top_k_rerank,
                num_queries=self._setting.retriever.num_queries,
                mode=self._setting.retriever.fusion_mode,
                verbose=True,
            )
        else:
            hybrid_retriever = TwoStageRetriever(
                retrievers=[bm25_retriever, vector_retriever],
                retriever_weights=self._setting.retriever.retriever_weights,
                llm=llm,
                query_gen_prompt=None,
                similarity_top_k=self._setting.retriever.similarity_top_k,
                num_queries=1,
                mode=self._setting.retriever.fusion_mode,
                verbose=True,
            )

        return hybrid_retriever

    def _get_router_retriever(
        self,
        vector_index: VectorStoreIndex,
        llm: LLM | None = None,
        language: str = "en",
    ):
        """
        Returns a router retriever.

        Args:
            vector_index (VectorStoreIndex): The vector store index.
            llm (LLM | None): The LLM object.

        Returns:
            RouterRetriever: The router retriever.
        """
        fusion_tool = RetrieverTool.from_defaults(
            retriever=self._get_hybrid_retriever(
                vector_index, llm, gen_query=True
            ),
            description="Use this tool when the user's query is ambiguous or unclear.",
            name="Fusion Retriever with BM25 and Vector Retriever and LLM Query Generation.",
        )
        two_stage_tool = RetrieverTool.from_defaults(
            retriever=self._get_hybrid_retriever(
                vector_index, llm, gen_query=False
            ),
            description="Use this tool when the user's query is clear and unambiguous.",
            name="Two Stage Retriever with BM25 and Vector Retriever and LLM Rerank.",
        )

        return RouterRetriever.from_defaults(
            selector=LLMSingleSelector.from_defaults(
                llm=llm,
                prompt_template_str=SingleSelectPrompt()(language=language),
            ),
            retriever_tools=[fusion_tool, two_stage_tool],
            llm=llm,
        )

    def get_retrievers(
        self,
        nodes: list[BaseNode],
        llm: LLM | None = None,
        language: str = "en",
    ):
        """
        Returns the appropriate retriever based on the number of nodes.

        Args:
            nodes (list[BaseNode]): The list of nodes.
            llm (LLM | None): The LLM object.

        Returns:
            VectorIndexRetriever or RouterRetriever: The retriever.
        """
        vector_index = VectorStoreIndex(nodes=nodes)

        retriever = self._get_normal_retriever(vector_index, llm)
        if len(nodes) > self._setting.retriever.top_k_rerank:
            retriever = self._get_router_retriever(
                vector_index, llm, language
            )

        return retriever
