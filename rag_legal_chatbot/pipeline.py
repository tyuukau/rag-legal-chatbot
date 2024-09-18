from llama_index.core import Settings
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.prompts import ChatMessage, MessageRole

from .core import (
    LocalChatEngineFactory,
    LocalDataIngestion,
    LocalRAGModelFactory,
    LocalEmbeddingFactory,
)


class LocalRAGPipeline:
    def __init__(self, host: str = "host.docker.internal") -> None:
        self._host = host

        self._model_name = "gpt-4o-mini"

        self._engine = LocalChatEngineFactory(host=host)
        self._default_model = LocalRAGModelFactory.set_model(self._model_name)
        self._query_engine = None
        self._ingestion = LocalDataIngestion()
        Settings.llm = LocalRAGModelFactory.set_model()
        Settings.embed_model = LocalEmbeddingFactory.set_embedding(host=host)

    #############
    # INGESTION #
    #############

    def process_document_dir(self):
        self._ingestion.process_documents()

    def store_nodes(self) -> None:
        self._ingestion.store_nodes()

    def check_store_exists(self) -> bool:
        return self._engine.check_store_exists()

    #############
    # LLM MODEL #
    #############

    def set_model(self, language: str = "en"):
        Settings.llm = LocalRAGModelFactory.set_model(
            model_name=self._model_name, language=language
        )
        self._default_model = Settings.llm

    # def pull_model(self, model_name: str):
    #     return LocalRAGModelFactory.pull(self._host, model_name)

    # def check_exist(self, model_name: str) -> bool:
    #     return LocalRAGModelFactory.check_model_exist(self._host, model_name)

    ###########
    # ENGINGE #
    ###########

    def set_engine(self, language: str = "en"):
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            nodes=self._ingestion.get_ingested_nodes(),
            language=language,
        )

    def set_chat_engine(self, language: str = "en"):
        self.set_model(language)
        self.set_engine(language)

    ################
    # CONVERSATION #
    ################

    def clear_conversation(self):
        self._query_engine.reset()

    def get_history(self, chatbot: list[list[str]]):
        history = []
        for chat in chatbot:
            if chat[0]:
                history.append(
                    ChatMessage(role=MessageRole.USER, content=chat[0])
                )
                history.append(
                    ChatMessage(role=MessageRole.ASSISTANT, content=chat[1])
                )
        return history

    #########
    # QUERY #
    #########

    def query(
        self, message: str, chatbot: list[list[str]]
    ) -> StreamingAgentChatResponse:
        self._query_engine.reset()
        return self._query_engine.stream_chat(message)

    async def aquery(
        self, message: str, chatbot: list[list[str]]
    ) -> StreamingAgentChatResponse:
        self._query_engine.reset()
        return await self._query_engine.astream_chat(message)
