from llama_index.core import Settings
from llama_index.core.chat_engine.types import StreamingAgentChatResponse
from llama_index.core.prompts import ChatMessage, MessageRole

from .core import (
    LocalChatEngineFactory,
    LocalDataIngestion,
    LocalRAGModelFactory,
    LocalEmbedding,
    LocalVectorStoreFactory,
)

from .core.prompts import SystemPrompt


class LocalRAGPipeline:
    def __init__(self, host: str = "host.docker.internal") -> None:
        self._host = host
        self._language = "eng"
        self._model_name = ""
        self._engine = LocalChatEngineFactory(host=host)
        self._default_model = LocalRAGModelFactory.set(
            self._model_name, host=host
        )
        self._query_engine = None
        self._ingestion = LocalDataIngestion()
        self._vector_store = LocalVectorStoreFactory(host=host)
        Settings.llm = LocalRAGModelFactory.set(host=host)
        Settings.embed_model = LocalEmbedding.set(host=host)

    def get_model_name(self):
        return self._model_name

    def set_model_name(self, model_name: str):
        self._model_name = model_name

    def set_language(self, language: str):
        self._language = language

    def set_model(self):
        Settings.llm = LocalRAGModelFactory.set(
            model_name=self._model_name,
            system_prompt=SystemPrompt()(language=self._language),
            host=self._host,
        )
        self._default_model = Settings.llm

    def reset_engine(self):
        self._query_engine = self._engine.set_engine(
            llm=self._default_model, nodes=[], language=self._language
        )

    def reset_documents(self):
        self._ingestion.reset()

    def clear_conversation(self):
        self._query_engine.reset()

    def reset_conversation(self):
        self.reset_engine()

    def set_embed_model(self, model_name: str):
        Settings.embed_model = LocalEmbedding.set(model_name, self._host)

    def pull_model(self, model_name: str):
        return LocalRAGModelFactory.pull(self._host, model_name)

    def check_exist(self, model_name: str) -> bool:
        return LocalRAGModelFactory.check_model_exist(self._host, model_name)

    def store_nodes(self, input_files: list[str] = None) -> None:
        self._ingestion.store_nodes(input_files=input_files)

    def set_chat_mode(self):
        self.set_language(self._language)
        self.set_model()
        self.set_engine()

    def set_engine(self):
        self._query_engine = self._engine.set_engine(
            llm=self._default_model,
            nodes=self._ingestion.get_ingested_nodes(),
            language=self._language,
        )

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

    def query(
        self, mode: str, message: str, chatbot: list[list[str]]
    ) -> StreamingAgentChatResponse:
        if mode == "chat":
            history = self.get_history(chatbot)
            return self._query_engine.stream_chat(message, history)
        else:
            self._query_engine.reset()
            return self._query_engine.stream_chat(message)
