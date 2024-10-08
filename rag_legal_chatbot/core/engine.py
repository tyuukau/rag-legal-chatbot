from typing import Literal
from dotenv import load_dotenv

from llama_index.core.schema import (
    BaseNode,
)
from llama_index.core.llms.llm import LLM
from llama_index.core.chat_engine import (
    CondensePlusContextChatEngine,
    SimpleChatEngine,
)
from llama_index.core.memory import ChatMemoryBuffer

from .vector_store import LocalVectorStoreFactory
from .prompts import CondensePrompt, ContextPrompt, SystemPrompt
from .retriever import LocalRetrieverFactory

from ..settings import RAGSettings

load_dotenv()


class LocalChatEngineFactory:
    def __init__(
        self,
        setting: RAGSettings | None = None,
        host: str = "host.docker.internal",
    ):
        super().__init__()
        self._setting = setting or RAGSettings()
        self.retriever_factory = LocalRetrieverFactory(self._setting)
        self.host = host

    def check_store_exists(self) -> bool:
        return LocalVectorStoreFactory(
            setting=self._setting
        ).check_exist_vector_store_index()

    def set_engine(
        self,
        llm: LLM,
        nodes: list[BaseNode],
        language: str = "eng",
        chat_mode: Literal["QA", "chat"] = "QA",
    ) -> CondensePlusContextChatEngine | SimpleChatEngine:

        # Normal chat engine
        if chat_mode == "chat":
            return SimpleChatEngine.from_defaults(
                llm=llm,
                memory=ChatMemoryBuffer(
                    token_limit=self._setting.OLLAMA.CHAT_TOKEN_LIMIT
                ),
            )

        # Chat engine with documents
        return CondensePlusContextChatEngine.from_defaults(
            retriever=self.retriever_factory.get_retrievers(
                llm=llm, nodes=nodes, language=language
            ),
            llm=llm,
            memory=ChatMemoryBuffer(
                token_limit=self._setting.OLLAMA.CHAT_TOKEN_LIMIT
            ),
            system_prompt=SystemPrompt()(language=language),
            context_prompt=ContextPrompt()(language=language),
            condense_prompt=CondensePrompt()(language=language),
        )
