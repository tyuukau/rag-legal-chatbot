from llama_index.core import VectorStoreIndex
from dotenv import load_dotenv

from ..settings import RAGSettings

load_dotenv()


class LocalVectorStoreFactory:
    def __init__(
        self,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None,
    ) -> None:
        # CHROMA VECTOR STORE
        self._setting = setting or RAGSettings()

    def get_index(self, nodes):
        if len(nodes) == 0:
            return None
        index = VectorStoreIndex(nodes=nodes)
        return index
