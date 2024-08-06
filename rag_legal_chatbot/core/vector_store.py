import chromadb

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..settings import RAGSettings


class LocalVectorStoreFactory:
    def __init__(
        self,
        host: str = "host.docker.internal",
        setting: RAGSettings | None = None,
    ) -> None:
        # CHROMA VECTOR STORE
        self._setting = setting or RAGSettings()
        self._persist_dir = self._setting.STORAGE.PERSIST_DIR
        self._collection_name = self._setting.STORAGE.COLLECTION_NAME

    def check_exist_vector_store_index(self) -> bool:
        db = chromadb.PersistentClient(path=self._persist_dir)
        col_exists = True
        try:
            _ = db.get_collection(self._collection_name)
        except Exception:
            col_exists = False

        return col_exists

    def get_or_create_vector_store_index(self, nodes) -> VectorStoreIndex:
        db = chromadb.PersistentClient(path=self._persist_dir)
        col_exists = True
        try:
            collection = db.get_collection(self._collection_name)
        except Exception:
            col_exists = False

        if col_exists:
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            index = VectorStoreIndex.from_vector_store(
                vector_store, storage_context=storage_context
            )
        else:
            collection = db.create_collection(self._collection_name)
            vector_store = ChromaVectorStore(chroma_collection=collection)
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store
            )
            index = VectorStoreIndex(
                nodes=nodes, storage_context=storage_context
            )

        return index
