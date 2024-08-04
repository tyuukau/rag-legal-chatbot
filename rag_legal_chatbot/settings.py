from typing import Union
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
load_dotenv(os.path.join(BASE_DIR, ".env"))


class OllamaSettings(BaseModel):
    LLM: str = Field(default="gpt-4o-mini", description="LLM model")
    API_KEY: Union[str, None] = Field(
        default=os.getenv("API_KEY", None), description="API key"
    )
    KEEP_ALIVE: str = Field(
        default="1h", description="Keep alive time for the server"
    )
    TFS_Z: float = Field(default=1.0, description="TFS normalization factor")
    TOP_K: int = Field(default=40, description="Top k sampling")
    TOP_P: float = Field(default=0.9, description="Top p sampling")
    REPEAT_LAST_N: int = Field(default=64, description="Repeat last n tokens")
    REPEAT_PENALTY: float = Field(default=1.1, description="Repeat penalty")
    REQUEST_TIMEOUT: float = Field(default=300, description="Request timeout")
    PORT: int = Field(default=11434, description="Port number")
    CONTEXT_WINDOW: int = Field(
        default=8000, description="Context window size"
    )
    TEMPERATURE: float = Field(default=0.1, description="Temperature")
    CHAT_TOKEN_LIMIT: int = Field(
        default=10000, description="Chat memory limit"
    )


class RetrieverSettings(BaseModel):
    NUM_QUERIES: int = Field(
        default=5, description="Number of generated queries"
    )
    SIMILARITY_TOP_K: int = Field(default=10, description="Top k documents")
    RETRIEVER_WEIGHTS: list[float] = Field(
        default=[0.4, 0.6], description="Weights for retriever"
    )
    TOP_K_RERANK: int = Field(default=10, description="Top k rerank")
    RERANK_LLM: str = Field(
        default="BAAI/bge-reranker-large", description="Rerank LLM model"
    )
    FUSION_MODE: str = Field(
        default="dist_based_score", description="Fusion mode"
    )


class IngestionSettings(BaseModel):
    EMBED_LLM: str = Field(
        default="text-embedding-3-small", description="Embedding LLM model"
    )
    EMBED_API_KEY: Union[str, None] = Field(
        default=os.getenv("API_KEY", None), description="API key"
    )
    EMBED_BATCH_SIZE: int = Field(
        default=8, description="Embedding batch size"
    )
    CACHE_FOLDER: str = Field(
        default="data/huggingface", description="Cache folder"
    )
    CHUNK_SIZE: int = Field(default=256, description="Document chunk size")
    CHUCK_OVERLAP: int = Field(
        default=32, description="Document chunk overlap"
    )
    CHUNKING_REGEX: str = Field(
        default="[^,.;。？！]+[,.;。？！]?", description="Chunking regex"
    )
    PARAGRAPH_SEP: str = Field(
        default="\n \n", description="Paragraph separator"
    )
    NUM_WORKERS: int = Field(default=0, description="Number of workers")


class StorageSettings(BaseModel):
    PERSIST_DIR: str = Field(
        default="./chroma", description="Chroma directory"
    )
    COLLECTION_NAME: str = Field(
        default="collection", description="Collection name"
    )


class RAGSettings(BaseModel):
    OLLAMA: OllamaSettings = OllamaSettings()
    RETRIEVER: RetrieverSettings = RetrieverSettings()
    INGESTION: IngestionSettings = IngestionSettings()
    STORAGE: StorageSettings = StorageSettings()
