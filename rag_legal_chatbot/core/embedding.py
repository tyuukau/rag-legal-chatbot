import os
import torch

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import AutoTokenizer

from ..settings import RAGSettings


class LocalEmbeddingFactory:
    @staticmethod
    def set_embedding(setting: RAGSettings | None = None, **kwargs):
        setting = setting or RAGSettings()
        model_name = setting.INGESTION.EMBED_LLM

        if model_name == "text-embedding-3-small":
            if setting.INGESTION.EMBED_API_KEY is None:
                raise ValueError(
                    "API key is required for the embedding model 'text-embedding-3-small'."
                )
            return OpenAIEmbedding(
                model=model_name, api_key=setting.INGESTION.EMBED_API_KEY
            )

        return HuggingFaceEmbedding(
            model_name=model_name,
            tokenizer=AutoTokenizer.from_pretrained(
                model_name, torch_dtype=torch.float16
            ),
            cache_folder=os.path.join(
                os.getcwd(), setting.INGESTION.CACHE_FOLDER
            ),
            trust_remote_code=True,
            embed_batch_size=setting.INGESTION.EMBED_BATCH_SIZE,
        )
