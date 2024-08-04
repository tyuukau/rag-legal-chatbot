import os
import torch
import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import AutoTokenizer
from dotenv import load_dotenv

from ..settings import RAGSettings

load_dotenv()


class LocalEmbeddingFactory:
    @staticmethod
    def set_embedding(setting: RAGSettings | None = None, **kwargs):
        setting = setting or RAGSettings()
        model_name = setting.INGESTION.EMBED_LLM

        if model_name == "text-embedding-3-small":
            if setting.INGESTION.EMBED_API_KEY is None:
                raise ValueError(
                    "API key is required for embedding model text-embedding-3-small."
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

    @staticmethod
    def pull(host: str, **kwargs):
        setting = RAGSettings()
        payload = {"name": setting.INGESTION.EMBED_LLM}
        return requests.post(
            f"http://{host}:11434/api/pull", json=payload, stream=True
        )

    @staticmethod
    def check_model_exist(host: str, **kwargs) -> bool:
        setting = RAGSettings()
        data = requests.get(f"http://{host}:11434/api/tags").json()
        list_model = [d["name"] for d in data["models"]]
        if setting.INGESTION.EMBED_LLM in list_model:
            return True
        return False
