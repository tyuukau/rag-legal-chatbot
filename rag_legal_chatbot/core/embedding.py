import os
import torch
import requests
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from transformers import AutoTokenizer
from dotenv import load_dotenv

from ..settings import RAGSettings

load_dotenv()


class LocalEmbedding:
    @staticmethod
    def set(setting: RAGSettings | None = None, **kwargs):
        setting = setting or RAGSettings()
        model_name = setting.ingestion.embed_llm

        if model_name == "text-embedding-3-small":
            if setting.ingestion.embed_api_key is None:
                raise ValueError(
                    "API key is required for embedding model text-embedding-3-small."
                )
            return OpenAIEmbedding(
                model=model_name, api_key=setting.ingestion.embed_api_key
            )

        return HuggingFaceEmbedding(
            model_name=model_name,
            tokenizer=AutoTokenizer.from_pretrained(
                model_name, torch_dtype=torch.float16
            ),
            cache_folder=os.path.join(
                os.getcwd(), setting.ingestion.cache_folder
            ),
            trust_remote_code=True,
            embed_batch_size=setting.ingestion.embed_batch_size,
        )

    @staticmethod
    def pull(host: str, **kwargs):
        setting = RAGSettings()
        payload = {"name": setting.ingestion.embed_llm}
        return requests.post(
            f"http://{host}:11434/api/pull", json=payload, stream=True
        )

    @staticmethod
    def check_model_exist(host: str, **kwargs) -> bool:
        setting = RAGSettings()
        data = requests.get(f"http://{host}:11434/api/tags").json()
        list_model = [d["name"] for d in data["models"]]
        if setting.ingestion.embed_llm in list_model:
            return True
        return False
