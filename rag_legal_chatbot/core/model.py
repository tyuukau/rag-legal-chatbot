# from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
from dotenv import load_dotenv

# import requests

from ..settings import RAGSettings

from .prompts import SystemPrompt

load_dotenv()


class LocalRAGModelFactory:
    @staticmethod
    def set_model(
        model_name: str = "gpt-4o-mini",
        language: str = "en",
        # host: str = "host.docker.internal",
        setting: RAGSettings | None = None,
    ):
        setting = setting or RAGSettings()
        system_prompt: str = SystemPrompt()(language=language)
        if model_name in ["gpt-4o-mini", "gpt-4o"]:
            if setting.OLLAMA.API_KEY is None:
                raise ValueError(
                    "API key is required for models gpt-4o-mini, gpt-4o."
                )
            return OpenAI(
                model=model_name,
                system_prompt=system_prompt,
                temperature=setting.OLLAMA.TEMPERATURE,
                api_key=setting.OLLAMA.API_KEY,
                logprobs=False,
                default_headers=None,
            )
        else:
            raise ValueError("Must use OpenAI models.")
            # settings_kwargs = {
            #     "tfs_z": setting.OLLAMA.TFS_Z,
            #     "top_k": setting.OLLAMA.TOP_K,
            #     "top_p": setting.OLLAMA.TOP_P,
            #     "repeat_last_n": setting.OLLAMA.REPEAT_LAST_N,
            #     "repeat_penalty": setting.OLLAMA.REPEAT_PENALTY,
            # }
            # return Ollama(
            #     model=model_name,
            #     system_prompt=system_prompt,
            #     base_url=f"http://{host}:{setting.OLLAMA.PORT}",
            #     temperature=setting.OLLAMA.TEMPERATURE,
            #     context_window=setting.OLLAMA.CONTEXT_WINDOW,
            #     request_timeout=setting.OLLAMA.REQUEST_TIMEOUT,
            #     additional_kwargs=settings_kwargs,
            # )

    # @staticmethod
    # def pull(host: str, model_name: str):
    #     setting = RAGSettings()
    #     payload = {"name": model_name}
    #     return requests.post(
    #         f"http://{host}:{setting.OLLAMA.PORT}/api/pull",
    #         json=payload,
    #         stream=True,
    #     )

    # @staticmethod
    # def check_model_exist(host: str, model_name: str) -> bool:
    #     setting = RAGSettings()
    #     data = requests.get(
    #         f"http://{host}:{setting.OLLAMA.PORT}/api/tags"
    #     ).json()
    #     if data["models"] is None:
    #         return False
    #     list_model = [d["name"] for d in data["models"]]
    #     if model_name in list_model:
    #         return True
    #     return False
