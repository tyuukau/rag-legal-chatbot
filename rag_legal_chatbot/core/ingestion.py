import re
import os
from typing import Any
import pymupdf
from dotenv import load_dotenv
from tqdm import tqdm

from llama_index.core import Document, Settings
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter

from ..settings import RAGSettings

load_dotenv()


class LocalDataIngestion:
    def __init__(self, setting: RAGSettings | None = None) -> None:
        self._node_store = {}
        self._input_files = []
        self._ingested_files = []
        self._setting = setting or RAGSettings()

    def _filter_text(self, text):
        # Define the regex pattern.
        pattern = pattern = (
            r'[a-zA-Z0-9 \u00C0-\u01BF\u1EA0-\u1EFF`~!@#$%^&*()_\-+=\[\]\n{}|\\;:\'",.<>/?§]+'
        )
        matches = re.findall(pattern, text)
        # Join all matched substrings into a single string
        normalized_text = " ".join(matches)

        return normalized_text

    def process_documents(self):
        document_dir = os.path.abspath(self._setting.STORAGE.DOCUMENT_DIR)

        input_files = []

        for root, _, files in os.walk(document_dir):
            for file in files:
                file_path = os.path.join(root, file)
                input_files.append(file_path)

        self._input_files = input_files

    def store_nodes(
        self,
        embed_model: Any | None = None,
    ) -> None:
        self._ingested_files = []

        if len(self._input_files) == 0:
            return

        splitter = SentenceSplitter.from_defaults(
            chunk_size=self._setting.INGESTION.CHUNK_SIZE,
            chunk_overlap=self._setting.INGESTION.CHUCK_OVERLAP,
            paragraph_separator=self._setting.INGESTION.PARAGRAPH_SEP,
            secondary_chunking_regex=self._setting.INGESTION.CHUNKING_REGEX,
        )

        Settings.embed_model = embed_model or Settings.embed_model

        for input_file in tqdm(self._input_files, desc="Ingesting data"):
            file_name = input_file.strip().split("/")[-1]
            self._ingested_files.append(file_name)

            if file_name in self._node_store:
                continue

            document = pymupdf.open(input_file)
            all_text = ""

            for _, page in enumerate(document):
                page_text = page.get_text("text")
                page_text = self._filter_text(page_text)
                all_text += " " + page_text

            document = Document(
                text=all_text.strip(),
                metadata={
                    "file_name": file_name,
                },
            )

            nodes = splitter([document], show_progress=True)

            nodes = Settings.embed_model(nodes, show_progress=True)

            self._node_store[file_name] = nodes

    def reset(self):
        self._node_store = {}
        self._ingested_files = []

    def get_ingested_nodes(self) -> list[BaseNode]:
        return_nodes = []
        for file_name in self._ingested_files:
            return_nodes.extend(self._node_store[file_name])
        return return_nodes

    def get_all_nodes(self) -> list[BaseNode]:
        return_nodes = []
        for nodes in self._node_store.values():
            return_nodes.extend(nodes)
        return return_nodes
