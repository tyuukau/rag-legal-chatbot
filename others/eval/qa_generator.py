import random
import re
import os
import uuid
from tqdm import tqdm

from llama_index.core.llms.utils import LLM
from llama_index.core.schema import MetadataMode, TextNode
from llama_index.core.storage.docstore import DocumentStore
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

from rag_chatbot.core import LocalRAGModelFactory
from rag_chatbot.core import LocalEmbedding
from rag_chatbot.core import LocalDataIngestion
from rag_chatbot.core import RAGSettings


_DEFAULT_QA_GENERATE_PROMPT_TMPL = """\
Context information is below.

---------------------
{context_str}
---------------------

Given the context information and not prior knowledge.
generate only questions based on the below query.

You are a Teacher/ Professor. Your task is to setup \
{num_questions_per_chunk} questions for an upcoming \
quiz/examination. The questions should be diverse in nature \
across the document. Restrict the questions to the context information provided. \
Only provide the questions, not the answers.\"
"""


# generate queries as a convenience function
def _generate_question_context_pairs(
    nodes: list[TextNode],
    llm: LLM,
    qa_generate_prompt_tmpl: str = _DEFAULT_QA_GENERATE_PROMPT_TMPL,
    num_questions_per_chunk: int = 2,
) -> EmbeddingQAFinetuneDataset:
    """Generate examples given a set of nodes."""
    node_dict = {
        node.node_id: node.get_content(metadata_mode=MetadataMode.NONE)
        for node in nodes
    }

    queries = {}
    relevant_docs = {}
    for node_id, text in tqdm(node_dict.items()):
        query = qa_generate_prompt_tmpl.format(
            context_str=text, num_questions_per_chunk=num_questions_per_chunk
        )
        response = llm.complete(query)

        result = str(response).strip().split("\n")
        questions = [
            re.sub(r"^\d+[\).\s]", "", question).strip()
            for question in result
        ]
        questions = [question for question in questions if len(question) > 0]

        for question in questions:
            question_id = str(uuid.uuid4())
            queries[question_id] = question
            relevant_docs[question_id] = [node_id]

    # construct dataset
    return EmbeddingQAFinetuneDataset(
        queries=queries, corpus=node_dict, relevant_docs=relevant_docs
    )


class QAGenxerator:
    def __init__(
        self,
        embed_model: str | None = None,
        llm: str | None = None,
        host: str = "host.docker.internal",
    ) -> None:
        setting = RAGSettings()
        setting.ingestion.embed_llm = (
            embed_model or setting.ingestion.embed_llm
        )
        self._embed_model = LocalEmbedding.set(setting)
        self._llm = LocalRAGModelFactory.set(
            model_name=llm or setting.ollama.llm, host=host
        )
        self._ingestion = LocalDataIngestion()

    def generate(
        self,
        input_files: list[str],
        output_dir: str = "val_dataset",
        max_nodes: int = 100,
        num_questions_per_chunk=2,
    ) -> None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if os.path.exists(os.path.join(output_dir, "docstore.json")):
            print("Docstore already exist! Skip ingestion.")

        nodes = self._ingestion.store_nodes(input_files, embed_nodes=True)
        random.shuffle(nodes)
        dataset = _generate_question_context_pairs(
            nodes=nodes[:max_nodes],
            llm=self._llm,
            num_questions_per_chunk=num_questions_per_chunk,
        )

        # save dataset
        dataset.save_json(os.path.join(output_dir, "dataset.json"))

        # save nodes
        docstore = DocumentStore()
        docstore.add_documents(nodes)
        docstore.persist(
            persist_path=os.path.join(output_dir, "docstore.json")
        )
