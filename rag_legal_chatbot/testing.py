import json
from tqdm import tqdm

from .pipeline import LocalRAGPipeline

from llama_index.core.schema import MetadataMode
from llama_index.core.chat_engine.types import (
    AgentChatResponse,
)


# def test() -> None:
#     pipeline = LocalRAGPipeline()
#     pipeline.set_chat_engine()

#     message = "Co reguluje zákon o živnostenském podnikání?"

#     query_engine: CondensePlusContextChatEngine = pipeline._query_engine
#     response: AgentChatResponse = query_engine.chat(message=message)
#     # sources = [
#     #     n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
#     #     for n in response.source_nodes
#     # ]
#     print(response)


def mass_test(input_json: str, output_json: str) -> None:
    print("Initializing the pipeline...")
    # Initialize the pipeline
    pipeline = LocalRAGPipeline()
    pipeline.set_chat_engine()

    print(f"Reading input data from {input_json}...")
    # Read the JSON input
    with open(input_json, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Process each entry
    results = []
    print("Processing questions...")
    for entry in tqdm(data, desc="Processing questions", unit="question"):
        question = entry.get("question")
        law = entry.get("law")
        section = entry.get("section")
        answer = entry.get("answer")

        # Query the engine
        llm_answer: AgentChatResponse = pipeline._query_engine.chat(
            message=question
        )
        response = llm_answer.response
        sources = [
            n.node.get_content(metadata_mode=MetadataMode.LLM).strip()
            for n in llm_answer.source_nodes
        ]

        # Append results to the list
        results.append(
            {
                "question": question,
                "law": law,
                "section": section,
                "answer": answer,
                "llm_answer": response,
                "sources": sources,
            }
        )

    print(f"Writing results to {output_json}...")
    # Write all results to JSON
    with open(output_json, "w", encoding="utf-8") as jsonfile:
        json.dump(results, jsonfile, ensure_ascii=False, indent=4)

    print("Mass test completed successfully.")


GENERATE_QUESTION_PROMPT = """As a legal professor, your task is to create 5 diverse questions in Czech for an upcoming quiz/examination based on the provided text from the document. The questions should cover various sections of the document. Ensure the questions are relevant and aligned with the context provided.

Name of the pdf: {pdf_name}

Content: {law_content}

Guidelines:

Formulate 5 diverse questions covering different sections of the document in Czech.
Provide comprehensive answers for each question in Czech.
Ensure the questions and answers are clear and concise.
Use only the provided text for creating the questions and answers.
Do not mention the section in the question.
Output:

Return a JSON file containing a list of objects with the following fields for each question:

1. question
2. law
3. section
4. answer
"""

SYSTEM_PROMPT = """You are ChatGPT, a helpful AI that excels at creating quiz/examination questions based on provided legal documents. You always return a valid JSON array of objects, where each object contains four keys: "question", "law", "section", and "answer". Ensure the questions and answers are in Czech, covering different sections of the document, and are clear and concise."""
