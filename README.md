# rag_legal_chatbot

This application enables interactive querying of internal documents with the help of a LLM. It uses semantic searching for retrieving relevant information and a ChromaDB vector database for storage.

## Getting Started

Follow these steps to set up and run the application:

### 1. Create and activate a virtual environment

1. Create a new virtual environment:

    ```bash
    python -m venv .venv
    ```

    Use Python 3.10 or above.

2. Activate the virtual environment:

    On Windows:

    ```bash
    .venv\Scripts\activate
    ```

    On macOS and Linux:

    ```bash
    source .venv/bin/activate
    ```

### 2. Install dependencies

Install the required packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

Install Ollama:

- macOS, Window: [Download](https://ollama.com/)
  
  - Note. On macOS, after downloading the app and run it for the first time, you must manually quit Ollama from the menubar. If you do not do this, the app may not run.

- Linux

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Provide a .env file with the following information

```bash
API_KEY=sk-proj-...
```

Place the .env file at the top level of the repository, ensuring it is at the same level as the `rag_legal_chatbot` folder.

Put your OpenAI API key in the `API_KEY` field.

### 4. Provide documents

Put your documents (`.txt`, `.pdf` files) in a folder named `data` at the same level as the `rag_legal_chatbot` folder. This repository has provided a sample `data` folder, which is the same folder used for testing.

If you want to use another set of documents, it is advisable to create a new folder for your documents still at the same level as the `rag_legal_chatbot` folder, for example, `data2`. Then, in `settings.py`, you must change both `COLLECTION_NAME` and `DOCUMENT_DIR`. Only after that do you run the app again.

When the application is run, the Chroma directory is typically housed in the `chroma` folder, at the same level as the `rag_legal_chatbot` folder. This folder will automatically be created when the application is run for the first time.

This is intended to emulate the actual use case in RAG system with internal documents, where users typically do not upload new documents.

```python
class StorageSettings(BaseModel):
    PERSIST_DIR: str = Field(
        default="./chroma", description="Chroma directory"
    )
    COLLECTION_NAME: str = Field(
        default="collection", description="Collection name" # New: "collection2"
    )
    DOCUMENT_DIR: str = Field(default="./data", description="Data directory") # New: "./data2"
```

Overall, the top level structure should look like thus:

```bash
.venv
.env
README.md
data
rag_legal_chatbot
├─ __main__.py
├─ ...
└─ core
   ├─ ...
...
```

## Run the Application

### Run mode

```bash
python -m rag_legal_chatbot --mode run
```

If it is the first time, the app will ingest the data. Afterwards, the app will not ingest the data again unless you use a new dataset. In that case, you must change both `COLLECTION_NAME` and `DOCUMENT_DIR` in the `settings.py` file as mentioned in the [Provide documents](#4-provide-documents) section.

### Test mode

```bash
python -m rag_legal_chatbot --mode test --input_json <path_to_input_json> --output_json <path_to_output_json>
```

Arguments:

- `--input_json`: Path to the input JSON file containing the test questions. If not specified, the default is `data/test_questions.json`.
- `--output_json`: Path to the output JSON file where the test results will be saved. If not specified, the default is `data/test_results.json`.

To run the test mode with default input and output JSON files, you can simply use:

```bash
python -m rag_legal_chatbot --mode test
```

The form of the input JSON file follows that of the provided JSON file `data/test_questions.json`.

## Demo

https://github.com/user-attachments/assets/44346b42-e11d-452c-9765-0633a9031b20
