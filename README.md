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

### 3. Provide a .env file with the following information

```bash
API_KEY=sk-proj-...
```

Place the .env file at the top level of the repository, ensuring it is at the same level as the `rag_legal_chatbot` folder.

Put your OpenAI API key in the `API_KEY` field.

### 4. Provide documents

Put your documents (`.txt`, `.pdf` files) in a folder named `data` at the same level as the `rag_legal_chatbot` folder.

When the application is run, the Chroma directory is typically at the `chroma` folder, at the same level as the `rag_legal_chatbot` folder. This folder will automatically be created when the application is run for the first time.

If you want to use another set of documents, it is advisable to create a new folder for your documents still at the same level as the `rag_legal_chatbot` folder, for example, `data2`. Then, in `settings.py`, you must change both `COLLECTION_NAME` and `DOCUMENT_DIR`.

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

```bash
python -m rag_legal_chatbot
```

If it is the first time, the app will ingest the data. Afterwards, the app will not ingest the data again unless you use a new dataset. In that case, you must change both `COLLECTION_NAME` and `DOCUMENT_DIR` in the `settings.py` file as mentioned in the [Provide documents](#4-provide-documents) section.
