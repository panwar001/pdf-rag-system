import uuid
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from get_embeddings import get_embedding_function
from langchain_huggingface import HuggingFaceEmbeddings

DB_PATH = "chroma"
DATA_PATH = "data"
COLLECTION_NAME = "document_collection"
FILE_PATH = "./data/TripSafe_Policy.pdf"

# STEP 1: Load PDF
def load_documents(pdf_path: str) -> List[Document]:
    print(f"Loading document pdf at path: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    return documents


# STEP 2: Chunk Documents
def chunk_document(documents: list[Document])-> List[Document]:
    print(f"Splitting document in chunks using TextSplitter")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " "]
    )

    chunks = splitter.split_documents(documents)

    # Enrich metadata (important for citations later)
    for chunk in chunks:
        chunk.metadata["chunk_id"] = str(uuid.uuid4())
        chunk.metadata.update({
            "travel_type": "international",
            "policy_region": "worldwide",
            "policy_scope": "international_only"
        })

    return chunks


# STEP 4: Store in ChromaDB
def add_to_chroma_db( chunks: List[Document],
    persist_dir: str,
    collection_name: str ):
    print(f"Adding Embedded chunks to Vector DB at path - {FILE_PATH}")

    try:
        embedding_function = get_embedding_function()

        vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_function,
            persist_directory=persist_dir
        )

        vector_store.add_documents(chunks)

        print(f"✅ Stored {len(chunks)} chunks in Chroma collection '{collection_name}'")


    except Exception as e:
        print(f"An error occurred: {e}")

# STEP 5: Store in ChromaDB
def load_vector_store():
    embedding_function = get_embedding_function()

    vector_store = Chroma(
        persist_directory=DATA_PATH,
        collection_name=COLLECTION_NAME,
        embedding_function=embedding_function
    )

    return vector_store

# MAIN PIPELINE
def run_pipeline():
    print("Loading PDF...")
    documents = load_documents(FILE_PATH)

    print("Chunking document...")
    chunks = chunk_document(documents)

    print("Generating embeddings & storing in ChromaDB...")
    add_to_chroma_db(
        chunks=chunks,
        persist_dir=DATA_PATH,
        collection_name=COLLECTION_NAME
    )

    print("Ingestion completed!")

if __name__ == "__main__":
    run_pipeline()
