import os

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import EMBEDDING_MODEL, INDEX_NAME, NAMESPACE


# ---

# Read-only: connects to the existing Pinecone index
def get_retriever(k: int = 3):
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

    vector_store = PineconeVectorStore(
        index=pc.Index(INDEX_NAME),
        embedding=embedding_model,
        namespace=NAMESPACE,
    )

    return vector_store.as_retriever(search_kwargs={"k": k})
