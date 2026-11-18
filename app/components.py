import os

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch

from app.config import (
    EMBEDDING_MODEL,
    INDEX_NAME,
    NAMESPACE,
    LLM_MODEL,
    LLM_TEMPERATURE,
)


# ---

# LLM (Groq)
llm = ChatGroq(
    model_name=LLM_MODEL,
    temperature=LLM_TEMPERATURE,
)


# ---

# PineCone Retriever
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


# ---

# Tavily Websearch
web_search = TavilySearch(
    max_results=5,
    topic = "general",
    include_answer = True,
    include_raw_content = False,
)

print ("\n-- Tavily Websearch --\n")
print("Tavily Websearch configured:", bool(os.getenv("TAVILY_API_KEY")))