import time
import os

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from app.config import SOURCE_URL, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, INDEX_NAME, NAMESPACE


# ---

# Load documents from the web using WebBaseLoader with custom headers
loader = WebBaseLoader(
    web_paths = (SOURCE_URL,),
    requests_kwargs = {
        "headers" : {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
    }
)

raw_documents = loader.load()

print("\n-- DOCUMENTS LOADER --\n")
print("Loaded documents:", len(raw_documents))
print("Source:", raw_documents[0].metadata.get("source"))
print("\nPreview:\n")
print(raw_documents[0].page_content[:1500])  # Print the first 1500 characters of the first document

# Split the documents into chunks for processing
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    add_start_index=True
)

chunks = splitter.split_documents(raw_documents)

print("\n-- CHUNKS SPLITTER --\n")
print("Number of chunks created:", len(chunks))
print("Preview of first chunk:\n")
print(chunks[0].page_content[:500])  # Print the first 500 characters of the first chunk

# Create embedding model for the chunks
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    encode_kwargs={"normalize_embeddings": True},  # unit-length vectors for cosine similarity
)

# Create the Pinnecone vector store from the chunks and embeddings
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create the index if it doesn't exist
existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    
while not pc.describe_index(INDEX_NAME).status["ready"]:
    time.sleep(2)

print("\n-- PINECONE VECTOR STORE --\n")
print("Pinecone index ready:", INDEX_NAME)

# # re-running ingest overwrites the same vectors
# ids = [f"{NAMESPACE}-{chunk.metadata['start_index']}" for chunk in chunks]

# Upload the chunks to the Pinecone index
PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    index_name=INDEX_NAME,
    namespace=NAMESPACE,
    # ids=ids,
)

print("Uploaded chunks:", len(chunks), "to namespace:", NAMESPACE)