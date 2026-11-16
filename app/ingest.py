from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from app.config import SOURCE_URL, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL


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
