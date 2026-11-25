import time
import os

from edgar import Company
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from app.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL, INDEX_NAME, TICKER, NAMESPACE

# ---


# Load the latest 10-K from SEC EDGAR
company = Company(TICKER)
filing = company.get_filings(form="10-K", amendments=False).latest()
tenk = filing.obj()

print("\n-- SEC FILING --\n")
print(filing)

# Sections we want from the 10-K
sections = [
    ("Item 1", "Business", tenk.business),
    ("Item 1A", "Risk Factors", tenk.risk_factors),
    ("Item 1C", "Cybersecurity", tenk["Item 1C"]),
    ("Item 2", "Properties", tenk["Item 2"]),
    ("Item 3", "Legal Proceedings", tenk["Item 3"]),
    ("Item 5", "Market for Stock, Dividends and Buybacks", tenk["Item 5"]),
    ("Item 7", "MD&A", tenk.management_discussion),
    ("Item 7A", "Market Risk", tenk["Item 7A"]),
]

# One Document per section, with metadata for filtering and citations
raw_documents = []
for item, name, text in sections:
    if not text:
        print(f"Skipped {item} ({name}): no text found")
        continue

    raw_documents.append(
        Document(
            page_content=text,
            metadata={
                "ticker": TICKER,
                "form": "10-K",
                "section": item,
                "section_name": name,
                "filing_date": str(filing.filing_date),
                "period": str(filing.period_of_report),
                "accession_no": filing.accession_no,
                "url": filing.homepage_url,
            },
        )
    )

print("\n-- DOCUMENTS --\n")
print("Sections loaded:", len(raw_documents))
for doc in raw_documents:
    print(f"{doc.metadata['section']} ({doc.metadata['section_name']}): {len(doc.page_content)} characters")

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
        dimension=768,
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

# re-running ingest overwrites the same vectors instead of duplicating
ids = [
    f"{c.metadata['ticker']}-{c.metadata['accession_no']}-{c.metadata['section']}-{c.metadata['start_index']}"
    for c in chunks
]

# Upload the chunks to the Pinecone index
PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embedding_model,
    index_name=INDEX_NAME,
    namespace=NAMESPACE,
    ids=ids,
)

print("Uploaded chunks:", len(chunks), "to namespace:", NAMESPACE)