import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import DirectoryLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
import time
from tqdm.auto import tqdm

loader = DirectoryLoader('/content/drive/MyDrive/scraped', glob="**/*.md", loader_cls=UnstructuredMarkdownLoader)
documents = loader.load()


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

#Embeddings Pipeline
print("Downloading Hugging Face embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# Reconnect to Pinecone index
vectorstore = PineconeVectorStore(index_name="rag-support", embedding=embeddings)

batch_size = 100
max_retries = 3

print(f"Starting rate-limited upload for {len(chunks)} chunks...")

for i in tqdm(range(0, len(chunks), batch_size), desc="Uploading to Pinecone"):
    batch = chunks[i : i + batch_size]


    for attempt in range(max_retries):
        try:
            vectorstore.add_documents(batch)
            time.sleep(1)
            break

        except Exception as e:
            if attempt == max_retries - 1:
                print(f"\nFailed on batch {i} after 3 attempts. Error: {e}")
            else:
                print(f"\nNetwork hiccup detected. Retrying batch {i} in 3 seconds...")
                time.sleep(3)

print("Data successfully uploaded and indexed!")
