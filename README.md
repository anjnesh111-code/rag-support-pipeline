# Retrieval-Augmented Generation (RAG) Support Pipeline

## Project Overview
This repository contains a production-grade Retrieval-Augmented Generation (RAG) pipeline designed to ingest scraped markdown documentation and power an AI support chatbot. The architecture focuses on high-speed semantic retrieval, strict hallucination guardrails, and programmatic performance benchmarking.

## System Architecture
The pipeline is entirely contained within a Google Colab notebook, optimized for GPU acceleration, and follows a strict modular data flow:
1. **Data Ingestion & Processing:** Parses raw markdown files and implements a `RecursiveCharacterTextSplitter` (1000-character chunks, 200-character overlap) to preserve contextual integrity.
2. **Vector Embeddings:** Utilizes the Hugging Face `sentence-transformers/all-mpnet-base-v2` model deployed on a T4 GPU to convert text chunks into 768-dimensional dense vectors.
3. **Vector Database:** Connects to a **Pinecone** serverless index for highly scalable, low-latency nearest-neighbor search.
4. **Orchestration & LLM:** Uses **LangChain** to bind the retrieval module to Google's **Gemini 2.5 Flash** model. Custom prompt engineering enforces strict adherence to the provided context, preventing the LLM from fabricating answers.

## Key Features
* **Interactive Chat Loop:** A real-time terminal interface for testing user queries against the ingested knowledge base.
* **Hallucination Prevention:** The prompt template strictly limits the LLM to the retrieved context (e.g., safely refusing out-of-domain questions like "What is the capital of France?").
* **Performance Benchmarks:** Automated latency tracking for both the Pinecone vector retrieval phase and the Gemini LLM generation phase (averaging ~1.5 seconds total pipeline latency).
* **Automated Unit Testing:** A built-in validation suite that tests standard support queries, out-of-bounds questions, and edge cases to ensure reliable deployment.

## Technologies Used
* **Framework:** LangChain
* **LLM:** Google Gemini API (`gemini-2.5-flash`)
* **Embeddings:** Hugging Face (`all-mpnet-base-v2`)
* **Vector Database:** Pinecone
* **Compute:** Google Colab (T4 GPU)

## How to Run
This project is designed to run seamlessly in Google Colab.
1. Click the "Open in Colab" badge at the top of the notebook.
2. Provide your `PINECONE_API_KEY` and `GOOGLE_API_KEY` in the designated credential cells.
3. Ensure the runtime is set to **T4 GPU** (`Runtime` > `Change runtime type`).
4. Run the cells sequentially to initialize the database, embed the data, and launch the interactive support agent.
