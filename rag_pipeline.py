import os
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


os.environ["GOOGLE_API_KEY"] = "***************************"
os.environ["PINECONE_API_KEY"] = "*******************************************8"
print("Connecting to your Pinecone database...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2",
    model_kwargs={'device': 'cuda'}
)
vectorstore = PineconeVectorStore(index_name="rag-support", embedding=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

#CHATBOT
print("Initializing Gemini LLM and RAG logic...")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

template = """
You are a helpful, AI-powered support chatbot agent for our platform.
Use the following pieces of retrieved context to answer the user's question.
If you don't know the answer based on the context, just say that you don't know.
Do not make up information.

Context: {context}

User Question: {input}

Helpful Answer:"""

custom_rag_prompt = PromptTemplate.from_template(template)

question_answer_chain = create_stuff_documents_chain(llm, custom_rag_prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
print("Systems online. Running test query...\n")
query = "What is the main topic of these documents?"
print(f"User: {query}")
response = rag_chain.invoke({"input": query})
print(f"Chatbot: {response['answer']}")
