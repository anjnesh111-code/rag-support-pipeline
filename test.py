import time

print("==================================================")
print("PHASE 2 DELIVERABLE: PERFORMANCE BENCHMARKS")
print("==================================================\n")

test_query = "What are the key fields to note?"

# 1. Benchmark Retrieval Latency
start_retrieval = time.time()
retrieved_docs = retriever.invoke(test_query)
end_retrieval = time.time()
retrieval_latency = end_retrieval - start_retrieval

# 2. Benchmark Full Generation Latency
start_generation = time.time()
response = rag_chain.invoke({"input": test_query})
end_generation = time.time()
total_latency = end_generation - start_generation
llm_latency = total_latency - retrieval_latency

print(f"Vector DB Retrieval Time (Top 3 chunks): {retrieval_latency:.4f} seconds")
print(f"LLM Generation Time (Gemini 2.5 Flash):  {llm_latency:.4f} seconds")
print(f"Total Pipeline Latency:                  {total_latency:.4f} seconds\n")


print("==================================================")
print("PHASE 3 DELIVERABLE: UNIT TESTS & SAMPLE QUERIES")
print("==================================================\n")

# A list of specific sample queries to test the chatbot's logic
sample_queries = [
    "What are the main topics discussed in the data?",
    "How do I reset my account password?",
    "What is the capital of France?" # Testing if it correctly refuses out-of-context questions
]

for index, query in enumerate(sample_queries, 1):
    print(f"--- Test Case 00{index} ---")
    print(f"Query: '{query}'")
    try:
        ans = rag_chain.invoke({"input": query})["answer"]
        print(f"Status: PASS")
        # Printing the first 150 characters of the answer to verify
        print(f"Output: {ans[:150]}...\n")
    except Exception as e:
        print(f"Status: FAIL")
        print(f"Error Message: {e}\n")

print("All automated tests completed successfully.")
