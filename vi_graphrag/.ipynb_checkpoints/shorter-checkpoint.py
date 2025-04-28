import os
import json
import numpy as np
import networkx as nx

from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.vectorstores import FAISS

# === Configuration ===
pdf_base = "os"
out_dir = f"corpora/{pdf_base}"
max_nodes = 20
top_k = 10  # Top N nodes
output_file = "os_question_answer_dataset_top10.jsonl"

model_questions = [
        "Imagine an online ticket booking system where multiple users can book tickets for a show. Some users (checkers) simply want to check the availability of seats, while others (bookers) want to book or cancel tickets. The system must ensure that while a user is booking or cancelling a ticket, other users cannot check the availability to avoid inconsistencies. However, multiple users should be able to check seat availability simultaneously if no one is booking or cancelling tickets. Analyse the necessity of process synchronization for this scenario and develop a pseudocode solution using synchronization techniques to ensure data consistency while allowing efficient access for multiple checkers and bookers.",
        "Imagine a library with a study room that has a limited number of study desks. Each desk has a single chair and a single lamp. Students come to the library to study, but they need both a desk and a lamp to do so. The rules are: Limited Resources: There are only five desks and five lamps. Resource Allocation: A student must acquire both a desk and a lamp to start studying. Resource Release: Once a student finishes studying, they release both the desk and the lamp. Develop a solution that ensures all students can efficiently use a shared study room with limited desks and lamps, avoiding deadlock and starvation.",
        "Describe the different file allocation methods used in operating systems. Discuss the characteristics, advantages and limitations of each method.",
        "A user program may disrupt the normal operation of the system by issuing illegal 1/0 instructions, by accessing memory locations within the operating system itself, or by refusing to relinquish the CPU. Analyse and illustrate the use of various mechanisms to ensure that such disruptions cannot take place in the system.",
        "A data object (such as a file or record) is to be shared among several concurrent processes. Some of these processes may want only to read the content of the shared object, whereas others may want to update (that is, to read and write) the shared object. Apply the semaphore solution to solve the problem. Specify the required data structure and algorithm.",
        "The direct-access nature of disks allows us flexibility in the implementation of files. In almost every case, many files will be stored on the same disk. The main problem is how to allocate space to these files so that disk space is utilized effectively and files can be accessed quickly. Identify the three major methods of allocating disk space and specify their advantages and disadvantages.",
        "Illustrate how free-space allocation methods influence the efficiency of use of disk space, the performance of the file system, and the reliability of secondary storage.",
        "Concurrent execution of cooperating processes requires mechanisms that allow processes to communicate with one another and to synchronize their actions. Illustrate how cooperating processes can communicate in a shared-memory environment and also via an interprocess communication (IPC) facility.",
        "Identify and illustrate some of the issues to consider with multithreaded programs."

    
]

# === Initialize Models & Load Graph ===
embeddings = OllamaEmbeddings(model="bge-m3")
faiss_store = FAISS.load_local(f"{out_dir}/vectorstore", embeddings, allow_dangerous_deserialization=True)
G = nx.read_gpickle(f"{out_dir}/{pdf_base}_kg.gpickle")

def compute_similarity(a, b):
    a_emb = embeddings.embed_query(a)
    b_emb = embeddings.embed_query(b)
    sim = np.dot(a_emb, b_emb) / (np.linalg.norm(a_emb) * np.linalg.norm(b_emb) + 1e-8)
    return sim

def generate_dataset():
    llm = OllamaLLM(model="mixtral:8x7b", temperature=0.7)

    node_list = list(G.nodes)[:max_nodes]
    similarity_scores = []

    # === Step 1: Compute similarity of each node with best model question ===
    for node in node_list:
        chunk_text = G.nodes[node]['text']
        similarities = [compute_similarity(chunk_text, mq) for mq in model_questions]
        max_sim = max(similarities)
        best_model_q = model_questions[similarities.index(max_sim)]
        similarity_scores.append((node, max_sim, best_model_q))

    # === Step 2: Sort and pick top-k ===
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_nodes = similarity_scores[:top_k]

    print(f"\n[INFO] Selected Top-{top_k} nodes:")
    for idx, (node, sim, _) in enumerate(top_nodes, start=1):
        print(f"{idx}. Node: {node} | Similarity: {sim:.2f}")

    dataset = []

    # === Step 3: Generate question-answer pairs ===
    for idx, (node, sim, best_model_q) in enumerate(top_nodes, start=1):
        chunk_text = G.nodes[node]['text']

        question_prompt = f"""
You are an Operating Systems professor. Read the following context and create a clear, exam-style question based on the content, similar to this model question: "{best_model_q}"

Context:
\"\"\"{chunk_text}\"\"\"

Only output the question text.
"""
        question = llm.invoke(question_prompt).strip()

        expanded_nodes = list(G.neighbors(node))
        context_chunks = [G.nodes[node]['text']] + [G.nodes[n]['text'] for n in expanded_nodes]
        context_text = "\n".join(context_chunks)

        answer_prompt = f"""
You are an Operating Systems professor. Read the following context and generate a detailed, ideal answer for the following question:

Question: {question}

Context:
\"\"\"{context_text}\"\"\"

The answer should be at least 300 words, well-structured, and based only on the context.
"""
        ideal_answer = llm.invoke(answer_prompt).strip()

        dataset.append({
            "question": question,
            "ideal_answer": ideal_answer,
            "source_chunk": chunk_text,
            "matched_model_question": best_model_q,
            "similarity": sim
        })

        print(f"[INFO] Processed Top Node {idx}/{top_k}")

    # === Save ===
    with open(output_file, "w", encoding="utf-8") as f:
        for entry in dataset:
            f.write(json.dumps(entry) + "\n")

    print(f"\n[DONE] Dataset saved to {output_file}")

if __name__ == "__main__":
    generate_dataset()
