import os
import re
import numpy as np
import networkx as nx

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate

##############################
#  Debug Utility
##############################
def debug(msg):
    print(f"[DEBUG] {msg}")

##############################
#  Text Cleaning
##############################
def clean_text(text):
    text = re.sub(r"Page\s*\d+\s*of\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

##############################
#  PDF Loading & Chunking
##############################
def load_textbook(pdf_path):
    debug(f"Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    debug(f"Loaded {len(docs)} pages")

    cleaned = [clean_text(doc.page_content) for doc in docs]
    return "\n\n".join(cleaned)

def chunk_text(text, chunk_size=1000, chunk_overlap=100):
    debug(f"Splitting text into chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = splitter.split_text(text)
    debug(f"Created {len(chunks)} chunks")
    return chunks

##############################
#  Knowledge Graph
##############################
def build_knowledge_graph(chunks):
    debug(f"Building knowledge graph with {len(chunks)} nodes")
    G = nx.DiGraph()
    for i, chunk in enumerate(chunks):
        node_id = f"chunk_{i}"
        G.add_node(node_id, text=chunk)
    return G

def create_chunk_relationships(G, embeddings_dict, threshold=0.8):
    debug(f"Creating edges (threshold={threshold})...")
    node_list = list(G.nodes)
    count = 0
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            n1, n2 = node_list[i], node_list[j]
            e1, e2 = embeddings_dict[n1], embeddings_dict[n2]
            sim = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-8)
            if sim >= threshold:
                G.add_edge(n1, n2, relation="similar")
                G.add_edge(n2, n1, relation="similar")
                count += 2
    debug(f"Added {count} edges")

##############################
#  Vector Store
##############################
def embed_nodes_create_faiss(G, model="bge-m3"):
    debug(f"Embedding nodes with model='{model}'")
    embeddings = OllamaEmbeddings(model=model)
    node_list = list(G.nodes)
    node_texts = [G.nodes[n]["text"] for n in node_list]

    faiss_store = FAISS.from_texts(node_texts, embeddings)

    embeddings_dict = {}
    for node in node_list:
        embeddings_dict[node] = embeddings.embed_query(G.nodes[node]["text"])

    debug("Embedding complete")
    return faiss_store, embeddings_dict

def vector_search(query, faiss_store, k=3):
    debug(f"Running vector search: '{query}'")
    results = faiss_store.similarity_search(query, k=k)
    debug(f"Found {len(results)} results")
    return results

##############################
#  Graph Expansion
##############################
def expand_with_graph(results, G, max_depth=1):
    debug(f"Expanding context (depth={max_depth})")
    expanded = set()
    node_list = list(G.nodes)

    for doc in results:
        content = doc.page_content
        for node in node_list:
            if G.nodes[node]['text'] == content:
                queue = [(node, 0)]
                visited = set()
                while queue:
                    current, depth = queue.pop()
                    if current in visited or depth > max_depth:
                        continue
                    visited.add(current)
                    expanded.add(G.nodes[current]['text'])
                    for neighbor in G.neighbors(current):
                        queue.append((neighbor, depth + 1))
    debug(f"Expanded context size: {len(expanded)}")
    return list(expanded)

##############################
#  Prompt
##############################
prompt_template = PromptTemplate(
    template="""
    You are a college professor tasked with generating a structured and comprehensive answer for a given question taking help from the retrieved knowledge.  
    The answer will be worth 10 marks in total so the answer MUST BE 500 WORDS OR MORE.

    Task Overview
    - Use Chain-of-Thought (CoT) reasoning to analyze the question step by step.
    - Extract key insights from the provided context.
    - Construct a well-structured, paragraph-based answer that directly satisfies the rubric criteria.

    <question>
    {question}
    </question>

    Retrieved Context
    The following context has been retrieved from reliable sources.  
    Use this information to construct an accurate and detailed response to the given question:

    <context>
    {context}
    </context>

    Response Generation Guidelines
    - The response must be a fully detailed and structured answer.  
    - DO NOT include any explanations, formatting, labels, or extra text—only generate the answer.  
    - The output should be a cohesive, well-written paragraph addressing all rubric points. 
    """,
    input_variables=["context", "question"]
)

##############################
#  Main
##############################
def main():
    pdf_path = "os.pdf"
    base = os.path.splitext(os.path.basename(pdf_path))[0]
    out_dir = f"corpora/{base}"
    os.makedirs(out_dir, exist_ok=True)

    # === Check if already processed ===
    if not os.path.exists(f"{out_dir}/vectorstore"):
        debug("No existing store. Starting fresh...")
        text = load_textbook(pdf_path)
        chunks = chunk_text(text)

        # Knowledge Graph
        G = build_knowledge_graph(chunks)

        # Embeddings & FAISS
        faiss_store, embeddings_dict = embed_nodes_create_faiss(G)
        create_chunk_relationships(G, embeddings_dict, threshold=0.8)

        # Save
        debug("Saving vectorstore & knowledge graph...")
        faiss_store.save_local(f"{out_dir}/vectorstore")
        nx.write_gpickle(G, f"{out_dir}/{base}_kg.gpickle")
        debug("Saved successfully")
    else:
        debug("Vectorstore & knowledge graph found. Loading...")

    # === Reload & Test ===
    embeddings = OllamaEmbeddings(model="bge-m3")
    faiss_store = FAISS.load_local(f"{out_dir}/vectorstore", embeddings, allow_dangerous_deserialization=True)
    G = nx.read_gpickle(f"{out_dir}/{base}_kg.gpickle")
    debug(f"Graph loaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

    # === Example Query ===
    query = "What is the use of a kernel?"
    results = vector_search(query, faiss_store, k=3)
    print("\n===== TOP 3 =====")
    print(results)
    expanded = expand_with_graph(results, G, max_depth=1)
    print("\n===== EXPANDED =====")
    print(expanded)
    context = "\n".join(expanded)

    # === LLM Answer ===
    prompt = prompt_template.format(context=context, question=query)
    debug(f"Invoking LLM: 'mixtral:8x7b'")
    llm = OllamaLLM(model="mixtral:8x7b", temperature=0.7)
    answer = llm.invoke(prompt)

    print("\n===== FINAL ANSWER =====")
    print(answer)

if __name__ == "__main__":
    main()
