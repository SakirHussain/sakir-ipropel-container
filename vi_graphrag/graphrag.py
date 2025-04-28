import re
import json
import subprocess
import os
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.graphs import Neo4jGraph

# ----------- Config -----------
NEO4J_URL = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
CORPUS_PATH = r"Operating_system_concepts_10th.pdf"

# ----------- Model -----------
model = OllamaLLM(model="Mixtral 8x7B", temperature=0.7)

# ----------- Utilities -----------
def clean_text(text):
    text = re.sub(r"Page \d+ of \d+", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    return text

def remove_think_tags(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

# ----------- Vector Store -----------
def load_and_clean_corpus(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    cleaned_documents = [clean_text(doc.page_content) for doc in documents]
    return cleaned_documents

def create_vector_store(corpus_path):
    documents = load_and_clean_corpus(corpus_path)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_texts(documents, embeddings)

# ----------- Knowledge Graph -----------
def extract_entities_relations(text):
    pattern = r"(\w+)\s+(is|has|contains)\s+(\w+)"
    matches = re.findall(pattern, text)
    triples = [(m[0], m[1], m[2]) for m in matches]
    return triples

def populate_graph(corpus):
    graph = Neo4jGraph(
        url=NEO4J_URL,
        username=NEO4J_USER,
    )
    for doc in corpus:
        triples = extract_entities_relations(doc)
        for s, r, o in triples:
            graph.add_node(s)
            graph.add_node(o)
            graph.add_edge(s, r, o)
    return graph

# ----------- Neo4j Startup -----------
def start_neo4j_server():
    neo4j_dir = "./neo4j-community-5.16.0"
    java_home = "/usr/lib/jvm/java-17-openjdk-amd64"

    if not os.path.exists(neo4j_dir):
        print("[ERROR] Neo4j directory not found.")
        return False

    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = f"{java_home}/bin:" + os.environ["PATH"]

    print("[INFO] Starting Neo4j server...")
    try:
        subprocess.Popen([f"{neo4j_dir}/bin/neo4j", "start"])
        print("[INFO] Neo4j server started.")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to start Neo4j: {e}")
        return False

# ----------- Retrieval Chain -----------
def graph_retrieve(query, graph):
    cypher_query = f"""
    MATCH (e1)-[r]->(e2)
    WHERE e1.name CONTAINS '{query}' OR e2.name CONTAINS '{query}'
    RETURN e1.name, r.type, e2.name
    LIMIT 5
    """
    results = graph.query(cypher_query)
    return [f"{r['e1.name']} {r['r.type']} {r['e2.name']}" for r in results]

def graph_rag_generate(question, rubric, corpus_path):
    vector_store = create_vector_store(corpus_path)
    corpus = load_and_clean_corpus(corpus_path)
    graph = populate_graph(corpus)

    # Run both retrievers
    vector_docs = vector_store.similarity_search(question, k=3)
    graph_facts = graph_retrieve(question, graph)

    context_parts = []
    context_parts.extend([doc.page_content for doc in vector_docs])
    context_parts.extend(graph_facts)

    context = "\n\n".join(context_parts)

    prompt_template = PromptTemplate(
        template="""
        You are an AI assistant generating a detailed, structured answer based on retrieved knowledge.
        Use Chain-of-Thought (CoT) reasoning and integrate knowledge from graph and vector retrieval.

        Rubric:
        <rubric>
        {rubric}
        </rubric>

        Question:
        <question>
        {question}
        </question>

        Context:
        <context>
        {context}
        </context>

        Provide a detailed, paragraph-style answer covering all rubric points.
        """,
        input_variables=["question", "context", "rubric"]
    )

    chain = prompt_template | model
    llm_response = chain.invoke({"question": question, "context": context, "rubric": rubric})
    cleaned_response = remove_think_tags(llm_response)
    return cleaned_response

# ----------- Execution -----------
if __name__ == "__main__":
    #if start_neo4j_server():
        #with open("rubrics.json", "r") as file:
            #data = json.load(file)

        question = "Define Operating Systems."
        rubric = """Definition of Operating Systems (4 marks),
        Discuss Moore’s Law (3 marks),
        Provide an Overview of Operating Systems (3 marks)
        """
        generated_answer = graph_rag_generate(question, rubric, CORPUS_PATH)
        print("\nGenerated Answer:\n")
        print(generated_answer)
   # else:
        #print("[ERROR] Neo4j server not running. Exiting.")