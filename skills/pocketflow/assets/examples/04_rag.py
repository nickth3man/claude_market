"""
PocketFlow Cookbook Example: RAG (Retrieval Augmented Generation)

Difficulty: ☆☆☆ Dummy Level
Source: https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-rag

Description:
A simple RAG system with offline indexing and online querying.
Demonstrates:
- Two-stage RAG pipeline (offline + online)
- Document embedding and indexing
- Similarity search
- Context-based answer generation
"""

from pocketflow import Node, Flow
# from utils import call_llm, get_embedding, build_index, search_index
import sys


# ============================================================
# OFFLINE FLOW: Index Documents
# ============================================================

class EmbedDocumentsNode(Node):
    """Embed all documents for indexing"""

    def prep(self, shared):
        return shared["texts"]

    def exec(self, texts):
        """Generate embeddings for all texts"""
        print(f"\n📊 Embedding {len(texts)} documents...")
        # embeddings = [get_embedding(text) for text in texts]
        embeddings = [[0.1] * 128 for _ in texts]  # Placeholder
        return embeddings

    def post(self, shared, prep_res, exec_res):
        shared["embeddings"] = exec_res
        print(f"✅ Embedded {len(exec_res)} documents")
        return "default"


class BuildIndexNode(Node):
    """Build search index from embeddings"""

    def prep(self, shared):
        return shared["embeddings"]

    def exec(self, embeddings):
        """Create vector index"""
        print(f"\n🔨 Building index...")
        # index = build_faiss_index(embeddings)
        index = "placeholder_index"  # Placeholder
        return index

    def post(self, shared, prep_res, exec_res):
        shared["index"] = exec_res
        print("✅ Index built")
        return "default"


# Build offline flow
embed_docs = EmbedDocumentsNode()
build_index = BuildIndexNode()
embed_docs >> build_index
offline_flow = Flow(start=embed_docs)


# ============================================================
# ONLINE FLOW: Query and Answer
# ============================================================

class EmbedQueryNode(Node):
    """Embed the user query"""

    def prep(self, shared):
        return shared["query"]

    def exec(self, query):
        """Generate query embedding"""
        print(f"\n🔍 Processing query: {query}")
        # query_embedding = get_embedding(query)
        query_embedding = [0.1] * 128  # Placeholder
        return query_embedding

    def post(self, shared, prep_res, exec_res):
        shared["query_embedding"] = exec_res
        return "default"


class RetrieveDocumentNode(Node):
    """Search index and retrieve most relevant document"""

    def prep(self, shared):
        return {
            "query_embedding": shared["query_embedding"],
            "index": shared["index"],
            "texts": shared["texts"]
        }

    def exec(self, inputs):
        """Find most similar document"""
        print(f"\n📚 Searching index...")
        # I, D = search_index(inputs["index"], inputs["query_embedding"], top_k=1)
        # best_doc = inputs["texts"][I[0][0]]

        # Placeholder: return first document
        best_doc = inputs["texts"][0]

        print(f"✅ Retrieved document ({len(best_doc)} chars)")
        return best_doc

    def post(self, shared, prep_res, exec_res):
        shared["retrieved_document"] = exec_res
        return "default"


class GenerateAnswerNode(Node):
    """Generate answer using retrieved context"""

    def prep(self, shared):
        return {
            "query": shared["query"],
            "context": shared["retrieved_document"]
        }

    def exec(self, inputs):
        """Generate answer with context"""
        print(f"\n✍️  Generating answer...")

        prompt = f"""
Context: {inputs['context']}

Question: {inputs['query']}

Answer the question using only the information from the context:
"""
        # answer = call_llm(prompt)
        answer = f"Based on the context, the answer is: [Answer would be generated here]"
        return answer

    def post(self, shared, prep_res, exec_res):
        shared["generated_answer"] = exec_res
        print(f"✅ Answer generated")
        return "default"


# Build online flow
embed_query = EmbedQueryNode()
retrieve = RetrieveDocumentNode()
generate = GenerateAnswerNode()
embed_query >> retrieve >> generate
online_flow = Flow(start=embed_query)


# ============================================================
# Main Demo
# ============================================================

def run_rag_demo():
    """Run complete RAG demonstration"""

    # Sample documents
    texts = [
        """Pocket Flow is a 100-line minimalist LLM framework.
        Lightweight: Just 100 lines. Zero bloat, zero dependencies, zero vendor lock-in.
        Expressive: Everything you love—(Multi-)Agents, Workflow, RAG, and more.
        Agentic Coding: Let AI Agents (e.g., Cursor AI) build Agents—10x productivity boost!
        To install, pip install pocketflow or just copy the source code (only 100 lines).""",

        """NeurAlign M7 is a revolutionary non-invasive neural alignment device.
        Targeted magnetic resonance technology increases neuroplasticity in specific brain regions.
        Clinical trials showed 72% improvement in PTSD treatment outcomes.
        Developed by Cortex Medical in 2024 as an adjunct to standard cognitive therapy.
        Portable design allows for in-home use with remote practitioner monitoring.""",

        """Q-Mesh is QuantumLeap Technologies' instantaneous data synchronization protocol.
        Utilizes directed acyclic graph consensus for 500,000 transactions per second.
        Consumes 95% less energy than traditional blockchain systems.
        Adopted by three central banks for secure financial data transfer.
        Released in February 2024 after five years of development in stealth mode.""",
    ]

    # Get query from command line or use default
    default_query = "How to install PocketFlow?"
    query = default_query

    for arg in sys.argv[1:]:
        if arg.startswith("--"):
            query = arg[2:]
            break

    print("=" * 60)
    print("PocketFlow RAG Demo")
    print("=" * 60)

    # Single shared store for both flows
    shared = {
        "texts": texts,
        "query": query
    }

    # Stage 1: Index documents (offline)
    print("\n📥 STAGE 1: Indexing Documents")
    print("-" * 60)
    offline_flow.run(shared)

    # Stage 2: Query and answer (online)
    print("\n🔍 STAGE 2: Query and Answer")
    print("-" * 60)
    online_flow.run(shared)

    # Display results
    print("\n" + "=" * 60)
    print("✅ RAG Complete")
    print("=" * 60)
    print(f"\nQuery: {shared['query']}")
    print(f"\nRetrieved Context Preview:")
    print(shared["retrieved_document"][:150] + "...")
    print(f"\nGenerated Answer:")
    print(shared["generated_answer"])


if __name__ == "__main__":
    run_rag_demo()
