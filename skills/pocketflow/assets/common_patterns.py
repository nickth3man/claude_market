"""
Common PocketFlow Patterns

Ready-to-use examples for common use cases
"""

from pocketflow import Node, BatchNode, Flow, BatchFlow
# from utils.call_llm import call_llm  # Implement your LLM wrapper


# ============================================================
# Pattern 1: Simple Sequential Workflow
# ============================================================

class LoadDataNode(Node):
    """Load data from file/API/database"""
    def prep(self, shared):
        return shared["source_path"]

    def exec(self, path):
        # TODO: Implement your data loading
        with open(path, 'r') as f:
            return f.read()

    def post(self, shared, prep_res, exec_res):
        shared["raw_data"] = exec_res
        return "default"


class ProcessDataNode(Node):
    """Process the data"""
    def prep(self, shared):
        return shared["raw_data"]

    def exec(self, data):
        # TODO: Your processing logic
        processed = data.upper()  # Example
        return processed

    def post(self, shared, prep_res, exec_res):
        shared["processed_data"] = exec_res
        return "default"


class SaveResultNode(Node):
    """Save results"""
    def post(self, shared, prep_res, exec_res):
        result = shared["processed_data"]
        # TODO: Save to file/API/database
        print(f"Saved: {result}")
        return "default"


# Build flow
load = LoadDataNode()
process = ProcessDataNode()
save = SaveResultNode()
load >> process >> save
simple_flow = Flow(start=load)


# ============================================================
# Pattern 2: Batch Processing (Map-Reduce)
# ============================================================

class ChunkAndSummarize(BatchNode):
    """Chunk large text and summarize each chunk"""

    def prep(self, shared):
        # Split into chunks
        text = shared["large_text"]
        chunk_size = 1000
        chunks = [text[i:i+chunk_size]
                  for i in range(0, len(text), chunk_size)]
        return chunks

    def exec(self, chunk):
        # Process each chunk
        # summary = call_llm(f"Summarize: {chunk}")
        summary = f"Summary of: {chunk[:50]}..."  # Placeholder
        return summary

    def post(self, shared, prep_res, exec_res_list):
        # Combine all summaries
        shared["summaries"] = exec_res_list
        shared["combined_summary"] = "\n\n".join(exec_res_list)
        return "default"


# ============================================================
# Pattern 3: Agent with Decision Making
# ============================================================

class DecideActionNode(Node):
    """Agent decides what action to take"""

    def prep(self, shared):
        return shared.get("context", ""), shared["query"]

    def exec(self, inputs):
        context, query = inputs

        # Simplified decision logic
        # In real implementation, use LLM to decide
        if "search" in query.lower():
            return {"action": "search", "term": query}
        else:
            return {"action": "answer", "response": f"Answer for: {query}"}

    def post(self, shared, prep_res, exec_res):
        shared["decision"] = exec_res
        return exec_res["action"]  # Return action for branching


class SearchNode(Node):
    """Search for information"""
    def exec(self, prep_res):
        term = self.shared.get("decision", {}).get("term")
        # TODO: Implement search
        return f"Search results for: {term}"

    def post(self, shared, prep_res, exec_res):
        shared["context"] = exec_res
        return "continue"


class AnswerNode(Node):
    """Provide final answer"""
    def prep(self, shared):
        return shared.get("decision", {}).get("response")

    def post(self, shared, prep_res, exec_res):
        shared["final_answer"] = prep_res
        return "done"


# Build agent flow
decide = DecideActionNode()
search = SearchNode()
answer = AnswerNode()

decide - "search" >> search
decide - "answer" >> answer
search - "continue" >> decide  # Loop back for more decisions

agent_flow = Flow(start=decide)


# ============================================================
# Pattern 4: RAG (Retrieval Augmented Generation)
# ============================================================

class ChunkDocuments(BatchNode):
    """Chunk documents for indexing"""

    def prep(self, shared):
        return shared["documents"]  # List of documents

    def exec(self, doc):
        # Chunk each document
        chunk_size = 500
        chunks = [doc[i:i+chunk_size]
                  for i in range(0, len(doc), chunk_size)]
        return chunks

    def post(self, shared, prep_res, exec_res_list):
        # Flatten all chunks
        all_chunks = [chunk for doc_chunks in exec_res_list
                      for chunk in doc_chunks]
        shared["chunks"] = all_chunks
        return "default"


class EmbedAndIndex(Node):
    """Embed chunks and create index"""

    def prep(self, shared):
        return shared["chunks"]

    def exec(self, chunks):
        # TODO: Create embeddings and build index
        # embeddings = [get_embedding(chunk) for chunk in chunks]
        # index = build_faiss_index(embeddings)
        return "index_placeholder"

    def post(self, shared, prep_res, exec_res):
        shared["index"] = exec_res
        return "default"


class QueryRAG(Node):
    """Query the RAG system"""

    def prep(self, shared):
        return shared["query"], shared["index"], shared["chunks"]

    def exec(self, inputs):
        query, index, chunks = inputs
        # TODO: Search index and retrieve relevant chunks
        # relevant = search_index(index, query, top_k=3)
        relevant = chunks[:3]  # Placeholder

        # Generate answer with context
        context = "\n".join(relevant)
        # answer = call_llm(f"Context: {context}\n\nQuestion: {query}")
        answer = f"Answer based on context"
        return answer

    def post(self, shared, prep_res, exec_res):
        shared["answer"] = exec_res
        return "default"


# Build RAG flow
chunk = ChunkDocuments()
index = EmbedAndIndex()
chunk >> index
rag_indexing_flow = Flow(start=chunk)

query = QueryRAG()
rag_query_flow = Flow(start=query)


# ============================================================
# Pattern 5: Error Handling with Fallback
# ============================================================

class ResilientNode(Node):
    """Node with error handling"""

    def __init__(self):
        super().__init__(max_retries=3, wait=5)

    def exec(self, prep_res):
        # Risky operation that might fail
        # result = call_external_api(prep_res)
        result = "Success"
        return result

    def exec_fallback(self, prep_res, exc):
        """Graceful degradation"""
        print(f"Primary method failed: {exc}")
        # Return cached/default result
        return "Fallback result"

    def post(self, shared, prep_res, exec_res):
        shared["result"] = exec_res
        return "default"


# ============================================================
# Usage Examples
# ============================================================

if __name__ == "__main__":
    print("Common PocketFlow Patterns")
    print("="*50)

    # Example 1: Simple workflow
    print("\n1. Simple Sequential Workflow")
    shared1 = {"source_path": "data.txt"}
    # simple_flow.run(shared1)

    # Example 2: Batch processing
    print("\n2. Batch Processing")
    shared2 = {"large_text": "..." * 1000}
    # batch_node = ChunkAndSummarize()
    # batch_node.run(shared2)

    # Example 3: Agent
    print("\n3. Agent with Decision Making")
    shared3 = {"query": "Search for PocketFlow"}
    # agent_flow.run(shared3)

    # Example 4: RAG
    print("\n4. RAG Pattern")
    shared4 = {
        "documents": ["doc1", "doc2", "doc3"],
        "query": "What is PocketFlow?"
    }
    # rag_indexing_flow.run(shared4)
    # rag_query_flow.run(shared4)

    print("\n✅ All patterns defined!")
    print("Uncomment the flow.run() calls to execute")
