"""
PocketFlow Cookbook Example: Research Agent

Difficulty: ☆☆☆ Dummy Level
Source: https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-agent

Description:
A research agent that can search the web and answer questions.
Demonstrates:
- Agent pattern with dynamic action selection
- Branching based on decisions
- Loop-back for iterative research
- Tool usage (web search)
"""

from pocketflow import Node, Flow
# from utils import call_llm, search_web  # You need to implement these


class DecideActionNode(Node):
    """Agent decides whether to search or answer"""

    def prep(self, shared):
        return {
            "question": shared["question"],
            "context": shared.get("context", "No information gathered yet")
        }

    def exec(self, inputs):
        """Decide next action using LLM"""
        question = inputs["question"]
        context = inputs["context"]

        prompt = f"""
Given:
Question: {question}
Current Context: {context}

Should I:
1. Search web for more information
2. Answer with current knowledge

Output in format:
Action: search/answer
Reasoning: [why]
Search Query: [if action is search]
"""
        # response = call_llm(prompt)
        # Parse response to get action

        # Placeholder logic
        if not context or "No information" in context:
            action = "search"
            search_query = question
        else:
            action = "answer"
            search_query = None

        print(f"\n🤔 Agent decided: {action}")

        return {
            "action": action,
            "search_query": search_query
        }

    def post(self, shared, prep_res, exec_res):
        shared["decision"] = exec_res
        # Branch based on action
        return exec_res["action"]


class SearchWebNode(Node):
    """Search the web for information"""

    def prep(self, shared):
        return shared["decision"]["search_query"]

    def exec(self, query):
        """Perform web search"""
        print(f"\n🔍 Searching: {query}")
        # results = search_web(query)
        results = f"Search results for '{query}':\n- Result 1\n- Result 2\n- Result 3"
        return results

    def post(self, shared, prep_res, exec_res):
        # Add to context
        current_context = shared.get("context", "")
        shared["context"] = current_context + "\n\n" + exec_res
        print(f"\n📚 Context updated ({len(shared['context'])} chars)")
        # Loop back to decide again
        return "continue"


class AnswerNode(Node):
    """Generate final answer"""

    def prep(self, shared):
        return {
            "question": shared["question"],
            "context": shared.get("context", "")
        }

    def exec(self, inputs):
        """Generate answer from context"""
        prompt = f"""
Context: {inputs['context']}

Question: {inputs['question']}

Provide a comprehensive answer:
"""
        # answer = call_llm(prompt)
        answer = f"Based on the research, here's the answer to '{inputs['question']}':\n\n[Answer based on context]"
        return answer

    def post(self, shared, prep_res, exec_res):
        shared["final_answer"] = exec_res
        print(f"\n✅ Answer generated")
        return "done"


# Build the agent flow
def create_agent_flow():
    """Create research agent with branching and looping"""
    decide = DecideActionNode()
    search = SearchWebNode()
    answer = AnswerNode()

    # Branching: decide can lead to search or answer
    decide - "search" >> search
    decide - "answer" >> answer

    # Loop: search leads back to decide
    search - "continue" >> decide

    flow = Flow(start=decide)
    return flow


# Example usage
def main():
    """Run the research agent"""
    # Default question
    question = "Who won the Nobel Prize in Physics 2024?"

    # Get question from command line if provided
    import sys
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])

    shared = {"question": question}

    print(f"\n🤔 Processing question: {question}")
    print("="*50)

    flow = create_agent_flow()
    flow.run(shared)

    print("\n" + "="*50)
    print("\n🎯 Final Answer:")
    print(shared.get("final_answer", "No answer found"))


if __name__ == "__main__":
    main()
