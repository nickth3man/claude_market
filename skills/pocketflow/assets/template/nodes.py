"""
PocketFlow Template - Node Definitions

Source: https://github.com/The-Pocket/PocketFlow-Template-Python

This module contains the node definitions for the QA flow.
Each node implements the prep/exec/post pattern.
"""

from pocketflow import Node
# from utils import call_llm  # Uncomment when implemented


class GetQuestionNode(Node):
    """Node to get user input"""

    def prep(self, shared):
        """Prepare: can access shared store but no data needed"""
        return None

    def exec(self, prep_res):
        """Execute: get user input"""
        question = input("\nEnter your question: ")
        return question

    def post(self, shared, prep_res, exec_res):
        """Post: store question in shared store"""
        shared["question"] = exec_res
        print(f"✓ Question received: {exec_res}")
        return "default"


class AnswerNode(Node):
    """Node to generate answer using LLM"""

    def prep(self, shared):
        """Prepare: get question from shared store"""
        return shared.get("question", "")

    def exec(self, question):
        """Execute: call LLM to get answer"""
        if not question:
            return "No question provided"

        # Call your LLM implementation
        # answer = call_llm(question)

        # Placeholder
        answer = f"This is a placeholder answer to: {question}\nImplement call_llm() in utils.py"
        return answer

    def post(self, shared, prep_res, exec_res):
        """Post: store answer in shared store"""
        shared["answer"] = exec_res
        print(f"✓ Answer generated ({len(exec_res)} chars)")
        return "default"
