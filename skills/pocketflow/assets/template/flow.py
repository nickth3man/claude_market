"""
PocketFlow Template - Flow Definition

Source: https://github.com/The-Pocket/PocketFlow-Template-Python

This module defines the QA flow by connecting nodes.
"""

from pocketflow import Flow
from nodes import GetQuestionNode, AnswerNode


def create_qa_flow():
    """
    Create a simple Question-Answer flow

    Flow structure:
        GetQuestionNode >> AnswerNode

    Returns:
        Flow: Configured QA flow
    """
    # Create nodes
    get_question_node = GetQuestionNode()
    answer_node = AnswerNode()

    # Connect nodes sequentially
    get_question_node >> answer_node

    # Create flow with start node
    qa_flow = Flow(start=get_question_node)

    return qa_flow


# For direct module execution
qa_flow = create_qa_flow()
