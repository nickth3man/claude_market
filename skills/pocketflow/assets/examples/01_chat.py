"""
PocketFlow Cookbook Example: Interactive Chat Bot

Difficulty: ☆☆☆ Dummy Level
Source: https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-chat

Description:
A basic chat bot with conversation history. Demonstrates:
- Self-looping nodes for continuous interaction
- Message history management
- User input handling
- Graceful exit conditions
"""

from pocketflow import Node, Flow
# from utils import call_llm  # You need to implement this


class ChatNode(Node):
    """Interactive chat node that maintains conversation history"""

    def prep(self, shared):
        """Get user input and maintain message history"""
        # Initialize messages if this is the first run
        if "messages" not in shared:
            shared["messages"] = []
            print("Welcome to the chat! Type 'exit' to end the conversation.")

        # Get user input
        user_input = input("\nYou: ")

        # Check if user wants to exit
        if user_input.lower() == 'exit':
            return None

        # Add user message to history
        shared["messages"].append({"role": "user", "content": user_input})

        # Return all messages for the LLM
        return shared["messages"]

    def exec(self, messages):
        """Call LLM with conversation history"""
        if messages is None:
            return None

        # Call LLM with the entire conversation history
        # response = call_llm(messages)
        response = "This is a placeholder response. Implement call_llm()."
        return response

    def post(self, shared, prep_res, exec_res):
        """Display response and continue or end conversation"""
        if prep_res is None or exec_res is None:
            print("\nGoodbye!")
            return None  # End the conversation

        # Print the assistant's response
        print(f"\nAssistant: {exec_res}")

        # Add assistant message to history
        shared["messages"].append({"role": "assistant", "content": exec_res})

        # Loop back to continue the conversation
        return "continue"


# Build the flow with self-loop
def create_chat_flow():
    """Create a chat flow that loops back to itself"""
    chat_node = ChatNode()
    chat_node - "continue" >> chat_node  # Loop back to continue conversation

    flow = Flow(start=chat_node)
    return flow


# Example usage
if __name__ == "__main__":
    shared = {}
    flow = create_chat_flow()
    flow.run(shared)

    # Conversation history is preserved in shared["messages"]
    print(f"\n\nTotal messages: {len(shared.get('messages', []))}")
