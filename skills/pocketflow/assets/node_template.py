"""
PocketFlow Node Template

Copy this template and customize for your needs
"""

from pocketflow import Node
# from utils.call_llm import call_llm  # Uncomment if using LLM


class TemplateNode(Node):
    """
    Brief description of what this node does

    Shared Store Schema:
        Input:
            - key1 (type): description
            - key2 (type): description

        Output:
            - result_key (type): description

    Actions:
        - "default": Normal flow
        - "error": If something goes wrong
        - "retry": If needs retry
    """

    def prep(self, shared):
        """
        Prepare data from shared store

        Args:
            shared (dict): Shared data store

        Returns:
            Any: Data to pass to exec()
        """
        # TODO: Get data from shared store
        input_data = shared.get("input_key")

        # Optional: Add validation
        if not input_data:
            raise ValueError("Missing required input")

        return input_data

    def exec(self, prep_res):
        """
        Execute the main logic (can fail and retry)

        Args:
            prep_res: Data from prep()

        Returns:
            Any: Result to pass to post()
        """
        # TODO: Implement your logic here

        # Example: Call LLM
        # result = call_llm(f"Process: {prep_res}")

        # Example: Process data
        result = f"Processed: {prep_res}"

        return result

    def post(self, shared, prep_res, exec_res):
        """
        Save results and return action

        Args:
            shared (dict): Shared data store
            prep_res: Original data from prep()
            exec_res: Result from exec()

        Returns:
            str: Action name for flow control
        """
        # TODO: Save results to shared store
        shared["result_key"] = exec_res

        # Optional: Conditional actions
        # if some_condition:
        #     return "special_action"

        return "default"

    def exec_fallback(self, prep_res, exc):
        """
        Optional: Handle errors gracefully

        Args:
            prep_res: Data from prep()
            exc: The exception that occurred

        Returns:
            Any: Fallback result (passed to post as exec_res)
        """
        # TODO: Implement fallback logic
        print(f"Error occurred: {exc}")

        # Option 1: Re-raise the exception
        # raise exc

        # Option 2: Return fallback value
        return "Fallback result"


# Example usage
if __name__ == "__main__":
    # Create node with retry settings
    node = TemplateNode(max_retries=3, wait=5)

    # Create shared store
    shared = {
        "input_key": "test input"
    }

    # Run node
    action = node.run(shared)

    print(f"Action: {action}")
    print(f"Result: {shared.get('result_key')}")
