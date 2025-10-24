"""
PocketFlow Flow Template

Copy this template and customize for your workflow
"""

from pocketflow import Flow, Node
# from nodes.my_nodes import Node1, Node2, Node3  # Import your nodes


class TemplateFlow(Flow):
    """
    Brief description of what this flow does

    Flow Architecture:
        node1 >> node2 >> node3
        node2 - "special" >> node4

    Shared Store Schema:
        Input:
            - input_data (str): Initial input

        Intermediate:
            - step1_result (str): Result from node1
            - step2_result (str): Result from node2

        Output:
            - final_result (str): Final output
    """

    def __init__(self):
        """Initialize the flow with nodes and connections"""

        # TODO: Create your nodes
        node1 = Node1()
        node2 = Node2()
        node3 = Node3()

        # TODO: Define flow connections

        # Simple sequence
        node1 >> node2 >> node3

        # Branching (conditional)
        # node2 - "error" >> error_handler
        # node2 - "success" >> node3

        # Looping
        # node3 - "retry" >> node1

        # Initialize with start node
        super().__init__(start=node1)


# Example with actual implementation
class SimpleWorkflow(Flow):
    """Example: Simple 3-step workflow"""

    def __init__(self):
        # Step 1: Load data
        load = LoadNode()

        # Step 2: Process
        process = ProcessNode()

        # Step 3: Save
        save = SaveNode()

        # Connect
        load >> process >> save

        super().__init__(start=load)


class ConditionalWorkflow(Flow):
    """Example: Workflow with branching"""

    def __init__(self):
        # Create nodes
        validate = ValidateNode()
        process_valid = ProcessValidNode()
        process_invalid = ProcessInvalidNode()
        finalize = FinalizeNode()

        # Branching based on validation
        validate - "valid" >> process_valid
        validate - "invalid" >> process_invalid

        # Both paths lead to finalize
        process_valid >> finalize
        process_invalid >> finalize

        super().__init__(start=validate)


class LoopingWorkflow(Flow):
    """Example: Workflow with retry loop"""

    def __init__(self):
        # Create nodes
        attempt = AttemptNode()
        verify = VerifyNode()
        finish = FinishNode()

        # Setup loop
        attempt >> verify

        # Branching: success or retry
        verify - "success" >> finish
        verify - "retry" >> attempt  # Loop back

        # Optional: max attempts check
        verify - "failed" >> finish

        super().__init__(start=attempt)


class NestedWorkflow(Flow):
    """Example: Flow containing sub-flows"""

    def __init__(self):
        # Create sub-flows
        preprocessing_flow = PreprocessFlow()
        processing_flow = ProcessFlow()
        postprocessing_flow = PostprocessFlow()

        # Connect sub-flows
        preprocessing_flow >> processing_flow >> postprocessing_flow

        super().__init__(start=preprocessing_flow)


# Example usage
if __name__ == "__main__":
    # Create flow
    flow = SimpleWorkflow()

    # Prepare shared store
    shared = {
        "input_data": "Hello, PocketFlow!"
    }

    # Run flow
    flow.run(shared)

    # Check results
    print(f"Final result: {shared.get('final_result')}")
