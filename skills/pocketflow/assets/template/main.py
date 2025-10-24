"""
PocketFlow Template - Main Entry Point

Source: https://github.com/The-Pocket/PocketFlow-Template-Python

This template demonstrates best practices for structuring a PocketFlow project.
"""

from flow import create_qa_flow


def main():
    """Main entry point for the application"""

    # Prepare shared data store
    shared = {
        "question": "In one sentence, what's the end of universe?",
        "answer": None
    }

    # Create and run the flow
    qa_flow = create_qa_flow()
    qa_flow.run(shared)

    # Display results
    print(f"\n{'='*60}")
    print("Results:")
    print(f"{'='*60}")
    print(f"Question: {shared['question']}")
    print(f"Answer: {shared['answer']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
