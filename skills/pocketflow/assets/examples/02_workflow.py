"""
PocketFlow Cookbook Example: Article Writing Workflow

Difficulty: ☆☆☆ Dummy Level
Source: https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-workflow

Description:
A writing workflow that outlines, writes content, and applies styling.
Demonstrates:
- Sequential multi-step workflow
- Progressive content generation
- Task decomposition pattern
"""

from pocketflow import Node, Flow
# from utils import call_llm  # You need to implement this


class GenerateOutlineNode(Node):
    """Generate article outline from topic"""

    def prep(self, shared):
        return shared["topic"]

    def exec(self, topic):
        """Create outline with LLM"""
        prompt = f"Create a detailed outline for an article about: {topic}"
        # outline = call_llm(prompt)
        outline = f"Outline for {topic}:\n1. Introduction\n2. Main Points\n3. Conclusion"
        print(f"\n📋 Outline Generated ({len(outline)} chars)")
        return outline

    def post(self, shared, prep_res, exec_res):
        shared["outline"] = exec_res
        return "default"


class WriteDraftNode(Node):
    """Write article draft from outline"""

    def prep(self, shared):
        return shared["outline"]

    def exec(self, outline):
        """Generate content based on outline"""
        prompt = f"Write content based on this outline:\n{outline}"
        # draft = call_llm(prompt)
        draft = f"Draft article based on outline:\n\n{outline}\n\n[Article content here...]"
        print(f"\n✍️  Draft Written ({len(draft)} chars)")
        return draft

    def post(self, shared, prep_res, exec_res):
        shared["draft"] = exec_res
        return "default"


class RefineArticleNode(Node):
    """Polish and refine the draft"""

    def prep(self, shared):
        return shared["draft"]

    def exec(self, draft):
        """Improve draft quality"""
        prompt = f"Review and improve this draft:\n{draft}"
        # final = call_llm(prompt)
        final = f"Refined version:\n\n{draft}\n\n[Enhanced with better flow and clarity]"
        print(f"\n✨ Article Refined ({len(final)} chars)")
        return final

    def post(self, shared, prep_res, exec_res):
        shared["final_article"] = exec_res
        print("\n✅ Article Complete!")
        return "default"


# Build the workflow
def create_article_flow():
    """Create sequential article writing workflow"""
    outline = GenerateOutlineNode()
    draft = WriteDraftNode()
    refine = RefineArticleNode()

    # Sequential pipeline
    outline >> draft >> refine

    flow = Flow(start=outline)
    return flow


# Example usage
def run_flow(topic="AI Safety"):
    """Run the article writing workflow"""
    shared = {"topic": topic}

    print(f"\n=== Starting Article Workflow: {topic} ===\n")

    flow = create_article_flow()
    flow.run(shared)

    # Output summary
    print("\n=== Workflow Statistics ===")
    print(f"Topic: {shared['topic']}")
    print(f"Outline: {len(shared['outline'])} characters")
    print(f"Draft: {len(shared['draft'])} characters")
    print(f"Final: {len(shared['final_article'])} characters")

    return shared


if __name__ == "__main__":
    import sys

    # Get topic from command line or use default
    topic = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "AI Safety"
    result = run_flow(topic)

    # Print final article
    print("\n=== Final Article ===")
    print(result["final_article"])
