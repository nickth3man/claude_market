#!/usr/bin/env python3
"""
PocketFlow Project Initializer
Creates a new PocketFlow project with best-practice structure
"""

import os
import sys

def create_project(project_name):
    """Create a new PocketFlow project structure"""

    # Create directories
    dirs = [
        f"{project_name}/nodes",
        f"{project_name}/flows",
        f"{project_name}/utils",
        f"{project_name}/tests",
        f"{project_name}/docs"
    ]

    for d in dirs:
        os.makedirs(d, exist_ok=True)
        # Create __init__.py for Python packages
        if d.endswith(('nodes', 'flows', 'utils', 'tests')):
            open(f"{d}/__init__.py", 'w').close()

    # Create main.py
    with open(f"{project_name}/main.py", 'w') as f:
        f.write('''#!/usr/bin/env python3
"""
Main entry point for {name}
"""

from flows.my_flow import MyFlow

def main():
    shared = {{
        "input": "Hello, PocketFlow!",
    }}

    flow = MyFlow()
    flow.run(shared)

    print(f"Result: {{shared.get('result')}}")

if __name__ == "__main__":
    main()
'''.format(name=project_name))

    # Create example LLM utility
    with open(f"{project_name}/utils/call_llm.py", 'w') as f:
        f.write('''"""
LLM wrapper - customize for your provider
"""

def call_llm(prompt):
    """Call your LLM provider"""
    # TODO: Implement your LLM call
    # Example for OpenAI:
    # from openai import OpenAI
    # client = OpenAI(api_key="YOUR_API_KEY")
    # response = client.chat.completions.create(
    #     model="gpt-4o",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content

    raise NotImplementedError("Implement your LLM provider")
''')

    # Create example node
    with open(f"{project_name}/nodes/my_node.py", 'w') as f:
        f.write('''"""
Example node implementation
"""

from pocketflow import Node
from utils.call_llm import call_llm

class ProcessNode(Node):
    """Example processing node"""

    def prep(self, shared):
        """Get input from shared store"""
        return shared.get("input", "")

    def exec(self, prep_res):
        """Process with LLM"""
        prompt = f"Process this: {prep_res}"
        result = call_llm(prompt)
        return result

    def post(self, shared, prep_res, exec_res):
        """Store result"""
        shared["result"] = exec_res
        return "default"
''')

    # Create example flow
    with open(f"{project_name}/flows/my_flow.py", 'w') as f:
        f.write('''"""
Example flow implementation
"""

from pocketflow import Flow
from nodes.my_node import ProcessNode

class MyFlow(Flow):
    """Example flow"""

    def __init__(self):
        # Create nodes
        process = ProcessNode()

        # Define flow
        # process >> next_node  # Add more nodes as needed

        # Initialize flow
        super().__init__(start=process)
''')

    # Create requirements.txt
    with open(f"{project_name}/requirements.txt", 'w') as f:
        f.write('''# PocketFlow dependencies
pocketflow

# LLM providers (uncomment what you need)
# openai
# anthropic
# google-generativeai

# Optional utilities
# beautifulsoup4
# requests
# faiss-cpu
''')

    # Create README
    with open(f"{project_name}/README.md", 'w') as f:
        f.write(f'''# {project_name}

PocketFlow project for [describe your use case]

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Configure your LLM provider
# Edit utils/call_llm.py

# Run
python main.py
```

## Project Structure

```
{project_name}/
├── main.py              # Entry point
├── nodes/               # Node implementations
├── flows/               # Flow definitions
├── utils/               # Utilities (LLM, DB, etc.)
├── tests/               # Unit tests
└── docs/                # Documentation
```

## Next Steps

1. Implement your LLM wrapper in `utils/call_llm.py`
2. Create your nodes in `nodes/`
3. Define your flow in `flows/`
4. Run and test!
''')

    # Create design doc template
    with open(f"{project_name}/docs/design.md", 'w') as f:
        f.write(f'''# {project_name} Design

## Problem Statement

What problem are you solving?

## Solution Overview

High-level approach using PocketFlow

## Flow Architecture

```mermaid
flowchart LR
    start[Start] --> process[Process]
    process --> end[End]
```

## Data Schema

```python
shared = {{
    "input": "...",
    "intermediate": "...",
    "result": "..."
}}
```

## Nodes

### Node 1: ProcessNode
- **Purpose:** What does it do?
- **Input:** What does it need from shared?
- **Output:** What does it produce?
- **Actions:** What actions can it return?

## Error Handling

How will you handle failures?

## Testing Strategy

How will you test this?
''')

    print(f"✅ Created PocketFlow project: {project_name}/")
    print(f"📁 Structure:")
    print(f"   ├── main.py")
    print(f"   ├── nodes/my_node.py")
    print(f"   ├── flows/my_flow.py")
    print(f"   ├── utils/call_llm.py")
    print(f"   ├── requirements.txt")
    print(f"   └── docs/design.md")
    print(f"\n🚀 Next steps:")
    print(f"   1. cd {project_name}")
    print(f"   2. Edit utils/call_llm.py (add your LLM API key)")
    print(f"   3. python main.py")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pocketflow_init.py <project_name>")
        sys.exit(1)

    create_project(sys.argv[1])
