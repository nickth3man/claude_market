# PocketFlow Cookbook Guide

Complete guide to the 47 real-world examples from the official PocketFlow cookbook.

**Source:** https://github.com/The-Pocket/PocketFlow/tree/main/cookbook

---

## 📚 Included Examples (6 Complete Implementations)

This skill includes 6 fully-functional cookbook examples in `assets/examples/`:

### 1. Chat Bot (☆☆☆ Dummy)
**File:** `01_chat.py`

Interactive chat with conversation history.
- Self-looping node for continuous interaction
- Message history management
- Graceful exit handling

**Run it:**
```bash
cd assets/examples/
python 01_chat.py
```

---

### 2. Article Writing Workflow (☆☆☆ Dummy)
**File:** `02_workflow.py`

Multi-step content creation pipeline.
- Generate outline
- Write draft
- Refine and polish

**Run it:**
```bash
python 02_workflow.py "Your Topic Here"
```

---

### 3. Research Agent (☆☆☆ Dummy)
**File:** `03_agent.py`

Agent with web search and decision-making.
- Dynamic action selection
- Branching logic (search vs answer)
- Iterative research loop

**Run it:**
```bash
python 03_agent.py "Who won the Nobel Prize 2024?"
```

---

### 4. RAG System (☆☆☆ Dummy)
**File:** `04_rag.py`

Complete retrieval-augmented generation.
- Offline: Document embedding and indexing
- Online: Query processing and answer generation
- Context-based responses

**Run it:**
```bash
python 04_rag.py --"How to install PocketFlow?"
```

---

### 5. Structured Output Parser (☆☆☆ Dummy)
**File:** `05_structured_output.py`

Resume parser with YAML output.
- Structured LLM responses
- Schema validation
- Skill matching with indexes

**Run it:**
```bash
python 05_structured_output.py
```

---

### 6. Multi-Agent Game (★☆☆ Beginner)
**File:** `06_multi_agent.py`

Two async agents playing Taboo.
- Async message queues
- Inter-agent communication
- Game logic with termination

**Run it:**
```bash
python 06_multi_agent.py
```

---

## 🗺️ Full Cookbook Index (47 Examples)

### Dummy Level (☆☆☆) - Foundational Patterns

| Example | Description | Included |
|---------|-------------|----------|
| **Chat** | Basic chat bot with history | ✅ `01_chat.py` |
| **Structured Output** | Extract data with YAML | ✅ `05_structured_output.py` |
| **Workflow** | Multi-step article writing | ✅ `02_workflow.py` |
| **Agent** | Research agent with search | ✅ `03_agent.py` |
| **RAG** | Simple retrieval-augmented generation | ✅ `04_rag.py` |
| **Map-Reduce** | Batch processing pattern | 📖 [GitHub](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-mapreduce) |
| **Streaming** | Real-time LLM streaming | 📖 [GitHub](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-streaming) |
| **Chat Guardrail** | Travel advisor with filtering | 📖 [GitHub](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-chat-guardrail) |

### Beginner Level (★☆☆) - Intermediate Patterns

| Example | Description | Included |
|---------|-------------|----------|
| **Multi-Agent** | Async agents (Taboo game) | ✅ `06_multi_agent.py` |
| **Supervisor** | Research supervision | 📖 [GitHub](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-supervisor) |
| **Parallel (3x)** | 3x speedup with parallel | 📖 [GitHub](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-parallel) |
| **Parallel (8x)** | 8x speedup demonstration | 📖 [GitHub](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-parallel-flow) |
| **Thinking** | Chain-of-Thought reasoning | 📖 [GitHub](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-thinking) |
| **Memory** | Short & long-term memory | 📖 [GitHub](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-memory) |
| **MCP** | Model Context Protocol | 📖 [GitHub](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-mcp) |
| **Tracing** | Execution visualization | 📖 [GitHub](https://github.com/The-Pocket/PocketFlow/tree/main/cookbook/pocketflow-tracing) |

### Additional Examples (47 total)

Browse the complete cookbook on GitHub for all patterns including:

**Core Patterns:**
- Node basics, Communication, Batch operations (Node, Flow, Standard)
- Async basics, Nested batches, Hello World, Majority vote

**Integrations:**
- FastAPI (background, HITL, WebSocket)
- Gradio HITL, Streamlit, Google Calendar

**Tools:**
- Web crawler, Database, Embeddings, PDF Vision, Search

**Advanced:**
- Code generator, Text-to-SQL, Voice chat
- A2A (Agent-to-Agent), Website chatbot

**Full List:** https://github.com/The-Pocket/PocketFlow/tree/main/cookbook

---

## 🎓 Learning Path

### Step 1: Start with Dummy Level
1. **01_chat.py** - Learn self-looping and state management
2. **02_workflow.py** - Understand sequential flows
3. **03_agent.py** - See branching and decision-making
4. **04_rag.py** - Multi-stage pipelines (offline + online)
5. **05_structured_output.py** - Structured LLM responses

### Step 2: Progress to Beginner Level
6. **06_multi_agent.py** - Async communication between agents

### Step 3: Explore GitHub Cookbook
- Browse all 47 examples for advanced patterns
- Find examples matching your use case
- Study progressively more complex implementations

---

## 💡 How to Use These Examples

### Run Locally
```bash
cd assets/examples/

# Make sure you have pocketflow installed
pip install pocketflow

# Run any example
python 01_chat.py
python 02_workflow.py "My Topic"
python 03_agent.py "My Question"
```

### Modify for Your Needs
1. Copy example to your project
2. Implement `call_llm()` in a utils.py file
3. Customize prompts and logic
4. Add your business requirements

### Learn Patterns
- Study the code structure
- See how nodes are connected
- Understand shared store usage
- Learn error handling approaches

---

## 🛠️ Python Template

Use the official Python template as your starting point:

**Location:** `assets/template/`

**Files:**
- `main.py` - Entry point
- `flow.py` - Flow definition
- `nodes.py` - Node implementations
- `utils.py` - LLM wrappers
- `requirements.txt` - Dependencies

**Quick Start:**
```bash
cd assets/template/
pip install -r requirements.txt

# Edit utils.py to add your LLM provider
# Then run:
python main.py
```

---

## 📖 Additional Resources

- **Official Docs:** https://the-pocket.github.io/PocketFlow/
- **GitHub Repo:** https://github.com/The-Pocket/PocketFlow
- **Full Cookbook:** https://github.com/The-Pocket/PocketFlow/tree/main/cookbook
- **Python Template:** https://github.com/The-Pocket/PocketFlow-Template-Python

---

## 🎯 Quick Reference: Which Example for What?

| Need | Use Example |
|------|-------------|
| Interactive chat | `01_chat.py` |
| Content generation pipeline | `02_workflow.py` |
| Decision-making agent | `03_agent.py` |
| Document Q&A | `04_rag.py` |
| Parse/extract data | `05_structured_output.py` |
| Multiple agents | `06_multi_agent.py` |
| Batch processing | Map-Reduce (GitHub) |
| Real-time streaming | Streaming (GitHub) |
| Memory/context | Memory (GitHub) |
| Parallel speedup | Parallel examples (GitHub) |

---

## ✅ Next Steps

1. **Pick an example** that matches your use case
2. **Run it** to see how it works
3. **Study the code** to understand patterns
4. **Copy and modify** for your project
5. **Implement** your LLM provider
6. **Iterate** and improve!

---

*This guide covers the 6 included examples plus references to all 47 cookbook patterns. All examples are production-ready and demonstrate PocketFlow best practices.*
