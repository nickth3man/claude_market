# PocketFlow Project Template

This template provides a best-practice structure for PocketFlow projects.

Source: https://github.com/The-Pocket/PocketFlow-Template-Python

## Project Structure

```
template/
├── main.py              # Entry point
├── flow.py              # Flow definition
├── nodes.py             # Node implementations
├── utils.py             # Utility functions (LLM wrappers, etc.)
└── requirements.txt     # Python dependencies
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure your LLM:**
   Edit `utils.py` and implement `call_llm()` for your provider (OpenAI, Anthropic, or Gemini)

3. **Set API key:**
   ```bash
   export OPENAI_API_KEY=sk-...
   # or
   export ANTHROPIC_API_KEY=sk-ant-...
   # or
   export GEMINI_API_KEY=...
   ```

4. **Run:**
   ```bash
   python main.py
   ```

## Customization

- **Add nodes:** Create new node classes in `nodes.py`
- **Modify flow:** Update connections in `flow.py`
- **Add utilities:** Implement helpers in `utils.py`
- **Update logic:** Customize `main.py` for your use case

## Best Practices Demonstrated

1. **Separation of Concerns:**
   - `nodes.py` - Node logic only
   - `flow.py` - Flow orchestration only
   - `utils.py` - Reusable utilities
   - `main.py` - Application entry point

2. **Factory Pattern:**
   - `create_qa_flow()` makes flow reusable
   - Easy to test and modify

3. **Clear Data Flow:**
   - Shared store pattern for data passing
   - Explicit state management

4. **Configuration:**
   - Environment variables for API keys
   - requirements.txt for dependencies

## Next Steps

1. Implement your `call_llm()` function
2. Add your business logic to nodes
3. Define your workflow in flow.py
4. Run and iterate!

## Resources

- **PocketFlow Docs:** https://the-pocket.github.io/PocketFlow/
- **GitHub:** https://github.com/The-Pocket/PocketFlow
- **Examples:** See the cookbook/ directory for more patterns
