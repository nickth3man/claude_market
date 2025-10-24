# claude_market

Claude Code plugin marketplace named "claude_market" with an included plugin "pocketflow" (a Skill + Python examples).

- start_here/marketplace_guide.md — Marketplaces overview and schema
- start_here/plugins_guide.md — Building and testing plugins
- start_here/plugins_reference.md — Technical reference (schemas, components)
- start_here/settings.md — User and repo configuration

## Current repository contents

```
claude_market/
├── .claude-plugin/
│   ├── marketplace.json        # Marketplace (name: "claude_market")
│   └── plugin.json             # Plugin manifest (name: "pocketflow")
├── start_here/
│   ├── marketplace_guide.md
│   ├── plugins_guide.md
│   ├── plugins_reference.md
│   └── settings.md
├── skills/
│   └── pocketflow/
│       ├── SKILL.md
│       ├── assets/
│       │   ├── COOKBOOK_GUIDE.md
│       │   ├── common_patterns.py
│       │   ├── examples/
│       │   │   ├── 01_chat.py
│       │   │   ├── 02_workflow.py
│       │   │   ├── 03_agent.py
│       │   │   ├── 04_rag.py
│       │   │   ├── 05_structured_output.py
│       │   │   └── 06_multi_agent.py
│       │   ├── flow_template.py
│       │   ├── node_template.py
│       │   └── template/
│       │       ├── README.md
│       │       ├── flow.py
│       │       ├── main.py
│       │       ├── nodes.py
│       │       ├── requirements.txt
│       │       └── utils.py
│       ├── references/
│       │   ├── core_abstraction.md
│       │   └── index.md
│       └── scripts/
│           ├── pocketflow_init.py
│           └── test_llm_connection.py
└── README.md
```


## Marketplace manifest

Already provided at `.claude-plugin/marketplace.json`:

```json
{
  "name": "claude_market",
  "owner": { "name": "claude_market" },
  "metadata": {
    "description": "Claude Code marketplace for PocketFlow and future plugins",
    "version": "0.1.0"
  },
  "plugins": [
    {
      "name": "pocketflow",
      "source": ".",
      "description": "PocketFlow Skill packaged as a plugin",
      "version": "0.1.0",
      "category": "skills",
      "keywords": ["pocketflow", "skills", "agents", "rag", "workflow"]
    }
  ]
}
```

Notes:
- Marketplace name is `claude_market`.
- You can add more plugins later by appending entries to `plugins` with `source` pointing to subfolders or external repos.

## Included plugin: pocketflow

This repository is also a plugin (manifest at `.claude-plugin/plugin.json`) that exposes the Skill at `skills/pocketflow/`.

```json
{
  "name": "pocketflow",
  "version": "0.1.0",
  "description": "PocketFlow Skill, cookbook examples, and templates for graph-based LLM workflows.",
  "author": { "name": "claude_market" },
  "keywords": ["skill", "pocketflow", "agents", "rag", "workflow"]
}
```

## Add this marketplace in Claude Code

Add the marketplace:

```shell
/plugin marketplace add ./
/plugin marketplace list
/plugin
```

Install the plugin from this marketplace:

```shell
/plugin install pocketflow@claude_market
```

Other useful commands:

```shell
/plugin enable plugin-name@claude_market
/plugin disable plugin-name@claude_market
/plugin uninstall plugin-name@claude_market
/plugin marketplace update claude_market
/plugin marketplace remove claude_market
```

## Team setup via settings

Add repository-level config in `.claude/settings.json` so teammates auto-discover this marketplace and specific plugins:

```json
{
  "enabledPlugins": {
    "pocketflow@claude_market": true
  },
  "extraKnownMarketplaces": {
    "claude_market": {
      "source": {
        "source": "github",
        "repo": "owner/claude_market"
      }
    }
  }
}
```

See `start_here/settings.md` for full options (git URL, local directory sources, enabling/disabling plugins, scopes).

## Use the PocketFlow examples locally

The PocketFlow Skill includes runnable Python examples and a project template under `skills/pocketflow/assets/`:

```shell
# optional: create a virtual environment first
pip install -r skills/pocketflow/assets/template/requirements.txt
python skills/pocketflow/assets/examples/01_chat.py
```

## References

- start_here/marketplace_guide.md — Managing and hosting marketplaces
- start_here/plugins_guide.md — Building and testing plugins
- start_here/plugins_reference.md — Schemas and component specs
- start_here/settings.md — Settings for marketplaces and plugins
