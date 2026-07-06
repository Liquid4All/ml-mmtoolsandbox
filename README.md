# MM-ToolSandbox: A Unified Framework for Evaluating Visual Tool-Calling Agents

<div align='center'>

[**📖 Paper**](https://arxiv.org/abs/2607.11818)

</div>

![MM-ToolSandbox](teaser.png)

We propose MM-ToolSandbox, a benchmark and an evaluation framework for visually grounded tool-calling agents. It provides a stateful execution environment spanning 500+ tools across 16 application domains and poses multi-image, multi-turn tasks in which an agent must ground progressively arriving visual inputs into executable tool calls while handling realistic conversational phenomena, such as goal revisions, error corrections, and state mutations. The environment supports both tool-use and code-execution interfaces, and the benchmark comprises 258 high-quality, human-verified scenarios, evaluated with rubric-based LLM and static entity-diff frameworks.

## Setup

### Prerequisites

- Python **3.11** (3.12 also supported)
- [`uv`](https://docs.astral.sh/uv/) (recommended) or `pip`

### 1. Install MM-ToolSandbox

```bash
git clone https://github.com/apple/ml-mmtoolsandbox
cd ml-mmtoolsandbox

# Create environment
uv venv --python 3.11
source .venv/bin/activate

# Install with development extras (pytest, mypy, ruff, ...)
uv pip install -e ".[dev]"
```

### 2. Install AppWorld
Clone [AppWorld](https://github.com/StonyBrookNLP/appworld) (Trivedi et al., 2024) as a **sibling directory**:

```bash
# From ml-mmtoolsandbox
cd ..
git clone https://github.com/StonyBrookNLP/appworld.git
cd appworld
uv pip install -e .
appworld install --repo
appworld download data
cd ../ml-mmtoolsandbox
```

MM-ToolSandbox auto-discovers AppWorld at `../appworld/`.  Set the `APPWORLD_ROOT` environment variable to override the location.

### 3. Download scenario images

Visual scenarios reference images sourced from diverse public vision-language datasets.  Downloading and processing them takes **~7 GB** of disk space and may take a while on a slow connection:

```bash
python download_images.py
```

By default the images are written to `data/`.  The script also produces an index JSONL that scenario loaders use to resolve `image_paths` references.

> **Note:** the HierText download fetches from S3 via the `aws` CLI, so install [`awscli`](https://aws.amazon.com/cli/) first if you don't have it.

### 4. Configure model API keys
Export the matching API key for each provider whose models you plan to use:
```bash
export OPENAI_API_KEY="..."
export ANTHROPIC_API_KEY="..."
```

Additionally provide an API key for the [Serper Google Search API](https://serper.dev):
```bash
export SERPER_API_KEY="..."
```

### 5. Verify

```bash
python -c "from mmtoolsandbox.appworld import APPWORLD_AVAILABLE; print('AppWorld:', APPWORLD_AVAILABLE)"
# Expected: AppWorld: True

```

## Quick Start

The benchmark ships with two scenario sets under [`scenarios/`](scenarios/):

- [`scenarios/nominal/`](scenarios/nominal/) — 258 multi-app, multi-turn task scenarios (the FULL benchmark).
- [`scenarios/ui/`](scenarios/ui/) — 50 UI-rendering scenarios that exercise the A2UI interactive surface.

### Example 1: Tool search + coding tool (function-calling mode)

The agent calls APIs through the model's native function-calling interface, with `api_docs_search_api_docs` for dynamic tool discovery and `execute_code` for ad-hoc Python:

```bash
mmtoolsandbox -d FULL \
    --dataset-config '{"scenario_dir": "scenarios/nominal", "auto_login": true}' \
    --agent GPT_5_4_2026_03_05_Reasoning_Agent \
    --user GPT_5_4_2026_03_05_User \
    --judge Claude_4_5_Sonnet_Judge \
    --enable-tool-search --enable-coding-tool --image-input \
    --image-base-path data \
    --output-dir runs/gpt54_tooluse \
    --parallel 16
```

### Example 2: Code execution mode

The agent writes Python directly in a persistent REPL with all tools pre-loaded as Python functions; tool discovery happens via `api_docs_search_api_docs` inside the code:

```bash
mmtoolsandbox -d FULL \
    --dataset-config '{"scenario_dir": "scenarios/nominal", "auto_login": true}' \
    --agent GPT_5_4_2026_03_05_Reasoning_CodeExec_Agent \
    --user GPT_5_4_2026_03_05_User \
    --judge Claude_4_5_Sonnet_Judge \
    --code-execution-mode --image-input \
    --image-base-path data \
    --output-dir runs/gpt54_codeexec \
    --parallel 16
```

Results are written to `<output-dir>/trajectories/<scenario_name>/`.

## Visualizing Results

```bash
python -m mmtoolsandbox.viz.visualize_results runs/<RUN_NAME> --scenario-dir scenarios/nominal
# Open http://localhost:8000 in your browser
```

## Package Structure

```
mmtoolsandbox/
  cli/          Command-line interface
  common/       Core framework (execution context, evaluation, entity diff,
                scenario, safety guard)
  datasets/     Unified scenario factory (FULL + MEDIUM) and scenario
                definitions
  tools/        Tool implementations (vision, tool_sandbox, appworld)
  toolbox/      Toolbox loading and management
  roles/        Agent, user, judge, and execution environment implementations
  appworld/     AppWorld runtime infrastructure (bridge, state, scenario
                runner)
  a2ui/         A2UI interactive UI framework (state, renderer, tools)
  viz/          Trajectory visualization web server
```

## Citation
If you find MM-ToolSandbox useful in your research, please cite:

```bibtex
@misc{ma2026mmtoolsandbox,
      title={MM-ToolSandbox: A Unified Framework for Evaluating Visual Tool-Calling Agents},
      author={Ma, Kaixin and Feng, Di and Metz, Alexander and Lu, Jiarui and Verma, Eshan and Dehghan, Afshin},
      year={2026},
      eprint={2607.11818},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2607.11818},
}
```

This work builds on the original ToolSandbox benchmark:

> Lu et al. *ToolSandbox: A Stateful, Conversational, Interactive Evaluation Benchmark for LLM Tool Use Capabilities.* arXiv:2408.04682, 2024. <https://arxiv.org/abs/2408.04682>

And the AppWorld simulated-app environment:

> Trivedi et al. *AppWorld: A Controllable World of Apps and People for Benchmarking Interactive Coding Agents.* ACL 2024. <https://github.com/StonyBrookNLP/appworld>

The interactive UI generation pipeline is built on the **A2UI** protocol — a declarative, component-based UI framework for agent-driven interfaces.  See <https://a2ui.org> for the spec and reference implementations.

## License
Released under the **Apple License**.  Use, modification, and redistribution are permitted subject to the terms in [LICENSE.md](LICENSE.md), including the attribution requirement and the restriction against using the Apple name or trademarks for endorsement.  The software is provided AS IS, with no warranties.
