# 🤖 GenAI Lab — Practical, Reusable Projects (RAG, Agents, Streamlit)

> Active repository where I’m building **stand-alone**, **reusable**, and **low-cost** Generative AI projects.
> Focused on **hybrid RAG**, **Agentic AI workflows**, **local + hosted models**, and **Streamlit UIs**.

---

## Table of Contents

* [Why this repo](#why-this-repo)
* [Key features](#key-features)
* [Repo structure](#repo-structure)
* [Quickstart](#quickstart)
* [Project suites](#project-suites)

  * [1) LangChain Projects](#1-langchain-projects)
  * [2) CrewAI Projects](#2-crewai-projects)
  * [3) LlamaIndex Projects](#3-llamaindex-projects)
* [RAG pipelines (hybrid patterns)](#rag-pipelines-hybrid-patterns)
* [Agentic workflows](#agentic-workflows)
* [Data ingestion & indexing](#data-ingestion--indexing)
* [Models & providers](#models--providers)
* [Configuration](#configuration)
* [Running the UIs (Streamlit)](#running-the-uis-streamlit)
* [Observability & evals](#observability--evals)
* [Cost control tips](#cost-control-tips)
* [Development workflow](#development-workflow)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)

---

## Why this repo

* **Practical first:** each project is runnable in isolation with minimal setup.
* **Reusable building blocks:** loaders, chunkers, embedders, retrievers, and agents are modular.
* **Low-cost by design:** prefer **local models (Ollama)** and **cheap inference**; API usage is optional; **local frontend UI (Streamlit)**.
* **Deep experiments:** hybrid RAG across **LangChain**, **LlamaIndex**, **Unstructured** pipeline, **multimodal (CLIP)**, **LangGraph**, and **CrewAI**.

---

## Key features

* 🔎 **Hybrid RAG**: ingestion + index + vector db + retrival (semantic, keyword, summary) + rerank; multi-index routing; cross-encoder rerank.
* 🧩 **Frameworks**: **LangChain**, **LlamaIndex**, **CrewAI**, **LangGraph**, **LangSmith**.
* 🖼️ **Multi-modal**: image understanding via **CLIP** or **GPT4**; text-image-table RAG demos.
* 🖥️ **Local models**: **Ollama** (e.g., `llama3.1`, `gpt`, `nomic-embed`), optional OpenAI/Anthropic/Groq.
* 🧪 **Evaluation**: LangSmith traces; offline eval prompts; Latency vs throughput.
* 🧰 **Streamlit UIs**: quick, interactive frontends.
* 📦 **Stand-alone**: each folder has its own `requirements.txt` and `README` snippet.

---

## Repo structure

```
genai-lab/
├─ langchain-projects/
│  ├─ basic-rag/             # 
│  ├─ hybrid-rag/
│  ├─ multimodal-rag/
│  ├─ langgraph-experiments/
├─ crewai-projects/
│  ├─ research-assistant/
│  ├─ feedback-loop-agents/
│  └─ tools/                 # common tools (web, file, code, rag)
├─ llamaindex-projects/
│  ├─ fast-rag/
│  └─ routers/               # index routers, retriever fusion
├─ data/
│  ├─ raw/                   # source docs, images, pdfs
│  └─ processed/             # cleaned text, embeddings, indexes
├─ configs/
│  ├─ providers.example.yaml
│  ├─ rag.basic.yaml
│  ├─ rag.hybrid.yaml
│  └─ agents.yaml
├─ scripts/
│  ├─ ingest.py
│  ├─ build_index.py
│  └─ eval_offline.py
├─ streamlit-apps/
│  ├─ rag_app/
│  └─ agents_app/
├─ .env.example
├─ Makefile
└─ README.md  ← you are here
```

---

## Quickstart

### 1) Clone & create environment

```bash
git clone https://github.com/<you>/genai-lab.git
cd genai-lab

# fresh venv
python -m venv .venv
source .venv/bin/activate  # (Windows) .venv\Scripts\activate
python -m pip install --upgrade pip
```

### 2) Install base + pick a suite

```bash
pip install -r langchain-projects/requirements.txt
# or
pip install -r crewai-projects/requirements.txt
# or
pip install -r llamaindex-projects/requirements.txt
```

### 3) Configure secrets

Copy `.env.example` → `.env` and fill in any providers you plan to use (all optional).

---

## Project suites

### 1) LangChain Projects

**Focus:** different RAG pipelines, **LangSmith** tracing, **LangGraph** orchestration.

Run examples:

```bash
# basic RAG
cd langchain-projects/basic-rag
python app.py

# hybrid RAG (BM25 + vector + rerank)
cd ../hybrid-rag
python app.py

# LangGraph experiments
cd ../langgraph-experiments
python graph_demo.py
```

**Highlights**

* Pluggable chunking: recursive, semantic, fixed.
* Embeddings: local (Ollama) or hosted (OpenAI/Groq).
* Rerankers: Cross-Encoder (optional).
* Caching: local SQLite / FAISS; swap to Chroma if preferred.

---

### 2) CrewAI Projects

**Focus:** **Agentic AI** with role-based, feedback-driven flows.

Run examples:

```bash
cd crewai-projects/research-assistant
python run.py

cd ../feedback-loop-agents
python run.py
```

**Highlights**

* Roles with tools (web, RAG retriever, code runner).
* **Feedback/critic** agents; self-reflection loops.
* Deterministic vs creative planning modes.
* Optional LangGraph for stateful control.

---

### 3) LlamaIndex Projects

**Focus:** **fast RAG** with light-weight routing & fusion.

Run examples:

```bash
cd llamaindex-projects/fast-rag
python app.py

cd ../routers
python router_demo.py
```

**Highlights**

* Auto-index builders.
* Retriever fusion & SourceNodes with scores.
* Minimal latency setups.

---

## RAG pipelines (hybrid patterns)

Common patterns you’ll find here:

| Pattern                     | When to use                         | Building blocks                      |
| --------------------------- | ----------------------------------- | ------------------------------------ |
| **BM25 only**               | tiny corpora, exact keywords matter | `rank_bm25`                          |
| **Vector only**             | semantic match, clean domain        | sentence/mini-lm, `FAISS/Chroma`     |
| **Hybrid (BM25 + Vector)**  | noisy data, mixed queries           | weighted fusion; top-k merge         |
| **Vector + Rerank**         | precision at top-k needed           | bi-encoder + cross-encoder           |
| **Multi-index Router**      | heterogeneous domains/modalities    | index routing; toolformer hints      |
| **Multimodal (Text+Image)** | docs/images/diagrams                | **CLIP** embeddings + text retriever |

Switch behavior via `configs/rag.*.yaml`.

---

## Agentic workflows

* **Planner → Researcher → Synthesizer → Critic** loop
* Guardrails: instruction filters, tool whitelists, max-hops
* Memory: short-term (graph state) vs long-term (vector store)
* Failure handling: retries, backoff, graceful degradation to local models

See: `crewai-projects/feedback-loop-agents/`.

---

## Data ingestion & indexing

Use the shared scripts (framework-agnostic):

```bash
# 1) Ingest (clean & split)
python scripts/ingest.py --src data/raw --out data/processed --chunk 1000 --overlap 150

# 2) Build index (choose store/provider)
python scripts/build_index.py --input data/processed --store faiss

# 3) Offline evaluation on a Q&A set
python scripts/eval_offline.py --dataset tests/qa.jsonl --config configs/rag.hybrid.yaml
```

**Unstructured pipeline**: PDFs, HTML, PPT, images → text blocks with metadata (page, coordinates where available).
Drop raw assets in `data/raw/`.

---

## Models & providers

**Local (preferred for cost):**

* **Ollama** backends: `llama3.1`, `mistral`, `llava` (vision), `nomic-embed-text`, `clip`.

  ```bash
  # install & pull models
  curl -fsSL https://ollama.com/install.sh | sh
  ollama pull llama3.1
  ollama pull nomic-embed-text
  ollama pull clip
  ```

**Hosted (optional):**

* **OpenAI**, **Anthropic**, **Groq** for fast/accurate inference.
  Provide keys in `.env` and flip providers in configs.

---

## Configuration

Copy `.env.example` → `.env`:

```ini
# Optional providers
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...

# Observability
LANGSMITH_API_KEY=...
LANGSMITH_TRACING=true
LANGSMITH_PROJECT=genai-lab

# Local services
OLLAMA_HOST=http://localhost:11434

# Vector stores
CHROMA_DIR=.chroma
FAISS_DIR=.faiss
```

YAML configs in `configs/` control:

* retriever type & top-k
* reranker on/off
* model/provider selection
* prompt templates
* tool allowlists for agents

---

## Running the UIs (Streamlit)

Each app is self-contained:

```bash
# RAG demo UI
cd streamlit-apps/rag_app
pip install -r requirements.txt
streamlit run Home.py

# Agents demo UI
cd ../agents_app
pip install -r requirements.txt
streamlit run Home.py
```

Features:

* upload docs → build index → ask questions
* toggle providers (Ollama/OpenAI/Anthropic/Groq)
* view sources & confidence
* cost/latency panel (approximate)

---

## Observability & evals

* **LangSmith** tracing: set `LANGSMITH_TRACING=true` to capture runs/latency/tokens.
* **Offline evals**: `scripts/eval_offline.py` supports:

  * exact-match / contains
  * semantic similarity
  * judge-LLM rubric (hallucination/faithfulness)
* **A/B configs**: run two YAMLs against same dataset and compare metrics.

---

## Cost control tips

* Default to **Ollama** for dev loops; upgrade to hosted only for benchmarks.
* Use **short contexts**, small top-k, and **rerank** instead of fetching huge chunks.
* Cache embeddings & stores under `data/processed/`, **don’t** recompute each run.
* Batch evals; avoid judge-LLM for every prompt during iteration.

---

## Development workflow

Common commands (via `Makefile`):

```bash
make setup        # create venv & base deps
make ingest       # run scripts/ingest.py with defaults
make index        # build vector store
make eval         # offline eval on tests/qa.jsonl
make ui-rag       # run Streamlit RAG app
make ui-agents    # run Streamlit Agents app
```

Code style:

* Python 3.10+
* Ruff / Black for lint/format (optional)
* Type hints on public modules

Testing:

```bash
pytest -q
```

---

## Roadmap

* [ ] Add **rerank** options (mono-T5, BGE-reranker) as plug-ins
* [ ] Add **structured outputs** with function/tool calling guards
* [ ] Expand **multimodal RAG** (vision-text tables/plots OCR)
* [ ] Dataset cards + reproducible **eval harness**
* [ ] Docker images for each suite
* [ ] Example **LangGraph** state machines for production flows

---

## License

MIT — see `LICENSE` for details.

---

### Badges (optional)

You can add these to the top once you wire them up:

```
![Python](https://img.shields.io/badge/python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-enabled-green)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-enabled-green)
![CrewAI](https://img.shields.io/badge/CrewAI-enabled-green)
![Ollama](https://img.shields.io/badge/Ollama-local%20LLMs-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
```

---

**Tip:** Keep each project’s sub-README short (what it does, how to run, config knobs). Link back to this root README for shared concepts (RAG patterns, agents, indexing, evals).
