#  Market Analyzer — Async Multi Agent Equity Research System

An asynchronous multi agent equity research platform built with **LangGraph**, **Groq LLMs**, **Streamlit**, and **Langfuse**.

The system performs parallel stock analysis across entire market sectors using isolated asset-level evaluation pipelines. Each stock is analyzed by multiple specialist AI agents before passing through a centralized Risk Governor responsible for approval logic and portfolio allocation constraints.

The project focuses on:

* hierarchical agent orchestration
* concurrent specialist evaluation
* async execution pipelines
* risk-governed synthesis
* observability-first infrastructure


# Features

* Dynamic sector based stock discovery
* Concurrent specialist agents per asset
* Hierarchical orchestration using LangGraph
* Async execution
* Risk governed portfolio synthesis
* Langfuse tracing
* Streamlit execution dashboard

---

# Architecture

The system uses a hierarchical map reduce orchestration model.

```text
START
  ↓
Screener Agent
  ↓
Parallel Asset Pipelines
  ├── Fundamental Agent
  ├── Technical Agent
  ├── Sentiment Oriented Agent
  └── Risk Governor
  ↓
Portfolio Manager
  ↓
END
```

Each asset pipeline operates independently, allowing:

* isolated execution boundaries
* parallel evaluation
* improved scalability
* simplified tracing & debugging

---

# Multi Agent Design

Every stock is evaluated through a localized multi agent pipeline.

### Specialist Agents

| Agent                    | Responsibility                                |
| ------------------------ | --------------------------------------------- |
| Fundamental Agent        | Valuation, growth, and profitability analysis |
| Technical Agent          | Momentum and price structure analysis         |
| Sentiment-Oriented Agent | Broad market sentiment reasoning              |
| Risk Governor            | Approval logic and position sizing            |

Specialist agents execute concurrently using:

```python
await asyncio.gather(*specialist_tasks)
```

This reduces total inference latency while maintaining isolated reasoning boundaries per asset.

---

# Concurrency Model

The platform uses asynchronous orchestration throughout the stack:

* Async LangGraph execution via `app.astream`
* Concurrent specialist execution using `asyncio.gather`
* Blocking I/O offloaded using `asyncio.to_thread`
* Isolated event loop execution for Streamlit compatibility

This allows the system to scale asset evaluations efficiently without blocking the UI runtime.

---

# Observability

The platform integrates Langfuse for:

* Nested execution traces
* Agent span visualization
* Prompt tracking
* Token usage monitoring
* Runtime latency analysis
* Hierarchical workflow replay


---

# Tech Stack

| Layer         | Technology           |
| ------------- | -------------------- |
| Orchestration | LangGraph            |
| LLM Inference | Groq + Llama 3.3 70B |
| Observability | Langfuse             |
| UI            | Streamlit            |
| Market Data   | YFinance             |
| Search Layer  | DuckDuckGo Search    |
| Schemas       | Pydantic             |
| Runtime       | AsyncIO              |

---

# Project Structure

```text
market-analysis-agent/
│
├── main.py
├── ui.py
├── requirements.txt
├── .env.example
├── README.md
│
├── screenshots/
│   ├── ui-demo.png
│   └── langfuse-traces.png
│
└── architecture/
    └── system-design.png
```


