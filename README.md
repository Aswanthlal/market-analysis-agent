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

