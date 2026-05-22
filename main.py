#v7
import os
import json
import uuid
import asyncio
from typing import List, Dict, TypedDict, Annotated, Literal
import yfinance
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from ddgs import DDGS
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langgraph.checkpoint.memory import MemorySaver
from langfuse.langchain import CallbackHandler
from langfuse import get_client


# 0. initialization

load_dotenv()

# llm
llm = ChatOpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    temperature=0
)



# 1. data schema

class Tickerlist(BaseModel):
    tickers: List[str] = Field(description="List of stock ticker symbols.")


class TradeSignal(BaseModel):
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"] = "NEUTRAL"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    rationale: str = Field(description="One-sentence professional summary.")


class RiskAssessment(BaseModel):
    approved: bool = False
    max_position_size: float = Field(default=0.0, ge=0.0, le=1.0)
    risk_reason: str = Field(description="Summary of risk profile and reason for approval or veto.")


def add_items(left: list, right: list) -> list:
    """Thread-safe reducer for compiling unified asset packets."""
    return (left or []) + (right or [])


class AssetResult(TypedDict):
    ticker: str
    signals: List[dict]
    risk_assessment: dict


class GraphState(TypedDict):
    sector_query: str
    num_companies: int
    discovered_tickers: List[str]
    asset_results: Annotated[List[AssetResult], add_items]
    final_report: str


# Bind schemas to LLMs
screener_llm = llm.with_structured_output(Tickerlist, method="function_calling")
analyst_llm = llm.with_structured_output(TradeSignal, method="function_calling")
risk_llm = llm.with_structured_output(RiskAssessment, method="function_calling")



# 2. data collection

def fetch_sector_tickers_sync(sector_query: str, max_results: int) -> str:
    print(f" [Screener] Locating top asset tickers for '{sector_query}'")
    try:
        with DDGS() as ddgs:
            query = f"largest {sector_query} stocks by market cap ticker symbols"
            results = ddgs.text(query, max_results=max_results)
            return "\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        print(f"Search failed ({e}). Fallback triggered.")
        return "Fallback: NVDA, AMD, INTC"


def fetch_stock_data_sync(ticker_symbol: str) -> dict:
    try:
        ticker = yfinance.Ticker(ticker_symbol)
        info = ticker.info
        fundamentals = {
            "market_cap": info.get("marketCap", "N/A"),
            "trailing_pe": info.get("trailingPE", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"),
            "revenue_growth": info.get("revenueGrowth", "N/A")
        }
        history = ticker.history(period="1mo")
        if not history.empty:
            start_price, end_price = history['Close'].iloc[0], history['Close'].iloc[-1]
            technical = {
                "current_price": round(end_price, 2),
                "movement_30d_pct": round(((end_price - start_price) / start_price) * 100, 2)
            }
        else:
            technical = {"current_price": "N/A", "movement_30d_pct": 0.0}
        return {"fundamentals": fundamentals, "technical": technical}
    except Exception:
        return {"fundamentals": {}, "technical": {}}


# 3. agents

async def run_fundamental_analyst(ticker: str, data: dict) -> dict:
    prompt = f"Analyze fundamental metrics for {ticker}: {data['fundamentals']}. Provide BULLISH/BEARISH/NEUTRAL."
    res = await analyst_llm.ainvoke(prompt)
    signal = res.model_dump()
    signal["specialist"] = "FUNDAMENTAL"
    return signal


async def run_technical_analyst(ticker: str, data: dict) -> dict:
    prompt = f"Analyze 30-day structural momentum for {ticker}: {data['technical']}. Provide BULLISH/BEARISH/NEUTRAL."
    res = await analyst_llm.ainvoke(prompt)
    signal = res.model_dump()
    signal["specialist"] = "TECHNICAL"
    return signal


async def run_sentiment_analyst(ticker: str) -> dict:
    prompt = f"Assess broad market sector sentiment and potential headwinds for {ticker}. Provide BULLISH/BEARISH/NEUTRAL."
    res = await analyst_llm.ainvoke(prompt)
    signal = res.model_dump()
    signal["specialist"] = "SENTIMENT"
    return signal


# 4. nodes

async def screener_node(state: GraphState):
    target_count = state.get("num_companies", 5)
    raw_data = await asyncio.to_thread(fetch_sector_tickers_sync, state["sector_query"], target_count + 2)

    prompt = f"""Extract exactly {target_count} stock tickers for '{state["sector_query"]}' from this data: {raw_data}. Return ONLY valid US ticker symbols."""
    result = await screener_llm.ainvoke(prompt)
    clean_tickers = [t.strip().upper() for t in result.tickers if len(t.strip()) <= 5]

    print(f"[Screener] Discovered Target Pool: {clean_tickers}")
    return {"discovered_tickers": clean_tickers}


async def evaluate_asset_node(state: dict):
    """CONDUCTOR NODE: Orchestrates concurrent specialists inside the asset boundary."""
    ticker = state["ticker"]
    print(f"[{ticker}] Initiating concurrent specialist evaluation...")

    stock_data = await asyncio.to_thread(fetch_stock_data_sync, ticker)

    specialist_tasks = [
        run_fundamental_analyst(ticker, stock_data),
        run_technical_analyst(ticker, stock_data),
        run_sentiment_analyst(ticker)
    ]
    compiled_signals = await asyncio.gather(*specialist_tasks)

    risk_prompt = f"""
    You are the Risk Governor for {ticker}. Review the compiled specialist intel:
    {json.dumps(compiled_signals, default=str)}

    Logic Constraints:
    - If direct conflict exists across vectors, reject (approved=False).
    - If average confidence is weak, reject.
    - If approved, set max_position_size (0.01 to 0.10).
    """
    risk_assessment = await risk_llm.ainvoke(risk_prompt)

    print(f"[{ticker}] Evaluation complete. Cleared Risk: {risk_assessment.approved}")

    return {
        "asset_results": [{
            "ticker": ticker,
            "signals": compiled_signals,
            "risk_assessment": risk_assessment.model_dump()
        }]
    }


async def portfolio_manager_node(state: GraphState):
    query = state.get("sector_query", "Unknown Sector")
    results = state.get("asset_results", [])

    approved = [r["ticker"] for r in results if r["risk_assessment"]["approved"]]
    vetoed = [r["ticker"] for r in results if not r["risk_assessment"]["approved"]]

    print(f"[Portfolio Manager] Drafting final synthesis for {len(results)} assets...")

    synthesis_prompt = f"Write a 2-paragraph macro synthesis for the '{query}' sector. Discuss why assets ({approved}) showed strength and the systemic risks that caused vetoes for ({vetoed}). Maintain institutional tone. No markdown headers."
    cio_synthesis = (await llm.ainvoke(synthesis_prompt)).content

    # Build Report
    report = f"## Sector Analysis: {query.upper()}\n\n### Executive Market Synthesis\n> {cio_synthesis}\n\n---\n\n"
    report += "### Execution Summary\n| Ticker | Verdict | Primary Driver |\n|--------|---------|----------------|\n"

    for r in results:
        ticker = r["ticker"]
        if r["risk_assessment"]["approved"]:
            #  fundamental direction
            fund_sig = next((s for s in r["signals"] if s["specialist"] == "FUNDAMENTAL"), None)
            verdict = fund_sig["direction"].capitalize() if fund_sig else "Approved"
            driver = r["risk_assessment"]["risk_reason"].replace("|", "-")
        else:
            verdict = "Vetoed"
            driver = r["risk_assessment"]["risk_reason"].replace("|", "-")

        report += f"| {ticker} | {verdict} | {driver} |\n"

    return {"final_report": report}


# 5. routing
def map_to_assets(state: GraphState):
    """Fans out execution to isolated asset sub-graphs."""
    return [Send("evaluate_asset_node", {"ticker": ticker}) for ticker in state["discovered_tickers"]]


workflow = StateGraph(GraphState)

workflow.add_node("screener_node", screener_node)
workflow.add_node("evaluate_asset_node", evaluate_asset_node)
workflow.add_node("portfolio_manager_node", portfolio_manager_node)

workflow.add_edge(START, "screener_node")
workflow.add_conditional_edges("screener_node", map_to_assets, ["evaluate_asset_node"])
workflow.add_edge("evaluate_asset_node", "portfolio_manager_node")
workflow.add_edge("portfolio_manager_node", END)

# state memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)


# 6. execution
async def main():
    print("\n" + "=" * 50)
    print("BOOTING V7 AUTONOMOUS RESEARCH KERNEL")
    print("=" * 50 + "\n")

    user_sector = input(
        "Enter target industry/sector (e.g., 'biotech', 'cloud computing'): ").strip() or "semiconductors"

    user_count_str = input("Enter number of companies to analyze (e.g., 3, 5): ").strip()
    try:
        user_count = int(user_count_str)
    except ValueError:
        user_count = 3

    print(f"\n[System] Compiling edge-first swarm for top {user_count} assets in '{user_sector}'...\n")

    initial_state = {
        "sector_query": user_sector,
        "num_companies": user_count
    }

    langfuse_handler = CallbackHandler()
    run_id = uuid.uuid4().hex

    thread_config = {
        "configurable": {"thread_id": f"run_{run_id}"},
        "callbacks": [langfuse_handler],
        "run_name": f"V7_EdgeKernel_{user_sector.upper()}",
        "metadata": {
            "session_id": f"session_{run_id}",
            "tags": f"{user_sector}, async-swarm"
        }
    }

    # execute graph
    async for output in app.astream(initial_state, config=thread_config):
        for node_name, state_update in output.items():
            print(f" Trace: Node '{node_name}' finished execution.")

    get_client().flush()

    # retrieve and output final state
    final_state = app.get_state(thread_config).values
    report_content = final_state.get("final_report", "Pipeline execution error occurred.")

    print("\n" + "=" * 50)
    print(" FINAL PORTFOLIO OUTCOME REPORT")
    print("=" * 50)
    print(report_content)
    print("=" * 50)

    with open("portfolio_execution_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("\nAnalytical report saved locally to -> portfolio_execution_report.md")


if __name__ == "__main__":
    asyncio.run(main())