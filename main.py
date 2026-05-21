# V5

import os
import json
import uuid
from typing import List, Dict, Any, TypedDict, Annotated, Literal
import operator
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

#  load env var
load_dotenv()

# 1. data schemas

class Tickerlist(BaseModel):
    """structured op"""
    tickers: List[str] = Field(description="List of top stock ticker symbols matching the target industry query.")


class TradeSignal(BaseModel):
    """analyst"""
    ticker: str=""
    agent_type: Literal["FUNDAMENTAL", "TECHNICAL"] = "FUNDAMENTAL"
    direction: Literal["BULLISH", "BEARISH", "NEUTRAL"] = "NEUTRAL"
    confidence: float=Field(default=0.5, ge=0.0, le=1.0, description="Confidence rating from 0.0 to 1.0")
    rationale: str=Field(description="One-sentence professional summary explaining the decision.")


class RiskAssessment(BaseModel):
    """risk evaluation"""
    ticker: str=""
    approved: bool = False
    max_position_size: float = Field(default=0.0, ge=0.0, le=1.0, description="Maximum allocation limit (0.0 to 1.0)")
    risk_reason: str=Field(description="Summary of risk profile and reason for approval or veto.")

def add_items(left: list, right: list) -> list:
    """reducer to accumulate lists in parallel branches."""
    if not left:
        left = []
    if not right:
        right = []
    return left + right
class GraphState(TypedDict):
    """system state"""
    sector_query: str
    num_companies: int
    discovered_tickers: List[str]
    current_ticker: str
    signals: Annotated[List[Dict], add_items]
    risk_assessments: Annotated[List[Dict], add_items]
    final_report: str

# 2. data acquisition
def search_sector_tickers_tool(sector_query: str, max_results: int) -> str:
    """ search web to extract 5 stock symbols from given industry"""
    print(f" [Screener Tool] Locating top asset tickers for '{sector_query}'")

    try:
        with DDGS() as ddgs:
            # targeted search query to prioritize tickers and market cap
            query=f"largest {sector_query} stocks by market cap ticker symbols"
            results = ddgs.text(query, max_results=max_results)
            # merge results
            return  "\n".join([f"Title: {r['title']}\nSnippet: {r['body']}" for r in results])
    except Exception as e:
        print(f"Search failed ({e}). Implementing immediate fallback assets.")
        return  "Fallback: NVDA, AMD, INTC"


def fetch_stock_data_tool(ticker_symbol: str) ->dict:
    """fetch valuation and price trends via yfinance"""
    print(f"[Data Tool] Querying infrastructure metrics for {ticker_symbol}")
    try:
        ticker=yfinance.Ticker(ticker_symbol)
        info=ticker.info

        #fundamental metrics
        fundamentals={
            "company_name": info.get("longName", ticker_symbol),
            "market_cap": info.get("marketCap", "N/A"),
            "trailing_pe": info.get("trailingPE", "N/A"),
            "profit_margin": info.get("profitMargins", "N/A"),
            "revenue_growth": info.get("revenueGrowth", "N/A")
        }

        # last 30day price data
        history=ticker.history(period="1mo")
        if not history.empty:
            start_price= history['Close'].iloc[0]
            end_price=history['Close'].iloc[-1]
            price_change_pct=((end_price-start_price)/start_price)*100

            technical= {
                "current_price": round(end_price, 2),
                "movement_30d_pct": round(price_change_pct, 2)
            }
        else:
            technical={"current_price": "N/A", "movement_30d_pct": 0.0}
        return {"fundamentals": fundamentals, "technical": technical}
    except Exception as e:
        print(f"Error pulling {ticker_symbol}: {(e)}")
        return {"fundamentals": {}, "technical": {}}



# 3 agent (llm setup)

# initialize grok
grok_llm=ChatOpenAI(
    api_key=os.getenv("XAI_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
    model="llama-3.3-70b-versatile",
    temperature=0
)

# bind schema to model
screener_llm=grok_llm.with_structured_output(Tickerlist,method="function_calling")
fundamental_llm=grok_llm.with_structured_output(TradeSignal, method="function_calling")
technical_llm=grok_llm.with_structured_output(TradeSignal, method="function_calling")
risk_llm=grok_llm.with_structured_output(RiskAssessment, method="function_calling")

# 4 nodes

def screener_node(state: GraphState):
    target_count = state.get("num_companies", 5)

    raw_data=search_sector_tickers_tool(state["sector_query"],max_results=target_count+2)
    prompt=f"""
    You are an elite Market Research Assistant.
    Your task is to identify EXACTLY {target_count} stock tickers for the '{state["sector_query"]}' sector.
    
    Recent Web Search Data:
    {raw_data}
    
    Instructions:
    1. Extract tickers from the web data if present.
    2. If the web data does not contain enough valid tickers, use your internal institutional knowledge to fill the gap.
    3. You MUST return exactly {target_count} tickers.
    4. Provide ONLY valid, publicly traded US stock ticker symbols (e.g., 'REGN', 'VRTX').
    """
    result= screener_llm.invoke(prompt)
    clean_tickers=[t.strip().upper() for t in result.tickers if len(t.strip())<=5]
    print(f"[Screener Agent] Discovered Target Pool: {clean_tickers}")
    return {"discovered_tickers": clean_tickers}

def fundamental_analyst_node(state: GraphState):
    ticker=state["current_ticker"]
    stock_data=fetch_stock_data_tool(ticker)
    prompt = f"""
        You are an institutional Fundamental Equity Analyst.
        Evaluate the asset valuation metrics, profit profiles, and growth factors for {ticker}:
        {stock_data['fundamentals']}

        Task:
        Provide a professional rating. Set 'direction' to BULLISH, BEARISH, or NEUTRAL. 
        Assign a 'confidence' rating between 0.0 and 1.0. Summary must be exactly 1 sentence.
        """
    signal=fundamental_llm.invoke(prompt)
    signal.ticker=ticker
    signal.agent_type="FUNDAMENTAL"
    return {"signals": [signal.model_dump()]}

def technical_analyst_node(state: GraphState):
    ticker=state["current_ticker"]
    stock_data=fetch_stock_data_tool(ticker)

    prompt = f"""
        You are a Quantitative Momentum and Technical Analyst.
        Evaluate the recent 30-day structural trailing data points for {ticker}:
        {stock_data['technical']}

        Task:
        Provide a trend-following rating. Set 'direction' to BULLISH, BEARISH, or NEUTRAL.
        Assign a 'confidence' rating between 0.0 and 1.0. Summary must be exactly 1 sentence.
        """
    signal=technical_llm.invoke(prompt)
    signal.ticker=ticker
    signal.agent_type="TECHNICAL"
    return {"signals": [signal.model_dump()]}


def risk_governor_node(state: GraphState):
    ticker = state["current_ticker"]
    ticker_signals = [s for s in state["signals"] if s["ticker"] == ticker]

    prompt = f"""
    You are the Chief Risk Officer. You hold total system veto authority over portfolio allocation.
    Review the concurrent specialist analysis outputs compiled for {ticker}:
    {json.dumps(ticker_signals, default=str)}

    Mandatory Logic Constraints:
    1. If there is a direct direction conflict (e.g., one BULLISH, one BEARISH) or average confidence < 0.60, reject the order (approved = False).
    2. If approved = True, establish a maximum portfolio risk exposure limits ('max_position_size') from 0.01 to 0.10.
    """
    assessment = risk_llm.invoke(prompt)
    assessment.ticker = ticker
    print(f"[Risk Governor] Processing complete for {ticker} -> Approved Status: {assessment.approved}")
    return {"risk_assessments": [assessment.model_dump()]}


def portfolio_manager_node(state: GraphState):
    query = state.get("sector_query", "Unknown Sector")
    signals = state.get("signals", [])
    risks = state.get("risk_assessments", [])

    print(f"[Portfolio Manager] Synthesizing final execution matrix")

    # 1. Pre-process stats per ticker
    ticker_stats = {}
    for r in risks:
        t = r.get("ticker")
        t_signals = [s for s in signals if s.get("ticker") == t]
        avg_conf = sum(s.get("confidence", 0) for s in t_signals) / len(t_signals) if t_signals else 0

        ticker_stats[t] = {
            "approved": r.get("approved"),
            "max_position_size": r.get("max_position_size", 0),
            "risk_reason": r.get("risk_reason"),
            "avg_confidence": avg_conf,
            "signals": t_signals
        }

    approved_tickers = [t for t, stats in ticker_stats.items() if stats["approved"]]
    vetoed_tickers = [t for t, stats in ticker_stats.items() if not stats["approved"]]

    # ======
    # NEW: THE CIO SYNTHESIZER AGENT
    # ======
    print(f"[CIO Agent] Writing macro synthesis for the {query.upper()} sector...")

    synthesis_prompt = f"""
    You are an elite Chief Investment Officer at a quantitative hedge fund.
    Review the algorithmic execution decisions for the '{query}' sector based on the data below.

    Approved Assets: {approved_tickers}
    Vetoed Assets: {vetoed_tickers}
    Raw Agent Data: {json.dumps(ticker_stats, default=str)}

    Task:
    Write a 2-paragraph "Executive Market Synthesis". 
    Explain the overall health/momentum of the sector based on this data. Highlight why the approved assets show strength, and note the specific systemic risks that caused the vetoes. 
    Maintain a highly professional, institutional tone. Do not use markdown formatting like headers, just write the paragraphs.
    """

    # We use the base LLM here (no structured output) because we WANT a fluid text narrative
    cio_synthesis = grok_llm.invoke(synthesis_prompt).content
    # ========================================================

    # 3. Build the Report Header & LLM Synthesis
    report = f"## Sector Analysis: {query.upper()}\n\n"
    report += f"### Executive Market Synthesis\n"
    report += f"> {cio_synthesis}\n\n"
    report += "---\n\n"

    # 4. Dynamically calculate Top Pick
    if approved_tickers:
        top_pick = max(approved_tickers, key=lambda t: ticker_stats[t]["avg_confidence"])
        fund_rationale = next(
            (s.get("rationale") for s in ticker_stats[top_pick]["signals"] if s.get("agent_type") == "FUNDAMENTAL"),
            "Strong structural parameters across both execution vectors.")
        report += f"## Top Pick\n"
        report += f"* **Ticker:** {top_pick}\n"
        report += f"* **Justification:** {fund_rationale}\n\n"
    else:
        report += f"## Top Pick\n* **Ticker:** N/A\n* **Justification:** No tracked assets cleared baseline risk parameters.\n\n"

    # 5. Dynamically calculate Risk Alert
    if vetoed_tickers:
        risk_pick = min(vetoed_tickers, key=lambda t: ticker_stats[t]["avg_confidence"])
        report += f"### Risk Alert\n"
        report += f"* **Ticker:** {risk_pick}\n"
        report += f"* **Justification:** {ticker_stats[risk_pick]['risk_reason']}\n\n"

    # 6. Build the Institutional Summary Table
    report += "### Summary Table\n"
    report += "| Ticker | Verdict | Core Driver |\n"
    report += "|--------|---------|-------------|\n"

    for t, stats in ticker_stats.items():
        if stats["approved"]:
            fund_dir = next((s.get("direction") for s in stats["signals"] if s.get("agent_type") == "FUNDAMENTAL"),
                            "BULLISH")
            verdict = "Bullish" if fund_dir == "BULLISH" else "Neutral"
            core_driver = next((s.get("rationale") for s in stats["signals"] if s.get("agent_type") == "FUNDAMENTAL"),
                               "Aligned fundamental metrics.")
        else:
            verdict = "Vetoed"
            core_driver = stats["risk_reason"]

        core_driver = core_driver.replace("|", "-").strip()
        report += f"| {t} | {verdict} | {core_driver} |\n"

    return {"final_report": report}

# 5 execution and routing

def gather_node(state: GraphState):
    """Synchronization point"""
    return {}
def continue_to_analysts(state: GraphState):
    """processing parellel workflows"""
    sends=[]
    for ticker in state["discovered_tickers"]:
        sends.append(Send("fundamental_analyst_node", {"current_ticker": ticker}))
        sends.append(Send("technical_analyst_node", {"current_ticker": ticker}))
    return sends

def continue_to_risk(state: GraphState):
    """maps parallel streams to individual risk governors"""
    return [
        Send("risk_governor_node", {
            "current_ticker": ticker,
            "signals": state.get("signals", [])  # FIX: Explicitly pass the aggregated signals
        })
        for ticker in state["discovered_tickers"]
    ]

#graph topology
workflow= StateGraph(GraphState)

workflow.add_node("screener_node", screener_node)
workflow.add_node("fundamental_analyst_node", fundamental_analyst_node)
workflow.add_node("technical_analyst_node", technical_analyst_node)
workflow.add_node("gather_node", gather_node)
workflow.add_node("risk_governor_node", risk_governor_node)
workflow.add_node("portfolio_manager_node", portfolio_manager_node)

workflow.add_edge(START, "screener_node")
workflow.add_conditional_edges("screener_node", continue_to_analysts, ["fundamental_analyst_node", "technical_analyst_node"])

workflow.add_edge("fundamental_analyst_node", "gather_node")
workflow.add_edge("technical_analyst_node", "gather_node")

workflow.add_conditional_edges("gather_node", continue_to_risk, ["risk_governor_node"])

workflow.add_edge("risk_governor_node", "portfolio_manager_node")
workflow.add_edge("portfolio_manager_node", END)

memory= MemorySaver()
app = workflow.compile(checkpointer=memory)

# 6 Run
if __name__ == "__main__":
    print("\n"+ "="*50)
    print("BOOTING MULTI-AGENT WORKFLOW")
    print("="*50+"\n")

    user_sector = input("Enter target industry/sector (e.g., 'biotech', 'cloud computing'): ").strip()
    if not user_sector:
        user_sector = "artificial intelligence"  # Default fallback
        print(f"   -> No input detected. Defaulting to '{user_sector}'.")

    user_count_str = input(" Enter number of companies to analyze (e.g., 3, 5, 10): ").strip()
    try:
        user_count = int(user_count_str)
        if user_count <= 0:
            user_count = 3
    except ValueError:
        user_count = 3  # Default fallback
        print(f"   -> Invalid number. Defaulting to {user_count} companies.")

    print(f"\n Booting swarm for top {user_count} companies in '{user_sector}'...\n")

    #config
    initial_state={
        "sector_query": user_sector,
        "num_companies": user_count
    }

    # Initialize Langfuse
    langfuse_handler = CallbackHandler()

    run_id = uuid.uuid4().hex

    thread_config = {
        "configurable": {"thread_id": f"run_{run_id}"},
        "callbacks": [langfuse_handler],
        "run_name": f"Portfolio_Analysis_{user_sector.upper()}",  # Gives the trace a clean, readable title
        "metadata": {
            "langfuse_session_id": f"session_{run_id}",  # Groups the entire run into one replayable session
            "langfuse_tags": [user_sector, f"companies:{user_count}"]  # Adds filterable tags to the dashboard
        }
    }
    # execute and capture
    for output in app.stream(initial_state, config=thread_config):
        for node_name, state_update in output.items():
            print(f" Trace: Node '{node_name}' finished execution.")

    get_client().flush()

    final_state = app.get_state(thread_config).values

    # DEBUG: print the raw risk data before building the report
    print("\n" + "-" * 50)
    print("RAW RISK DATA RECEIVED BY PORTFOLIO MANAGER:")
    print(json.dumps(final_state.get("risk_assessments", []), indent=2))
    print("-" * 50 + "\n")

    report_content = final_state.get("final_report", "Pipeline execution error occurred.")
    print("\n" + "=" * 50)
    print(" FINAL PORTFOLIO OUTCOME REPORT")
    print("=" * 50)
    print(report_content)
    print("=" * 50)

    with open("portfolio_execution_report.md", "w", encoding="utf-8") as f:
        f.write(report_content)
    print("\nAnalytical report saved locally to -> portfolio_execution_report.md")