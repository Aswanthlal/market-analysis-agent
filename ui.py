import streamlit as st
import asyncio
import uuid
from langfuse.langchain import CallbackHandler
from langfuse import get_client

# Import your compiled LangGraph app from your main file
from main import app

# Page Config
st.set_page_config(page_title="AI Quant Kernel", page_icon="⚡", layout="wide")

st.title("Market Analyzer")
st.markdown("### Automated stock analysis and risk evaluation")

# UI Layout
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        sector = st.text_input("Target Sector", value="eg: semiconductors")
    with col2:
        num_assets = st.number_input("Number of Assets", min_value=1, max_value=10, value=3)

# Execution Trigger
if st.button("Initialize Swarm", type="primary"):

    # UI Elements for streaming
    st.divider()
    status_box = st.status("Booting Multi-Agent Graph...", expanded=True)
    report_placeholder = st.empty()


    async def run_graph():
        run_id = uuid.uuid4().hex
        langfuse_handler = CallbackHandler()

        thread_config = {
            "configurable": {"thread_id": f"run_{run_id}"},
            "callbacks": [langfuse_handler],
            "run_name": f"UI_Run_{sector.upper()}",
            "metadata": {
                "session_id": f"session_{run_id}",
                "tags": f"{sector}, streamlit-ui"
            }
        }

        initial_state = {
            "sector_query": sector,
            "num_companies": num_assets
        }

        # Stream the graph execution to the Streamlit UI
        async for output in app.astream(initial_state, config=thread_config):
            for node_name, state_update in output.items():
                if node_name == "screener_node":
                    discovered = state_update.get("discovered_tickers", [])
                    status_box.write(f"**Screener** locked target assets: `{discovered}`")
                elif node_name == "evaluate_asset_node":
                    # Because we use map-reduce, this fires as each asset finishes
                    results = state_update.get("asset_results", [])
                    ticker = results[0]["ticker"]
                    cleared = results[0]["risk_assessment"]["approved"]
                    icon = "🟢" if cleared else "🔴"
                    status_box.write(f"{icon} **Risk Governor** evaluated `{ticker}`")
                elif node_name == "portfolio_manager_node":
                    status_box.write(f"**CIO Agent** is drafting the final synthesis...")

        get_client().flush()

        # Fetch final state
        final_state = app.get_state(thread_config).values
        return final_state.get("final_report", "Error generating report.")


    # Run the async loop SAFELY inside Streamlit
    with st.spinner("Agents are analyzing vectors..."):
        try:
            # 1. Create a fresh event loop specifically for this thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            # 2. Run the graph on this isolated loop
            report = loop.run_until_complete(run_graph())

            # 3. Close the loop to prevent memory leaks
            loop.close()

            status_box.update(label="Graph Execution Complete", state="complete", expanded=False)

            # Render the final Markdown report beautifully
            st.success("Portfolio Synthesis Generated")
            report_placeholder.markdown(report)

            # ==========================================
            # NEW: AUTOMATIC BACKGROUND MARKDOWN EXPORT
            # ==========================================
            try:
                with open("portfolio_execution_report.md", "w", encoding="utf-8") as f:
                    f.write(report)
                st.caption("Analytical report backed up locally to `portfolio_execution_report.md`")
            except Exception as write_error:
                st.warning(f"Report generated successfully, but local file write failed: {str(write_error)}")

        except Exception as e:
            status_box.update(label="Execution Failed", state="error", expanded=True)
            st.error(f"System Error: {str(e)}")