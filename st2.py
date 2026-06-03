import streamlit as st
from langchain_core.messages import HumanMessage

# import your existing graph + function
# make sure your agent code is in a file like agent.py
from newsletter import run_newsletter_agent  

st.set_page_config(page_title="AI Newsletter Agent", layout="wide")

st.title("🧠 Autonomous AI Newsletter Agent")
st.markdown("Generate weekly AI agent newsletters using LangGraph + LLM tools")

# ----------------------------
# INPUT
# ----------------------------
goal = st.text_area(
    "Enter your goal:",
    value="Create a weekly newsletter on latest AI agent news and research papers"
)

mode = st.selectbox("Mode", ["autonomous", "human-in-loop"])

run_btn = st.button("🚀 Generate Newsletter")

# ----------------------------
# OUTPUT
# ----------------------------
if run_btn:
    if not goal.strip():
        st.error("Please enter a goal")
    else:
        with st.spinner("Running AI agent workflow..."):

            result = run_newsletter_agent(goal, mode)

        st.success("Newsletter Generated Successfully!")

        st.subheader("📩 Final Newsletter")

        st.markdown(result)

        # ----------------------------
        # DOWNLOAD BUTTON
        # ----------------------------
        st.download_button(
            label="⬇ Download Markdown",
            data=result,
            file_name="newsletter.md",
            mime="text/markdown"
        )