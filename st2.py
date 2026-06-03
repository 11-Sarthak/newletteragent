import streamlit as st

from newsletter import run_newsletter_agent

st.set_page_config(
    page_title="AI Newsletter Agent",
    layout="wide"
)

st.title("🧠 Autonomous AI Newsletter Agent")

st.markdown(
    "Generate weekly AI agent newsletters using LangGraph + Tavily + Groq"
)

goal = st.text_area(
    "Enter your goal:",
    value="Create a weekly newsletter on latest AI agent news and research papers"
)

run_btn = st.button("🚀 Generate Newsletter")

if run_btn:

    if not goal.strip():
        st.error("Please enter a goal")

    else:

        with st.spinner("Running AI agent workflow..."):

            result = run_newsletter_agent(goal)

        st.success("Newsletter Generated Successfully!")

        st.subheader("📩 Final Newsletter")

        st.markdown(result)

        st.download_button(
            label="⬇ Download Markdown",
            data=result,
            file_name="newsletter.md",
            mime="text/markdown"
        )