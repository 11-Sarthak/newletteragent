from langchain_tavily import TavilySearch
from langchain_groq import ChatGroq

from langchain_core.messages import HumanMessage, SystemMessage

from typing_extensions import TypedDict
from typing import Annotated

from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from dotenv import load_dotenv

load_dotenv()




tav = TavilySearch(max_results=5)

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.2,
    max_tokens=2000
)




class State(TypedDict):
    messages: Annotated[list, add_messages]




PLANNER_PROMPT = SystemMessage(content="""
You are a planning agent.

Create a short plan for generating an AI newsletter.

Return:
1. News topics to search
2. Research topics to search
3. Newsletter structure
""")

WRITER_PROMPT = SystemMessage(content="""
You are a professional AI newsletter writer.

Create a Markdown newsletter.

Sections:

#  Top AI Agent News

#  Research Papers

#  AI Tools & Frameworks

#  Key Insights

#  Weekly Summary

Use information provided.
""")

CRITIC_PROMPT = SystemMessage(content="""
You are a newsletter editor.

Review the newsletter.

Check:
- clarity
- structure
- factual consistency
- readability

Return only the improved newsletter.
""")



def planner(state: State):

    goal = state["messages"][-1].content

    plan = llm.invoke([
        PLANNER_PROMPT,
        HumanMessage(content=goal)
    ])

    return {
        "messages": [plan]
    }


def research(state: State):

    news = tav.invoke(
        "latest AI agent news this week"
    )

    papers = tav.invoke(
        "latest AI agent research papers 2026"
    )

    combined = f"""
NEWS

{news}

PAPERS

{papers}
"""

    return {
        "messages": [
            HumanMessage(content=combined)
        ]
    }


def writer(state: State):

    research_data = state["messages"][-1].content

    draft = llm.invoke([
        WRITER_PROMPT,
        HumanMessage(content=research_data)
    ])

    return {
        "messages": [draft]
    }


def critic(state: State):

    draft = state["messages"][-1].content

    final = llm.invoke([
        CRITIC_PROMPT,
        HumanMessage(content=draft)
    ])

    return {
        "messages": [final]
    }


def output_node(state: State):

    newsletter = state["messages"][-1].content

    try:
        with open("newsletter.md", "w", encoding="utf-8") as f:
            f.write(newsletter)
    except:
        pass

    return state




builder = StateGraph(State)

builder.add_node("planner", planner)
builder.add_node("research", research)
builder.add_node("writer", writer)
builder.add_node("critic", critic)
builder.add_node("output", output_node)

builder.add_edge(START, "planner")
builder.add_edge("planner", "research")
builder.add_edge("research", "writer")
builder.add_edge("writer", "critic")
builder.add_edge("critic", "output")
builder.add_edge("output", END)

memory = MemorySaver()

graph = builder.compile(
    checkpointer=memory
)





def run_newsletter_agent(goal: str):

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(content=goal)
            ]
        },
        config={
            "configurable": {
                "thread_id": "newsletter-thread"
            }
        }
    )

    return result["messages"][-1].content