# AI Newsletter Agent

## Overview

This project is an autonomous AI Newsletter Agent built using LangGraph, LangChain, Groq LLM, and Tavily Search.

The agent receives a goal such as:

> "Create a weekly newsletter on latest AI agent news and research papers."

It then autonomously:

1. Plans the newsletter creation process
2. Researches current AI agent news
3. Collects recent AI research information
4. Generates a structured newsletter
5. Reviews and improves the generated content
6. Saves the final newsletter as a Markdown file

---

## Features

* Multi-step agent workflow
* Autonomous planning and execution
* Web research using Tavily Search
* Research paper discovery
* Newsletter generation in Markdown format
* Self-review / critique stage
* Built using LangGraph state-based workflows
* Simple Streamlit frontend

---

## Agent Workflow

```text
User Goal
    ↓
Planner
    ↓
Research
    ↓
Writer
    ↓
Critic
    ↓
Output
```

### Nodes

#### Planner

Creates a plan for gathering information and generating the newsletter.

#### Research

Fetches latest AI agent news and research-related information.

#### Writer

Generates a structured newsletter using the collected information.

#### Critic

Reviews the draft and improves clarity, structure, and completeness.

#### Output

Saves the final newsletter as a Markdown file.

---

## Tech Stack

* Python
* LangGraph
* LangChain
* Groq LLM
* Tavily Search API
* Streamlit

---

## Installation

```bash
pip install -r requirements.txt
```

Create a `.env` file:

```env
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

---

## Run Agent

```bash
python newsletter.py
```

---

## Run Streamlit App

```bash
streamlit run st2.py
```

---

## Example Goal

```text
Create a weekly newsletter on latest AI agent news and research papers
```

---

## Output

The generated newsletter is saved as:

```text
newsletter.md
```

and displayed in the Streamlit interface.

---

## Author

Sarthak Choudhary
