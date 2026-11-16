import os
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.tools.tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()

# Streamlit secrets compatibility
try:
    import streamlit as st
    if hasattr(st, "secrets"):
        for key in ["GROQ_API_KEY", "TAVILY_API_KEY", "GROQ_MODEL"]:
            if key in st.secrets and key not in os.environ:
                os.environ[key] = st.secrets[key]
except Exception:
    pass

# Model Initialization with Groq
# Default model: qwen/qwen3.8-27b (currently active on Groq, 131K context, strong tool calling)
# Other available models: openai/gpt-oss-20b, openai/gpt-oss-120b
groq_api_key = os.getenv("GROQ_API_KEY") or os.getenv("OPENAI_API_KEY", "")
groq_model = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")

try:
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        model=groq_model,
        groq_api_key=groq_api_key or "gsk_dummy",
        temperature=0,
        max_tokens=900,
    )
except ImportError:
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=groq_model,
        api_key=groq_api_key or "gsk_dummy",
        base_url="https://api.groq.com/openai/v1",
        temperature=0,
        max_tokens=900,
    )


# 1st Agent : Search Agent
def build_search_agent():
    return create_agent(
        model= llm,
        tools=[web_search],
       
    )

# 2nd Agent : Reader Agent
def build_reader_agent():
    return create_agent(
        model= llm,
        tools=[scrape_url],

    )


#writer chain 

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt | llm | StrOutputParser()




#critic_chain 

critic_prompt = ChatPromptTemplate.from_messages([
     ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("human", """Review the research report below and evaluate it strictly.

Report:
{report}

Respond in this exact format:

Score: X/10

Strengths:
- ...
- ...

Areas to Improve:
- ...
- ...

One line verdict:
..."""),
])

critic_chain = critic_prompt | llm | StrOutputParser()

