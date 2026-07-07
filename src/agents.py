"""
Agent definitions for the agentic RAG crew.

Three agents, each with a narrow responsibility:
  1. Retriever Agent  - calls the FAISS tool to fetch candidate chunks
  2. Grader Agent     - filters out irrelevant/noisy chunks (the "agentic" check)
  3. Answer Agent     - synthesizes the final answer from graded context

All agents share the same local Ollama gemma:7b model.
"""

from crewai import Agent, LLM

from src.config import OLLAMA_MODEL, OLLAMA_BASE_URL
from src.tools import DocumentRetrieverTool


def get_llm() -> LLM:
    """
    CrewAI's LLM class routes through litellm, which expects the
    'ollama/<model>' provider prefix to talk to a local Ollama server.
    """
    return LLM(
        model=f"ollama/{OLLAMA_MODEL}",
        base_url=OLLAMA_BASE_URL,
        temperature=0.2,  # low temp: we want grounded, factual answers, not creativity
    )


def build_retriever_agent(llm: LLM) -> Agent:
    return Agent(
        role="Document Retriever",
        goal=(
            "Given a user query, retrieve the most relevant chunks from the "
            "personal knowledge base (resume, project notes) using the document_retriever tool."
        ),
        backstory=(
            "You are a meticulous research assistant. You never answer from memory - "
            "you always pull raw context from the knowledge base first and hand it off "
            "to your teammates."
        ),
        tools=[DocumentRetrieverTool()],
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_grader_agent(llm: LLM) -> Agent:
    return Agent(
        role="Relevance Grader",
        goal=(
            "Evaluate retrieved chunks against the original query and discard any "
            "chunk that is not genuinely relevant. Flag it if none of the chunks "
            "are relevant, so the pipeline knows retrieval failed."
        ),
        backstory=(
            "You are a strict fact-checker. You would rather say 'no relevant "
            "context found' than let noisy or unrelated context pollute the final answer."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def build_answer_agent(llm: LLM) -> Agent:
    return Agent(
        role="Answer Synthesizer",
        goal=(
            "Write a clear, direct answer to the user's query using ONLY the "
            "graded, relevant context provided. If the context is insufficient, "
            "say so explicitly instead of guessing."
        ),
        backstory=(
            "You are a precise technical writer. You never hallucinate facts not "
            "present in the given context, and you always cite which source(s) "
            "you drew from."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )
