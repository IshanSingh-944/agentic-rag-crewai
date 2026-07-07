"""
Task definitions for the agentic RAG crew.

Tasks are chained via the `context` parameter so each task's output
feeds into the next - this is what makes it a pipeline rather than
three independent LLM calls.
"""

from crewai import Task, Agent


def build_retrieve_task(agent: Agent, query: str) -> Task:
    return Task(
        description=(
            f"Use the document_retriever tool to fetch the most relevant chunks "
            f"for this query: '{query}'. Return the raw chunks exactly as retrieved, "
            f"including their source tags."
        ),
        expected_output=(
            "A list of retrieved chunks with source tags, or a clear statement "
            "that no chunks were found."
        ),
        agent=agent,
    )


def build_grade_task(agent: Agent, query: str, retrieve_task: Task) -> Task:
    return Task(
        description=(
            f"Review the chunks retrieved for the query: '{query}'. "
            f"Keep only chunks that are genuinely relevant to answering this query. "
            f"Discard the rest. If NONE are relevant, state that explicitly."
        ),
        expected_output=(
            "The filtered list of relevant chunks (with sources), or an explicit "
            "'no relevant context' statement if nothing qualifies."
        ),
        agent=agent,
        context=[retrieve_task],
    )


def build_answer_task(agent: Agent, query: str, grade_task: Task) -> Task:
    return Task(
        description=(
            f"Using ONLY the graded, relevant context, write a clear and direct "
            f"answer to this query: '{query}'. "
            f"If the context is insufficient to answer confidently, say so explicitly "
            f"rather than guessing. Cite the source(s) you used."
        ),
        expected_output=(
            "A concise, well-grounded answer to the query, with source citations, "
            "or an explicit statement that the knowledge base lacks sufficient info."
        ),
        agent=agent,
        context=[grade_task],
    )
