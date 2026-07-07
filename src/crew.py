"""
Assembles the Retriever -> Grader -> Answer agents and tasks into a
single sequential Crew, and exposes a simple run_query() entry point.
"""

from crewai import Crew, Process

from src.agents import get_llm, build_retriever_agent, build_grader_agent, build_answer_agent
from src.tasks import build_retrieve_task, build_grade_task, build_answer_task


def run_query(query: str) -> str:
    llm = get_llm()

    retriever_agent = build_retriever_agent(llm)
    grader_agent = build_grader_agent(llm)
    answer_agent = build_answer_agent(llm)

    retrieve_task = build_retrieve_task(retriever_agent, query)
    grade_task = build_grade_task(grader_agent, query, retrieve_task)
    answer_task = build_answer_task(answer_agent, query, grade_task)

    crew = Crew(
        agents=[retriever_agent, grader_agent, answer_agent],
        tasks=[retrieve_task, grade_task, answer_task],
        process=Process.sequential,  # fixed pipeline order, matches allow_delegation=False
        verbose=True,
    )

    result = crew.kickoff()
    return str(result)
