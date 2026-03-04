# ensure the top-level package directory is on PYTHONPATH when the
# test is executed as a script.  When you run
#
#    python agentic_rag/graph/chains/tests/test_chains.py
#
# the default sys.path[0] is the tests directory itself, so `graph`
# can't be found.  The preferred way to run the test suite is from the
# repository root using pytest:
#
#    cd /home/mrson/ai_chatbot_agent/agentic_rag
#    python -m pytest graph/chains/tests
#
# The code below makes the file more forgiving by adding the project
# root to sys.path when run directly.
import sys
import os

# determine project root relative to this file and insert it early in
# the search path if it's not already present
root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if root not in sys.path:
    sys.path.insert(0, root)

from pprint import pprint

from dotenv import load_dotenv

load_dotenv()

# many HTTP clients or APIs used in downstream code emit a log line when
# the USER_AGENT environment variable is missing.  It's not harmful, but
# it clutters test output.  Provide a sensible default for the duration
# of the test run so that the warning disappears; real deployments can
# still override via .env or the shell.
os.environ.setdefault("USER_AGENT", "agentic_rag/tests")


from graph.chains.generation import generation_chain
from graph.chains.hallucination_grader import (GradeHallucinations,
                                               hallucination_grader)
from graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from graph.chains.router import RouteQuery, question_router
from ingestion import retriever


def test_retrival_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_retrival_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_txt = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": "how to make pizaa", "document": doc_txt}
    )

    assert res.binary_score == "no"


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    generation = generation_chain.invoke({"context": docs, "question": question})
    pprint(generation)


def test_hallucination_grader_answer_yes() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )
    assert res.binary_score


def test_hallucination_grader_answer_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    res: GradeHallucinations = hallucination_grader.invoke(
        {
            "documents": docs,
            "generation": "In order to make pizza we need to first start with the dough",
        }
    )
    assert not res.binary_score


def test_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "vectorstore"


def test_router_to_websearch() -> None:
    question = "how to make pizza"

    res: RouteQuery = question_router.invoke({"question": question})
    assert res.datasource == "websearch"
