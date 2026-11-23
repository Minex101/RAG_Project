from typing import List, Literal, TypedDict
from pydantic import BaseModel, Field
from langchain_core.documents import Document 
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

from app.components import llm, get_retriever, web_search

class RouteDecision(BaseModel):
    route: Literal["kb", "direct"] = Field(
        description="Use kb for question needing Agentic RAG docs; direct for greeting/simple chat."
        )

class EvidenceGrade(BaseModel):
    grade: Literal["good", "weak"] = Field(
        description="good means evidence can answer the question; weak means not enough evidence to answer the question."
        )

class AgentState(TypedDict):
    question: str
    current_query: str
    kb_docs: List[Document]
    web_results: str
    kb_grade: str
    web_grade: str
    answer: str
    source_used: str
    retry_count: int

# -- Nodes --
# -- NODE 1: Route Question --
router_llm = llm.with_structured_output(RouteDecision, method="json_mode")

def route_question(state: AgentState):
    question = state["question"]
    decision = router_llm.invoke(f'''
        You are a routing LLM. 
        Route to "kb" if the question is about a company's business, products, segments,
        risk factors, competition, results, revenue, margins, outlook or anything in its SEC 10-K filing.
        Route to "direct" only for greetings or simple chat.

        Question: {question}

        Return your response as valid JSON.
        Example:
        {{
            "route": "kb"
        }}
    ''')
    
    print("[ROUTER]", decision.route)
    
    return {
        "current_query": question,
        "source_used": decision.route
    }

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    if state["source_used"] == "kb":
        return "retrieve_kb"
    return "direct_answer"

# -- NODE 2: Retrieve from KB Docs --

def retrieve_kb(state: AgentState):
    query = state["current_query"]
    docs = get_retriever().invoke(query)

    print("[KB Retriever] Query:", query)
    print(f"[KB Retriever] Retrieved: {len(docs)} chunks")

    return {"kb_docs": docs}

# -- NODE 3: Grade KB Evidence --

kb_grade_llm = llm.with_structured_output(EvidenceGrade, method="json_mode")

def grade_kb_evidence(state: AgentState):
    question = state["question"]

    context = "\n\n".join(
        f"[{doc.metadata.get('ticker')} {doc.metadata.get('form')} | "
        f"{doc.metadata.get('section')} {doc.metadata.get('section_name')} | "
        f"filed {doc.metadata.get('filing_date')}]\n{doc.page_content}"
        for doc in state["kb_docs"]
    )

    grade = kb_grade_llm.invoke(f'''
        You are an evidence grading LLM.
        Grade "good" only if the filing excerpts directly answer the question.
        If the question asks for a number that is not in the excerpts, grade "weak".
        
        Question: 
        {question}

        Private KB Evidence:
        {context}

        Can the evidence answer the question? Grade as "good" if it can answer.
        return "weak" if it cannot answer or is incomplete.

        return your response as valid JSON.
        Example:
        {{"grade": "good"}}
    ''')

    print("[KB Grader] Grade:", grade.grade)

    return {"kb_grade": grade.grade}

def decide_after_kb_grader(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    if state["kb_grade"] == "good":
        return "generate_from_kb"
    return "search_web"

# -- NODE 4: Tavilly Web Search --

def search_web(state: AgentState):
    query = state["current_query"]

    print("[Tavily Search] Query:", query)

    result = web_search.invoke({"query": query})

    if isinstance(result, dict):
        answer = result.get("answer", "")
        results = result.get("results", [])
        lines = []
        if answer:
            lines.append(f"Tavily answer: {answer}")

        for item in results:
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")

        web_text = "\n\n".join(lines) if lines else str(result)

    else:
        web_text = str(result)

    print("[Tavilly Search] Result characters:", len(web_text))

    return {
        "web_results" : web_text,
        "source_used" : "web",
    }

# -- NODE 5: Grade Web Evidence --
web_grader_llm = llm.with_structured_output(EvidenceGrade, method="json_mode")

def grade_web_evidence(state: AgentState):
    question = state["question"]
    web_results = state["web_results"]

    grade = web_grader_llm.invoke(f'''
    You are an evidence grader    

    Question:
    {question}

    Web search evidence:
    {web_results}

    Can this web evidence answer the question?
    Return "good" if it can answer.
    Return "weak" if it cannot answer or is incomplete.

    Return your answer as a valid JSON.
    Example: 
    {{"grade": "good"}}
    ''')

    print("[Web Grader]", grade.grade)

    return {"web_grade": grade.grade}

MAX_RETRIES = 1

def decide_after_web_grade(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "answer_insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"

    if state["retry_count"] < MAX_RETRIES:
        return "rewrite_query"

    return "answer_insufficient"

# -- NODE 6: Rewrite the Query --

def rewrite_query(state: AgentState):
    question = state["question"]
    retry_count = state["retry_count"] + 1

    rewritten = llm.invoke(f'''
    Rewrite the question for better retrieval and web search.

    Rules:
    - Preserve the original intent
    - Make it specific and search friendly
    - Do not answer
    - Return only the rewritten query

    Original question:
    {question}
    ''').content.strip()

    print("[Rewriter]", rewritten)

    return {
        "current_query": rewritten,
        "retry_count": retry_count
    }

# -- NODE 7: Generate from private KB --

def generate_from_kb(state: AgentState):
    question = state["question"]

    context = "\n\n".join(
        f"[{doc.metadata.get('ticker')} {doc.metadata.get('form')} | "
        f"{doc.metadata.get('section')} {doc.metadata.get('section_name')} | "
        f"filed {doc.metadata.get('filing_date')}]\n{doc.page_content}"
        for doc in state["kb_docs"]
    )

    answer = llm.invoke(f'''
    You are an equity research assistant.
    Answer using ONLY the SEC filing excerpts below.

    Rules:
    - Do not invent facts or numbers. Only quote numbers that appear in the excerpts.
    - Cite each claim with its section, e.g. (AAPL 10-K, Item 1A).
    - If the excerpts only partly answer, say what is missing.
    - Do not give buy/sell/hold advice.

    Question:
    {question}

    Private KB context:
    {context}
    ''').content

    return {
        "answer": answer,
        "source_used": "private_kb"
        }

# -- NODE 8: Generate from Web Search --

def generate_from_web(state: AgentState):
    question = state["question"]
    web_context = state["web_results"]

    answer = llm.invoke(f'''
    You are a technical instructor
        The private KB was insufficient, so web search was used.
    
        Rules:
        - Beginner-friendly explanation.
        - Do not invent unsupported details.
        - Mention that the answer is based on Tavily web search.
        - Include source type: Web Search.
        - If URLs are present in context, include the most useful URLs.

        Question:
        {question}

        Context:
        {web_context}
    ''').content

    return {
        "answer": answer,
        "source_used": "web_search" 
    }

# -- NODE 9: Direct Answer

def direct_answer(state: AgentState):
    question = state["question"]

    answer = llm.invoke(f'''
    Respond briefly and naturally

    Message:
    {question}
    ''').content

    return {
        "answer": answer,
        "source_used": "direct"
    }

# -- NODE 10: Insufficient Evidence answer --

def answer_insufficient(state: AgentState):
    answer = (
        "I could not find enough reliable evidence in the private knowledge base "
        "or the web search results to answer this confidently. "
        "Please provide more specific documents or rephrase the question."
    )

    return {
        "answer": answer,
        "source_used": "insufficient_evidence"
    }

# -- LANGGRAPH WORKFLOW --

workflow = StateGraph(AgentState)

workflow.add_node("route_question", route_question)
workflow.add_node("retrieve_kb", retrieve_kb)
workflow.add_node("grade_kb_evidence", grade_kb_evidence)
workflow.add_node("search_web", search_web)
workflow.add_node("grade_web_evidence", grade_web_evidence)
workflow.add_node("rewrite_query", rewrite_query)
workflow.add_node("generate_from_kb", generate_from_kb)
workflow.add_node("generate_from_web", generate_from_web)
workflow.add_node("direct_answer", direct_answer)
workflow.add_node("answer_insufficient", answer_insufficient)

workflow.add_edge(START, "route_question")

workflow.add_conditional_edges(
    "route_question",
    route_after_router,
    {
        "retrieve_kb": "retrieve_kb",
        "direct_answer": "direct_answer",
    },
)

workflow.add_edge("retrieve_kb", "grade_kb_evidence")

workflow.add_conditional_edges(
    "grade_kb_evidence",
    decide_after_kb_grader,
    {
        "generate_from_kb": "generate_from_kb",
        "search_web": "search_web",
    },
)

workflow.add_edge("search_web", "grade_web_evidence")

workflow.add_conditional_edges(
    "grade_web_evidence",
    decide_after_web_grade,
    {
        "generate_from_web": "generate_from_web",
        "rewrite_query": "rewrite_query",
        "answer_insufficient": "answer_insufficient",
    },
)

workflow.add_edge("rewrite_query", "retrieve_kb")
workflow.add_edge("generate_from_kb", END)
workflow.add_edge("generate_from_web", END)
workflow.add_edge("direct_answer", END)
workflow.add_edge("answer_insufficient", END)

app = workflow.compile()

print("Agentic RAG graph compiled.")

# -- RUN --

def ask_agent(question: str):
    initial_state = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
    }

    result = app.invoke(initial_state)

    print("\nSOURCE USED:", result["source_used"])
    print("\nANSWER:\n", result["answer"])
    return result


if __name__ == "__main__":
    while True:
        q = input("\nAsk (or 'exit'): ")
        if q.lower() == "exit":
            break
        ask_agent(q)