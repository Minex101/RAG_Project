import json
import time

from app.agent import app, route_question
from app.components import get_retriever

K = 3

with open("evals/eval_set.json") as f:
    eval_set = json.load(f)

retriever = get_retriever(k=K)   # build once, not every question

route_correct = 0
retrieval_hits, retrieval_total = 0, 0
keyword_hits, keyword_total = 0, 0
rows = []

for case in eval_set:
    q = case["question"]
    print(f"\n=== {q} ===")

    # 1) Routing
    route = route_question({"question": q})["source_used"]
    route_ok = route == case["route"]
    route_correct += route_ok

    # 2) Retrieval hit@k (only for kb questions)
    hit = None
    if case["section"]:
        docs = retriever.invoke(q)
        sections = [d.metadata.get("section") for d in docs]
        hit = case["section"] in sections
        retrieval_hits += hit
        retrieval_total += 1
        print("Retrieved sections:", sections)

    # 3) Full agent run + keyword check
    result = app.invoke({
        "question": q, "current_query": q, "kb_docs": [], "web_results": "",
        "kb_grade": "", "web_grade": "", "answer": "", "source_used": "", "retry_count": 0,
    })
    answer = result["answer"].lower()

    kw_ok = None
    if case["keywords"]:
        kw_ok = all(k.lower() in answer for k in case["keywords"])
        keyword_hits += kw_ok
        keyword_total += 1

    rows.append({
        "question": q,
        "route_ok": route_ok,
        "retrieval_hit": hit,
        "keywords_ok": kw_ok,
        "source_used": result["source_used"],
    })

    time.sleep(1)   # groq rate limits

# ---- Report ----
print("\n\n===== RESULTS =====")
for r in rows:
    print(r)

print(f"\nRouting accuracy: {route_correct}/{len(eval_set)}")
print(f"Retrieval hit@{K}: {retrieval_hits}/{retrieval_total}")
print(f"Answer keyword match: {keyword_hits}/{keyword_total}")

with open("evals/results.json", "w") as f:
    json.dump(rows, f, indent=2)
