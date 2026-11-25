import json
import time

from app.agent import ask_agent

with open("evals/eval_set.json") as f:
    eval_set = json.load(f)

route_score = 0
section_score = 0
keyword_score = 0
kb_total = 0   # questions that need the filing

for case in eval_set:
    print("\n=== ", case["question"], " ===")

    result = ask_agent(case["question"])

    # 1) Routing: did it go to kb or direct correctly?
    route = "direct" if result["source_used"] == "direct" else "kb"
    route_ok = route == case["route"]
    route_score += route_ok
    print("Route correct:", route_ok)

    if case["route"] == "kb":
        kb_total += 1

        # 2) Retrieval: did we get chunks from the right section?
        sections = [doc.metadata["section"] for doc in result["kb_docs"]]
        section_ok = case["section"] in sections
        section_score += section_ok
        print("Section correct:", section_ok)

        # 3) Answer: does it contain the keywords?
        answer = result["answer"].lower()
        keywords_ok = all(word.lower() in answer for word in case["keywords"])
        keyword_score += keywords_ok
        print("Keywords correct:", keywords_ok)

    time.sleep(15)   # groq rate limits

print("\n===== RESULTS =====")
print(f"Routing:  {route_score}/{len(eval_set)}")
print(f"Section:  {section_score}/{kb_total}")
print(f"Keywords: {keyword_score}/{kb_total}")
