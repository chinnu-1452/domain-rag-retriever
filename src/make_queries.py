import json, os, time
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

chunks = [json.loads(l) for l in open("data/chunks.jsonl", encoding="utf-8")]

# already done ayinavi skip chesthundi
done = set()
if os.path.exists("data/train_pairs.jsonl"):
    for l in open("data/train_pairs.jsonl", encoding="utf-8"):
        done.add(json.loads(l)["chunk_id"])

with open("data/train_pairs.jsonl", "a", encoding="utf-8") as f:
    for i, c in enumerate(chunks):
        if c["id"] in done:
            continue
        company = c["source"].split("|")[0]
        prompt = (
            f"This HR policy text is from an Indian {company}. Write 3 different "
            "questions an employee of THIS company might ask that ONLY this text "
            "answers. Each question must include a specific detail from the text "
            "(a number, a named rule, or a term unique to this policy) so it cannot "
            "match other companies' policies. Use simple words, do not copy "
            "sentences. Output exactly 3 lines, one question per line, no "
            "numbering.\n\nTEXT:\n" + c["text"]
        )
        try:
            r = client.chat.completions.create(
                model="openai/gpt-oss-120b",
                reasoning_effort="low",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                temperature=0.7,
            )
            lines = [x.strip("-•0123456789. ").strip()
                     for x in r.choices[0].message.content.strip().split("\n")
                     if x.strip()]
            for q in lines[:3]:
                f.write(json.dumps({"query": q, "chunk_id": c["id"]}, ensure_ascii=False) + "\n")
            print(i + 1, "/", len(chunks))
            time.sleep(1)
        except Exception as e:
            print("error", e)
            time.sleep(5)
print("Done")