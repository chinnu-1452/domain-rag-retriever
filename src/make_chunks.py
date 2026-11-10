import json, time, random, os
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

topics = [
    "casual leave", "sick leave", "privilege/earned leave", "maternity leave",
    "paternity leave", "notice period and resignation", "probation and confirmation",
    "gratuity eligibility and calculation", "provident fund contribution",
    "ESI benefits", "work from home policy", "attendance and late coming",
    "overtime pay", "salary and payslip", "performance appraisal",
    "code of conduct", "POSH (sexual harassment) policy", "grievance redressal",
    "disciplinary action and termination", "full and final settlement",
    "leave encashment", "bonus and incentives", "travel reimbursement",
    "background verification", "notice period buyout", "internship stipend rules",
    "holiday list and optional holidays", "relocation policy",
    "insurance and medical benefits", "exit interview and handover",
]
styles = ["IT services company", "manufacturing company", "startup",
          "bank", "hospital", "retail chain", "government PSU",
          "BPO/call centre", "pharma company", "NGO"]

os.makedirs("data", exist_ok=True)
n = 0
with open("data/chunks.jsonl", "w") as f:
    for t in topics:
        for s in random.sample(styles, 10):
            prompt = (
                f"Write a realistic HR policy section (150-200 words) on '{t}' "
                f"for an Indian {s}. Include specific numbers (days, percentages, "
                f"limits), eligibility rules and exceptions. Plain text only, no title."
            )
            try:
                r = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    max_tokens=1500,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.9,
                )
                text = r.choices[0].message.content.strip()
                f.write(json.dumps({"id": f"syn_{n}", "source": f"{s}|{t}", "text": text}) + "\n")
                n += 1
                print(n, t, "|", s)
                time.sleep(1)
            except Exception as e:
                print("error", e)
                time.sleep(5)
print("Done:", n, "chunks")
