from context import retrieve_context
from llm import generate_answer
from Cases import Cases
import pandas as pd
import time
import json

# PROMPT CHANGE REQUIRED IN CASE OF EVALUATION PROGRAM

def askQuestion(query):
    context = retrieve_context(query)
    answer = generate_answer(context, query)
    return {answer}


#EVALUATION MODE | RP ONLY
def clean_output(output):
    if "ANSWER:" in output:
        output = output.split("ANSWER:")[-1]
    if "{" in output:
        output = output[output.index("{"):]
    if "}" in output:
        output = output[:output.rindex("}")+1]

    return output.strip()
def extract_fields(output):
    try:
        # trying direct JSON parse
        data = json.loads(output)
        return {
            "legal_issue": data.get("Legal Issue", ""),
            "law": data.get("Applicable Law", ""),
            "analysis": data.get("Analysis", ""),
            "ethical": data.get("Ethical Consideration", ""),
            "final_judgment": data.get("Final Judgment", ""),
            "confidence": data.get("Confidence", "")
        }
    except:
        # fallback/  Handleing
        return {
            "legal_issue": "",
            "law": "",
            "analysis": output[:500], 
            "ethical": "",
            "final_judgment": "",
            "confidence": ""
        }
def run_evaluation():
    results = []

    for case in Cases:
        print(f"Running {case['id']}...")

        try:
            response = askQuestion(case["facts"])
            raw_output = response["answer"]
            cleaned_output = clean_output(raw_output)
            parsed = extract_fields(cleaned_output)

            row = {
                "case_id": case["id"],
                "description": case["description"],
                "type": case["type"],
                "facts": case["facts"],
                "human_judgement": case["human_judgement"],
                "model_output": cleaned_output,
                **parsed
            }

            results.append(row)

            df = pd.DataFrame(results)
            df["accuracy"] = ""
            df["reasoning_score"] = ""
            df["ethical_score"] = ""
            df["hallucination"] = ""

            df.to_csv("legal_eval_results.csv", index=False)

            time.sleep(1)

        except Exception as e:
            print(f"Error in {case['id']}: {e}")

    print("\nEvaluation complete. Saved to legal_eval_results.csv")

