from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llm import generate_answer
from Cases import Cases
import pandas as pd
import time

# -------------------------------
# LOAD EMBEDDINGS + VECTORSTORE
# -------------------------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)

# -------------------------------
# RETRIEVAL
# -------------------------------
def retrieve_context(query, k=4):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs[:2]])

# -------------------------------
# CORE PIPELINE (USED EVERYWHERE)
# -------------------------------
def askQuestion(query):
    context = retrieve_context(query)
    answer = generate_answer(context, query)
    return {
        "context": context,
        "answer": answer
    }

# -------------------------------
# FIELD EXTRACTION (for evaluation)
# -------------------------------
import json

def clean_output(output):
    # Remove prompt if repeated
    if "ANSWER:" in output:
        output = output.split("ANSWER:")[-1]

    # Remove everything before first {
    if "{" in output:
        output = output[output.index("{"):]

    # Remove trailing garbage after last }
    if "}" in output:
        output = output[:output.rindex("}")+1]

    return output.strip()

def extract_fields(output):
    try:
        # Try direct JSON parse
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
        # fallback (if JSON fails)
        return {
            "legal_issue": "",
            "law": "",
            "analysis": output[:500],  # keep partial for debugging
            "ethical": "",
            "final_judgment": "",
            "confidence": ""
        }
# -------------------------------
# EVALUATION MODE (NEW)
# -------------------------------
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

            # SAVE AFTER EACH CASE (IMPORTANT)
            df = pd.DataFrame(results)
            df["accuracy"] = ""
            df["reasoning_score"] = ""
            df["ethical_score"] = ""
            df["hallucination"] = ""

            df.to_csv("legal_eval_results.csv", index=False)

            time.sleep(1)

        except Exception as e:
            print(f"Error in {case['id']}: {e}")

    print("\n✅ Evaluation complete. Saved to legal_eval_results.csv")

# -------------------------------
# MAIN ENTRY
# -------------------------------
if __name__ == "__main__":
    mode = input("Choose mode (chat / eval): ").strip().lower()

    if mode == "eval":
        run_evaluation()

    else:
        while True:
            query = input("\nAsk a legal question (or type 'exit'): ")
            if query.lower() == "exit":
                break

            context = retrieve_context(query)

            print("\n****Retrieved Context****\n")
            print(context)
            print("\n****End of Context****\n")

            answer = generate_answer(context, query)

            print("\nAnswer :")
            print(answer)