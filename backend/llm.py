import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

MODEL_ID = "microsoft/phi-2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)
def generate_answer(context, question):
    prompt = f"""
You are an AI legal assistant. Decide the case using ONLY the given CONTEXT and QUESTION.

Rules:
- Do NOT add facts.
- Use ONLY laws present in CONTEXT.
- If law is missing or unclear → say "Insufficient information".
- No explanations, no reasoning, no analysis.

Output:
- ONE LINE ONLY
- Give final judgement with the act under which you are making the decisions, also mention subparts if required

Do NOT output anything else.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    output = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=False,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
    )
    generated_tokens = output[0]
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    if "ANSWER:" in decoded:
        response = decoded.split("ANSWER:")[-1].strip()
    else:
        response = decoded.strip()
    
    for i in range(len(response) - 1, -1, -1):
        if response[i] == ".":
            response = response[:i+1]
            break
    return response