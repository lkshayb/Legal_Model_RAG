import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#Using phi-2 from microsoft as it is a light weight model and can be used on low end GPUs
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

#Configuration for the quantization of the model for my GPU level
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

#tokenizer to convert txt into tokens(using autotokenizer here)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

#Loading the model locally
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

#dfl prompt for the model to answer the queries
def generate_answer(context, question):
    prompt = f"""
You are an AI legal assistant evaluating a legal case based strictly on Indian law.

You MUST rely ONLY on the information provided in the CONTEXT and QUESTION.

STRICT RULES:
- Do NOT copy legal text verbatim.
- Summarize laws in simple words.
- Only mention laws that appear in the CONTEXT.
- Copy Act names and Section numbers EXACTLY as written in the CONTEXT.
- Do NOT invent or guess any law, section, or precedent.
- Do NOT assume, modify, or add ANY facts beyond what is explicitly stated in the QUESTION.
- If the CONTEXT does NOT clearly support a law or section, say:
  "No authoritative legal provision found in the provided material."
- If uncertain, prefer saying "Insufficient information" rather than guessing.

Return output in valid JSON format with keys:
- Legal Issue: <one-line description of the core legal problem>
- Applicable Law: <Act name – Section number (ONLY if clearly present in CONTEXT, otherwise write "Not found in context")>
- Analysis:
    - Apply ONLY the given facts (do NOT add or change facts)
    - Clearly connect facts → law → conclusion
    - Keep reasoning simple, logical, and grounded
- Ethical Consideration:
    - Briefly discuss fairness, rights, or misuse of trust if relevant
    - If truly not applicable, write "Not significant"
- Final Judgment:
    - Give a clear outcome (e.g., liable / not liable / constitutional / not constitutional)
    - If CONTEXT is insufficient, write "Cannot be determined from context"
- Confidence:
    - High → Law + facts clearly supported by CONTEXT
    - Medium → Partial support
    - Low → Weak or missing support

Return ONLY valid JSON in ONE LINE.

Do NOT include:
- explanations
- markdown (no ```json)
- extra text
- line breaks inside JSON

If unsure about law, write:
"Applicable Law": "Not found in context"

Keep output under 120 words.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    #load tthe input into tokenizer
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    #generate output with max 300 tokens to control length
    output = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.2,
        do_sample=False,
        repetition_penalty=1.2,
        eos_token_id=tokenizer.eos_token_id,
        # pad_token_id=tokenizer.eos_token_id
    )
    generated_tokens = output[0]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)
