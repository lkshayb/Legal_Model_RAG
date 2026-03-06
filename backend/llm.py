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
You are an assistant for Indian legal research.

Use ONLY the information in the CONTEXT to answer the question.

Rules:
- Do NOT copy the legal text verbatim.
- Summarize the law in simple words.
- Only mention laws that appear in the CONTEXT.
- Copy the Act name and Section number exactly as written.
- If the answer is not present in the CONTEXT, say:
"No authoritative legal provision found in the provided material."
Return the answer in EXACTLY this format:

Legal Issue: <short description>

Applicable Law: <Act name – Section number>

Legal Position: <brief explanation of the law in simple words>

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
        do_sample=False,
        repetition_penalty=1.2,
        # eos_token_id=tokenizer.eos_token_id,
        # pad_token_id=tokenizer.eos_token_id
    )
    generated_tokens = output[0]
    return tokenizer.decode(generated_tokens, skip_special_tokens=True)
