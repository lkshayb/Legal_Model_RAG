import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#Using phi-2 from microsoft as it is a light weight model and can be used on low end GPUs
MODEL_ID = "microsoft/phi-2"

# #Configuration for the quantization of the model for my GPU level
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_compute_dtype=torch.float16,
#     bnb_4bit_use_double_quant=True,
#     bnb_4bit_quant_type="nf4"
# )

#tokenizer to convert txt into tokens(using autotokenizer here)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

#Loading the model locally
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto"
)

#dfl prompt for the model to answer the queries
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
- Give final judgement with the act under which you are making the decisions

Do NOT output anything else.

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
