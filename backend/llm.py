import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

#Using phi-2 from microsoft as it is a light weight model and can be used on low end GPUs
MODEL_ID = "microsoft/phi-2"

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

Rules:
1. Answer ONLY using the legal text provided in the CONTEXT.
2. Do NOT use outside knowledge or make assumptions.
3. Do NOT invent laws, Act names, or Section numbers.
4. If citing a law, copy the Act name and Section number exactly from the context.
5. If the answer is not present in the context, respond exactly with:
"No authoritative legal provision found in the provided material."
6. Summarize the law briefly. Do NOT copy large blocks of the context.
7. Do NOT generate follow-up questions, examples, or extra commentary.

Format your answer exactly like this:

Legal Issue:
(short description)

Applicable Law:
Act Name – Section Number

Legal Position:
(brief explanation of what the law states)

Limitations:
(if the context does not fully answer the question)

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
        do_sample=False
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
