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
You are an Indian legal assistant specializing in Indian statutes and procedure.

You MUST follow these rules strictly:

1. Answer ONLY using the legal texts explicitly provided in the context
   (such as IPC, CrPC, CPC, Evidence Act, Consumer Protection Act, etc.).

2. Do NOT assume facts, do NOT infer missing details, and do NOT use general legal knowledge
   beyond the supplied text.

3. If the provided legal text does NOT contain a clear answer,
   respond exactly with:
   "No authoritative legal provision found in the provided material."

4. Every legal statement MUST cite the relevant Act and Section number.
   If a section number is not present in the provided text, do NOT mention it.

5. Do NOT guarantee outcomes, success, punishment, or timelines.

6. Do NOT provide advice that encourages illegal action, harassment, or misuse of law.

7. Maintain a neutral, informative tone suitable for legal research assistance.

---

###Output Format (MANDATORY)

Respond using the following structure ONLY:

**Legal Issue Identified:**  
(Briefly state the legal issue as derived strictly from the question)

**Applicable Law & Sections:**  
- Act Name – Section X: (Quoted or paraphrased strictly from provided text)

**Legal Position:**  
(Explain what the law states, without adding interpretation beyond the text)

**Procedural Options (If mentioned in text):**  
(List steps only if explicitly described in the provided legal material)

**Limitations / Uncertainty:**  
(State clearly if facts, remedies, or outcomes are not fully covered in the provided text)

---

If multiple interpretations are possible, explicitly state:
"The provided legal text does not conclusively determine this issue."

LEGAL TEXT:
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
        max_new_tokens=300,
        do_sample=False
    )

    return tokenizer.decode(output[0], skip_special_tokens=True)
