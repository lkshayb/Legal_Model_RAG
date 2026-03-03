import pdfplumber

def pdf_to_text(pdf_path, output_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = text + page.extract_text() + "\n"


    with open(output_path,"w", encoding="utf-8") as f: f.write(text)

pdf_to_text("./data/bare_acts/ipc.pdf","./cleaned_text/ipc.txt")