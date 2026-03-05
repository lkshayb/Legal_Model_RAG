import re

with open("./cleaned_text/bns.txt", "r", encoding="utf-8") as f:
    text = f.read()

sections = re.split(r"\n(\d+\.)", text)

documents = []

for i in range(1, len(sections), 2):
    section_number = sections[i].replace(".", "")
    section_text = sections[i+1]

    doc = f"""
Act: Bharatiya Nyaya Sanhita
Section: {section_number}

Text:
{section_text.strip()}
"""

    documents.append(doc)

with open("./cleaned_text/bns.txt", "w", encoding="utf-8") as f:
    for d in documents:
        f.write(d + "\n\n---\n\n")