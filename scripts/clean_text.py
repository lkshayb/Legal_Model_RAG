import re

def clean_text(txt):
    txt = re.sub(r"\n+","\n",txt)
    txt = re.sub(r"Page \d+","",txt)
    return txt




if __name__ == "__main__":
    with open("./cleaned_text/bns.txt","r", encoding="utf-8") as f: raw_text = f.read()
    cleaned = clean_text(raw_text)

    with open("./cleaned_text/bns.txt","w", encoding='utf-8') as f: f.write(cleaned)
    print("cleaning of the file is now completed")