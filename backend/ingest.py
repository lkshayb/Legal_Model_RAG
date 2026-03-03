from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_text(path):
    with open(path, "r", encoding="utf-8") as f: return f.read()

constitution_text = load_text("./cleaned_text/constitution.txt")
bns_text = load_text("./cleaned_text/bns.txt")

#we break the txt into parts for ease
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=5) 
constitution_docs = text_splitter.create_documents([constitution_text],metadatas=[{"source": "Indian Constitution"}])
bns_docs = text_splitter.create_documents([bns_text],metadatas=[{"source": "Bharatiya Nyaya Sanhita"}])

#final document
documents = constitution_docs + bns_docs

print(f"No. of chunks parted from text : {len(documents)}")

#Load model for embeddings
# using miniLM-L6-v2
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Creating the actual vectore database, using FAISS for vector DB
vectorstore = FAISS.from_documents(documents,embeddings)
os.makedirs("vectorstore",exist_ok=True)
vectorstore.save_local("vectorstore")

print("Succecfully indexed into FAISS for RAG")
