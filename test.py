from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#load model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local("vectorstore", embeddings, allow_dangerous_deserialization=True)

docs = list(vectorstore.docstore._dict.values())
print("Total documents stored:", len(vectorstore.docstore._dict))
for i, doc in enumerate(docs[:10],1):
    print(f"\n----- CHUNK {i} -----")
    print("Metadata:", doc.metadata)
    print("Content:\n", doc.page_content)