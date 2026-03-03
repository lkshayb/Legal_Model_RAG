from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llm import generate_answer

#load model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Loading the vectore store(FAISS data) for RAG
vectorstore = FAISS.load_local("vectorstore",embeddings,allow_dangerous_deserialization=True)

#searching the vectorestore for generation
def retrieve_context(query, k=4):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([f"[{d.metadata['source']}]\n{d.page_content}" for d in docs])



if __name__ == "__main__":
    while True:
        query = input("\nAsk a legal question (or type 'exit'): ")
        if query.lower() == "exit":
            break

        context= retrieve_context(query)
        print("\nRetrieved Context ==>\n")
        print(context)
        print("\n")

        answer = generate_answer(context, query)

        print("\nAnswer :")
        print(answer)
