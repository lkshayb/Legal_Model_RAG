from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from llm import generate_answer

#load model for embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

#Loading the vectore store(FAISS data) for RAG
vectorstore = FAISS.load_local("vectorstore",embeddings,allow_dangerous_deserialization=True)

#searching the vectorestore for generation
def retrieve_context(query, k=4):
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs[:2]])
    return context

def askQuestion(query):
    context = retrieve_context(query)
    answer = generate_answer(context, query)
    return {
        "context":context,
        "answer":answer
    }

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
