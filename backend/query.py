from llm import generate_answer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Context Retrieval : 
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(
    "vectorstore",
    embeddings,
    allow_dangerous_deserialization=True
)
def retrieve_context(vectorstore,query, k=4):
    docs = vectorstore.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs[:2]])


# run
if __name__ == "__main__":
    while True:
        query = input("\nAsk a legal question (type 'exit' to close): ")
        if query.lower() == "exit":
            break

        context = retrieve_context(vectorstore,query)
        print("\n****Retrieved Context****\n")
        print(context)
        print("\n****End of Context****\n")

        answer = generate_answer(context, query)

        print("\nAnswer :")
        print(answer)