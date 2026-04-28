from context import retrieve_context
from llm import generate_answer

if __name__ == "__main__":
    while True:
        query = input("\nAsk a legal question (type 'exit' to close): ")
        if query.lower() == "exit":
            break

        context = retrieve_context(query)
        print("\n****Retrieved Context****\n")
        print(context)
        print("\n****End of Context****\n")

        answer = generate_answer(context, query)

        print("\nAnswer :")
        print(answer)