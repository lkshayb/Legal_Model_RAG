import customtkinter as ctk
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


def generate_Res(query):
    docs = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in docs[:2]])
    answer = generate_answer(context, query)
    return answer


class LegalApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Practicum Project - Legal RAG Based System")
        

        # Title
       

        # Chat display
        self.chatbox = ctk.CTkTextbox(self, width=550, height=500)
        self.chatbox.pack(padx=15, pady=15)
        self.chatbox.configure(state="disabled")
        self.chatbox.tag_config("user_tag", foreground="#4FC3F7")
        self.chatbox.tag_config("ai_tag", foreground="#81C784")
        self.chatbox.tag_config("thinking_tag", foreground="gray")

        # Input area
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(pady=10)
        self.entry = ctk.CTkEntry(self.input_frame, width=400, placeholder_text="Enter your legal query...")
        self.entry.pack(side="left", padx=10, pady=5)
        self.ask_button = ctk.CTkButton(self.input_frame, text="Ask", command=self.handle_query)
        self.ask_button.pack(side="left",padx=5)
        

    def handle_query(self):

        
        query = self.entry.get().strip()
        if not query:
            return
        
        self.chatbox.configure(state="normal")
        self.chatbox.insert("end","\n")
        self.chatbox.insert("end", "You: ", "user_tag")
        self.chatbox.insert("end", f"{query}\n")
        self.chatbox.insert("end", "\nAI: Thinking...\n","thinking_tag")
        self.chatbox.update()
        response = generate_Res(query)
        self.chatbox.delete("end-2l", "end-1l")
        self.chatbox.insert("end","AI: ","ai_tag")
        self.chatbox.insert("end", f"{response}\n\n")
        self.chatbox.configure(state="disabled")
        self.entry.delete(0, "end")
        self.chatbox.yview("end")
    

if __name__ == "__main__":
    app = LegalApp()
    app.mainloop()