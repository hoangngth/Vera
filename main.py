from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

template = """
You are a helpful querky assistant.

Here is the conversation so far: {context}

Question: {question}
Provide a detailed and informative answer.
"""

model = OllamaLLM(model="llama3")
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def handle_conversation():
    context = ""
    print("Vera: Hi! I'm Vera, your querky assistant. How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ['exit', 'quit']:
            print("Vera: Goodbye!")
            break
        result = chain.invoke({"context": context, "question": user_input})
        print("Vera:", result)
        context += f"\nYou: {user_input}\nVera: {result}"

if __name__ == "__main__":
    handle_conversation()