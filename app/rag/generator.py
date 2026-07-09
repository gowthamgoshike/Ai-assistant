from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

def generate_answer(query: str, context: list[str]):
    llm = ChatOllama(model="llama3.1", temperature=0)
    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the context to answer the question. 
    If not in context, say 'I don't have enough information.'
    
    Context: {context}
    Question: {question}
    """)
    chain = prompt | llm
    return chain.invoke({"context": "\n\n".join(context), "question": query})