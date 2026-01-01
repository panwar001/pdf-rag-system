
from dotenv import load_dotenv
from langchain_ollama import ChatOllama


def get_llm():
    load_dotenv()
    llm = ChatOllama(
        model="llama3",
        temperature=0
    )
    return llm
