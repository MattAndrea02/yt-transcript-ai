import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM, OllamaEmbeddings

load_dotenv()


def get_llm(basic=True):
    if basic:
        return get_ollama_llm()
    else:
        return get_advanced_llm()


def get_ollama_llm() -> OllamaLLM:
    return OllamaLLM(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("OLLAMA_MODEL", "llama3.2"),
            temperature=float(os.getenv("OLLAMA_TEMPERATURE", "0.5")),
        )

def get_advanced_llm():
    ''' 
    Da definire poi modello avanzato se necessario
    '''
    return None

def embedding_model():

    return OllamaEmbeddings(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.getenv("EMBEDDING_MODEL"),
            temperature=0.0
        )
    return None