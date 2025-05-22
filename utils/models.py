from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os

load_dotenv()

def initialize_embeddings():
    """
    Initialize the embedding model.
    """
    
    return OpenAIEmbeddings(
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_EMBEDDING_MODEL"),
        disallowed_special=()
    )

def initialize_llm():
    """
    Initialize the LLM model.
    """
    
    llm = ChatOpenAI(
        model_name=os.getenv("OPENAI_MODEL"),
        temperature=0.0
    )
    
    return llm
