from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from typing import List
from .models import initialize_llm

 # Initialize LLM
llm = initialize_llm()

def chat_history_to_str(chat_history: List):
    """
    Convert chat history to string format.
    
    Args:
        chat_history: The chat history object
    Returns:
        str: The formatted chat history
    """
    
    chat_history_text = ""
    if chat_history and len(chat_history) > 0:
        for message in chat_history:
            role = "User" if message["role"] == "user" else "Assistant"
            chat_history_text += f"{role}: {message['content']}\n\n"

    return chat_history_text


def qa_search(query: str, vectorstore, chat_history=None, k=5):
    """
    Answer questions about the codebase using the vectorstore and LLM.
    
    Args:
        query (str): The question to answer
        vectorstore: The vectorstore containing the indexed documents
        chat_history: The chat history
        k (int): The number of documents to retrieve
    Returns:
        str: The answer to the question with sources
    """
    
    # Create retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    
    # Define prompt template    
    template = """You are a helpful assistant that provides accurate information based on the given context.
    
    Chat History:
    {chat_history}

    Context:
    {context}
    
    Question:
    {question}
    
    Answer the question based only on the provided context.
    If the context is incomplete or vague, do your best to respond using the available information. 
    If the question appears incomplete (e.g. starts with “Can you find”, “Do you”, or cuts off unexpectedly), politely inform the user that the question seems incomplete and ask them to rephrase. 
    If the question refers to previous questions or responses, use the chat history to understand what the user is referring to.
    If a full answer isn’t possible, clearly state what’s missing and politely ask the user for clarification. 
    Do not invent or assume facts that are not explicitly present in the context. 
    Keep responses brief and focused (no more than 300 words). 
    Avoid repetition and long paragraphs. 
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create document chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Create retrieval chain
    retrieval_chain = (
        {
            "context": RunnableLambda(lambda x: retriever.invoke(x["question"])), 
            "question": RunnablePassthrough(), 
            "chat_history": RunnablePassthrough()
        }
        | document_chain
        | StrOutputParser()
    )
    
    # Run the chain
    try:
        # First, retrieve documents
        retrieved_docs = retriever.invoke(query)
        
        # Extract metadata from each document
        docs_metadata = []
        for doc in retrieved_docs:
            # Create a copy of the metadata to avoid modification issues
            meta = dict(doc.metadata) if hasattr(doc, "metadata") and doc.metadata else {}
            
            # Add similarity score if available
            if hasattr(doc, "similarity_score"):
                meta["similarity_score"] = doc.similarity_score
                
            docs_metadata.append(meta)
        
        # Get answer from LLM
        answer = retrieval_chain.invoke(
            {
                "question": query,
                "chat_history":chat_history_to_str(chat_history)
            }
        )
        
        # Return both the formatted answer and the document metadata
        return {
            "answer": answer,
            "document_metadata": docs_metadata,
            "raw_answer": answer,
            "retrieved_count": len(retrieved_docs)
        }
        
    except Exception as e:
        error_message = f"Error querying the model: {str(e)}"
        return {
            "answer": error_message,
            "document_metadata": [],
            "raw_answer": error_message,
            "error": str(e)
        }
    
if __name__ == "__main__":
    # Example usage
    import pprint
    from utils.vectorstore import get_vectorstore
    vectorstore = get_vectorstore()
    
    if vectorstore:
        query = "Tell me about the spending habits of the young customers."
        result = qa_search(query, vectorstore)
        pprint.pprint(result)
    else:
        print("Vectorstore not initialized.")
