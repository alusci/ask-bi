from .vectorstore import get_vectorstore, init_vectorstore


def index_documents(documents):
    """Index documents in the vector store"""
    
    vectorstore = get_vectorstore()
    if vectorstore:
        vectorstore.add_documents(documents)
    else:
        vectorstore = init_vectorstore(documents)
    return vectorstore 

