import argparse
import json
from utils.vectorstore import save_vectorstore
from utils.document_processor import index_documents
from langchain.schema import Document

def parse_args():
    """
    Parse command line arguments.
    """
    
    parser = argparse.ArgumentParser(description="Index documents into a vector store.")
    parser.add_argument(
        "--data-path", 
        type=str, 
        default="sales_data_statistics/documents.json", 
        help="Path to the JSON file containing the documents."
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    
    args = parse_args()

    # Load data
    with open(args.data_path, "r") as f:
        data = json.load(f)
    
    documents = []

    for record in data:
        documents.append(
            Document(
                page_content = record["text"],
                metadata = record["metadata"]
            )
        )

    # Index documents
    vectorstore = index_documents(documents)
    save_vectorstore(vectorstore)

    print("Documents indexed successfully.")


    