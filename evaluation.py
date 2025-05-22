from langchain.evaluation.qa import QAEvalChain
from langchain_openai import ChatOpenAI
from utils.qa_chain import qa_search
from utils.vectorstore import get_vectorstore
from dotenv import load_dotenv
import pandas as pd


load_dotenv()


if __name__ == "__main__":
    
    # Load evaluation data
    fpath = "Datasets/evaluation_dataset.jsonl"

    df_data = pd.read_json(fpath, lines=True)

    vectorstore = get_vectorstore()

    # Get RAG answers
    rag_answers = []
    for index, row in df_data.iterrows():
        print(row)
        result = qa_search(row["query"], vectorstore, k=6)
        rag_answers.append(result["answer"])

    df_data["rag_answer"] = rag_answers
    
    # Define the LLM
    llm = ChatOpenAI(temperature=0, model="gpt-4")

    # Create the evaluation chain
    qa_eval_chain = QAEvalChain.from_llm(llm)

    # Create prediction and references dictionaries
    predictions, references = [], []
    
    for i, row in df_data.iterrows():
        predictions.append({"result": row["rag_answer"]})
        references.append({"query": row["query"], "answer": row["answer"]})  

    # Score RAG answers
    graded_outputs = qa_eval_chain.evaluate(
       references,
       predictions
    )
        
        
    for i, row in df_data.iterrows():
        print(f"Example {i + 1}")
        print("Question:", row["query"])
        print("Ground Truth:", row["answer"])
        print("Generated:", row["rag_answer"])
        print("Evaluation:", graded_outputs[i]["results"])
        print("-" * 60)


    
    
