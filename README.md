# 📊 Ask-BI

A Retrieval Augmented Generation (RAG) system that provides AI-powered business intelligence insights about sales data through natural language queries.

## Overview

This application uses a RAG architecture to enable users to query sales data through natural language questions. The system retrieves relevant information from a pre-processed knowledge base and generates comprehensive answers with visualizations.

## Features

- **Natural Language Queries**: Ask questions about sales data in plain English
- **Data Visualization**: Automatically generates and displays relevant charts and plots
- **Conversation Memory**: Maintains chat history to provide context-aware responses
- **Comprehensive Analytics**: Answers questions about:
  - Sales trends by region, product, or time period
  - Customer demographics and preferences
  - Satisfaction ratings and purchase patterns
  - Comparative analysis between different segments

## Installation

### Prerequisites
- Python 3.10+ 
- Conda (recommended for environment management)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ask-bi.git
   cd ask-bi
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n ask-bi python=3.10
   conda activate ask-bi
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the project root with the following variables:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   MODEL_NAME=gpt-4o-mini
   EMBEDDING_MODEL=text-embedding-3-small
   ```

## Usage

### 1. Generate Statistical Summaries

Process sales data to create statistical summaries and visualizations:

```bash
python create_statistical_summary.py
```

This script:
- Calculates sales statistics (by product, region, year, quarter, etc.)
- Creates documents summarizing the findings
- Generates descriptive plots using matplotlib and seaborn for each document

### 2. Index Documents

Build a searchable knowledge base from the documents:

```bash
python index_documents.py
```

This script:
- Embeds and indexes documents using OpenAIEmbeddings, LangChain, and FAISS
- Stores paths to corresponding plots as metadata
- Creates the knowledge base for the application

### 3. Evaluate System Performance (Optional)

```bash
python evaluation.py
```

This script evaluates the RAG system performance using the QAEvalChain class from LangChain.

### 4. Launch the Application

```bash
streamlit run app.py
```

The Streamlit UI allows you to:
- Ask questions about sales data in natural language
- View responses with relevant visualizations
- See conversation history
- Choose from sample questions in the sidebar

## Project Structure

```
├── app.py                 # Streamlit application
├── create_statistical_summary.py  # Generate statistics and plots
├── index_documents.py     # Create searchable knowledge base
├── evaluation.py          # Evaluate RAG system performance
├── Datasets/              # Sales data and evaluation datasets
│   ├── sales_data.csv     # Raw sales data
│   ├── evaluation_dataset.jsonl  # Test questions and answers
│   └── PDF Folder/        # PDF documents
├── faiss_index/           # Vector database
│   ├── index.faiss        # FAISS index
│   └── index.pkl          # Index metadata
├── sales_data_statistics/ # Generated statistics and plots
├── logs/                  # Application logs
└── utils/                 # Helper functions
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.