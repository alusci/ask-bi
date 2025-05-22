import streamlit as st
from utils.qa_chain import qa_search
from utils.vectorstore import get_vectorstore
from utils.streamlit import display_message, clear_chat
import time
import logging
import os
import datetime

# Set up logging
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

current_date = datetime.datetime.now().strftime("%Y-%m-%d")
log_file = os.path.join(log_dir, f"interactions_{current_date}.log")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("bi_assistant")

app_name = "Intelligent BI Assistant"

# Set page configuration
st.set_page_config(
    page_title=app_name,
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Log app startup
logger.info(f"Application started: {app_name}")

# App title
st.title(f"📊 {app_name}")
st.markdown("#### Ask questions about your sales data to get insights and visualizations")

# Initialize session state if not already initialized
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "should_process_query" not in st.session_state:
    st.session_state.should_process_query = False

# Function to format message history for the LLM
def format_chat_history(messages, max_messages=10):
    # Limit to last max_messages
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    # Format messages for the LLM
    formatted_history = []
    for msg in recent_messages:
        role = "user" if msg["role"] == "user" else "assistant"
        formatted_history.append({"role": role, "content": msg["content"]})
    
    return formatted_history

# Get vectorstore in a cached way
@st.cache_resource
def get_cached_vectorstore():
    return get_vectorstore()

# Function to set query and flag for processing
def set_query(query):
    st.session_state.last_query = query
    st.session_state.should_process_query = True
    logger.info(f"Query set: {query}")

# Clear chat history with logging
def clear_chat_with_logging():
    logger.info("Chat history cleared")
    clear_chat()

# Load vectorstore on app start
if st.session_state.vectorstore is None:
    with st.spinner("Loading knowledge base..."):
        st.session_state.vectorstore = get_cached_vectorstore()
    if st.session_state.vectorstore:
        logger.info("Knowledge base loaded successfully")
        st.success("Knowledge base loaded successfully!")
    else:
        logger.error("Failed to load knowledge base")
        st.error("Failed to load knowledge base. Please check your data and try again.")

# Sidebar
st.sidebar.header("About")
st.sidebar.markdown("""
This app uses a Retrieval Augmented Generation (RAG) system to provide insights about sales data.
Ask questions about:
- Sales trends by region, product, or time period
- Customer demographics and preferences
- Satisfaction ratings and purchase patterns
- Comparative analysis between different segments
""")

st.sidebar.header("Sample Questions")
sample_questions = [
    "What are the overall sales trends across all regions?",
    "Which product has the highest average sales value?",
    "How do sales in the East region compare to the West region?",
    "What are the purchasing patterns of younger customers?",
    "Which age group has the highest customer satisfaction?",
    "How have Widget A sales changed over time?"
]

# Use buttons to set predefined questions
for question in sample_questions:
    if st.sidebar.button(question, key=f"sample_{hash(question)}"):
        set_query(question)

# Create a container for the query form to keep it at the top
query_container = st.container()

# Create another container for the latest interaction
latest_interaction_container = st.container()

# Create a container for the conversation history
history_container = st.container()

# Query form in the query container
with query_container:
    # Main query input and buttons
    query = st.text_input("Ask a question about the sales data:", value=st.session_state.last_query, key="query_input")

    # Create columns for the buttons
    col1, col2, col3 = st.columns([1, 1, 4])

    # Run button
    with col1:
        run_button = st.button("Run", key="run_button", type="primary")
        if run_button and query:
            set_query(query)

    # Clear button
    with col2:
        st.button("Clear", key="clear_button", on_click=clear_chat_with_logging)

# Process query when flagged
if st.session_state.should_process_query and st.session_state.last_query:
    query = st.session_state.last_query
    
    # Reset flag
    st.session_state.should_process_query = False
    
    # Add user query to messages
    st.session_state.messages.append({"role": "user", "content": query})
    
    # Log the question
    logger.info(f"USER QUESTION: {query}")
    
    # Show progress
    with st.spinner("Finding relevant information and generating response..."):
        start_time = time.time()
        
        # Format chat history for LLM
        chat_history = format_chat_history(st.session_state.messages[:-1]) # Exclude current query
        
        # Query the vector store with history context
        if st.session_state.vectorstore:
            try:
                result = qa_search(query, st.session_state.vectorstore, chat_history, k=6)
                
                # Log the answer and metadata
                logger.info(f"AI ANSWER: {result['answer']}")
                logger.info(f"Retrieved {result['retrieved_count']} documents")
                
                # Log source metadata in a more structured way
                for i, meta in enumerate(result["document_metadata"]):
                    source_info = f"Source {i+1}"
                    if "type" in meta:
                        source_info += f" - Type: {meta['type']}"
                    if "plot_path" in meta:
                        source_info += f" - Plot: {meta['plot_path']}"
                    logger.info(source_info)
                
                # Add answer to messages
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "metadata": result["document_metadata"]
                })
                
                # Keep only the most recent messages (10 messages)
                if len(st.session_state.messages) > 10:
                    logger.info("Trimming conversation history to last 10 messages")
                    st.session_state.messages = st.session_state.messages[-10:]
                
                processing_time = time.time() - start_time
                logger.info(f"Response generated in {processing_time:.2f} seconds")
                st.success(f"Response generated in {processing_time:.2f} seconds")
                
            except Exception as e:
                error_msg = f"Error generating response: {str(e)}"
                logger.error(error_msg, exc_info=True)
                st.error(error_msg)
        else:
            error_msg = "Knowledge base not initialized. Please try again."
            logger.error(error_msg)
            st.error(error_msg)

# Display the latest interaction in the latest_interaction_container
with latest_interaction_container:
    if len(st.session_state.messages) >= 2:
        st.write("---")
        st.header("Latest Interaction")
        # Get the last question and answer
        last_question = next((m for m in reversed(st.session_state.messages) if m["role"] == "user"), None)
        last_answer = next((m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None)
        
        if last_question and last_answer:
            # Display the last question-answer pair
            display_message(last_question)
            display_message(last_answer)

# Display previous conversation history
with history_container:
    if len(st.session_state.messages) > 2:
        st.write("---")
        st.header("Previous Conversations")
        
        # Display all messages except the last Q&A pair
        for i in range(0, len(st.session_state.messages) - 2, 2):
            if i+1 < len(st.session_state.messages):
                st.markdown("---")
                display_message(st.session_state.messages[i])
                display_message(st.session_state.messages[i+1])

# Footer
st.write("---")
st.caption("💡 **Tip:** For the best results, ask specific questions about sales data trends, regional comparisons, or customer demographics.")