import streamlit as st
import os
from PIL import Image
from typing import List


def display_plot(plot_path: str):
    """
    Display a plot image in the Streamlit app.
    Args:
        plot_path (str): Path to the plot image.
    """
    
    if plot_path and os.path.exists(plot_path):
        try:
            image = Image.open(plot_path)
            st.image(image, use_container_width=True)
        except Exception as e:
            st.error(f"Error displaying image: {str(e)}")
    else:
        st.warning("Plot image not found")


def extract_display_metadata(metadata: List) -> dict:
    """
    Extract and format metadata for display.
    Args:
        metadata (List): Metadata to extract.
    Returns:
        dict: Formatted metadata for display.
    """

    display_data = {}
    
    # Check for important fields
    if "type" in metadata:
        display_data["Type"] = metadata["type"].capitalize()
    
    # Extract time period, product, region, or age group
    for key in ["period", "product", "region", "age_group"]:
        if key in metadata:
            display_name = key.replace("_", " ").title()
            display_data[display_name] = metadata[key]
    
    # Add statistical information if available in raw_data
    if "raw_data" in metadata and isinstance(metadata["raw_data"], dict):
        raw_data = metadata["raw_data"]
        
        if "total_sales" in raw_data:
            display_data["Total Sales"] = f"${raw_data['total_sales']:,.2f}"
            
        if "average_sale" in raw_data:
            display_data["Average Sale"] = f"${raw_data['average_sale']:.2f}"
            
        if "average_satisfaction" in raw_data:
            display_data["Avg. Satisfaction"] = f"{raw_data['average_satisfaction']:.2f}/5"
            
        if "total_records" in raw_data:
            display_data["Total Records"] = raw_data["total_records"]
    
    return display_data


def clear_chat():
    """
    Clear the chat history and reset the state.
    """

    st.session_state.messages = []
    st.session_state.last_query = ""
    st.session_state.should_process_query = False


def format_chat_history(messages: List, max_messages=10) -> List:
    """
    Format the chat history for the LLM.
    Args:
        messages (List): List of messages in the chat history.
        max_messages (int): Maximum number of messages to include.
    Returns:
        List: Formatted chat history for the LLM."""

    # Limit to last max_messages
    recent_messages = messages[-max_messages:] if len(messages) > max_messages else messages
    
    # Format messages for the LLM
    formatted_history = []
    for msg in recent_messages:
        role = "user" if msg["role"] == "user" else "assistant"
        formatted_history.append({"role": role, "content": msg["content"]})
    
    return formatted_history


def display_message(message: dict):
    """
    Display a message in the Streamlit app.
    Args:
        message (dict): Message to display.
    """

    if message["role"] == "user":
        st.subheader("🙋 Question:")
        st.write(message["content"])
    else:
        st.subheader("🤖 Answer:")
        st.info(message["content"])
        
        # Display sources and plots if available
        if "metadata" in message and message["metadata"]:
            st.subheader("📚 Sources:")
            
            # Create three columns for the plots
            cols = st.columns(min(3, len(message["metadata"])))
            
            # Display each source with its plot
            for i, metadata in enumerate(message["metadata"]):
                with cols[i % len(cols)]:
                    # Format and display metadata
                    display_data = extract_display_metadata(metadata)
                    
                    # Display type and subject in a card
                    with st.expander("📄 Source Details", expanded=True):
                        # Display type and subject
                        if "Type" in display_data:
                            if display_data["Type"] == "Time_period" and "Period" in display_data:
                                st.markdown(f"**{display_data['Type']}: {display_data['Period']}**")
                            elif display_data["Type"] == "Product" and "Product" in display_data:
                                st.markdown(f"**{display_data['Product']}**")
                            elif display_data["Type"] == "Region" and "Region" in display_data:
                                st.markdown(f"**{display_data['Region']} Region**")
                            elif display_data["Type"] == "Demographic" and "Age Group" in display_data:
                                st.markdown(f"**Age Group: {display_data['Age Group']}**")
                            elif display_data["Type"] == "Overall":
                                st.markdown("**Overall Summary**")
                            else:
                                st.markdown(f"**{display_data.get('Type', 'Source')}**")
                        
                        # Display stats
                        for key in ["Total Sales", "Average Sale", "Avg. Satisfaction", "Total Records"]:
                            if key in display_data:
                                st.text(f"{key}: {display_data[key]}")
                    
                    # Display the plot if available
                    if "plot_path" in metadata:
                        display_plot(metadata["plot_path"])
