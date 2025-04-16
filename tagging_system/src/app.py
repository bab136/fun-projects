import streamlit as st
import pandas as pd
from typing import List, Dict
import os
import sys
import tempfile

# Add the src directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_processor import DataProcessor
from src.llm_processor import LLMProcessor
from src.feedback_storage import FeedbackStorage

def initialize_session_state():
    """Initialize session state variables."""
    if 'current_index' not in st.session_state:
        st.session_state.current_index = 0
    if 'feedback_storage' not in st.session_state:
        st.session_state.feedback_storage = FeedbackStorage()
    if 'data_processor' not in st.session_state:
        st.session_state.data_processor = None
    if 'llm_processor' not in st.session_state:
        st.session_state.llm_processor = LLMProcessor()
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None

def load_data(uploaded_file, text_column: str):
    """Load and process the data."""
    try:
        # Save uploaded file to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            st.session_state.uploaded_file_path = tmp_file.name

        # Initialize data processor with the temporary file
        st.session_state.data_processor = DataProcessor(st.session_state.uploaded_file_path, text_column)
        st.session_state.data_processor.load_data()
        st.session_state.llm_processor.load_model()
        return True
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return False

def process_text(text: str) -> Dict[str, str]:
    """Process text using the LLM."""
    return st.session_state.llm_processor.extract_entities(text)

def submit_feedback(rating: str):
    """Submit feedback for the current text."""
    current_text = st.session_state.data_processor.df.iloc[st.session_state.current_index][st.session_state.data_processor.text_column]
    tags = process_text(current_text)
    
    st.session_state.feedback_storage.add_feedback(
        text=current_text,
        component=tags['component'],
        condition=tags['condition'],
        rating=rating
    )
    
    # Move to next text
    st.session_state.current_index += 1
    if st.session_state.current_index >= len(st.session_state.data_processor.df):
        st.session_state.current_index = 0
        st.success("You've reviewed all texts! Starting over...")

def show_feedback_stats():
    """Display feedback statistics."""
    stats = st.session_state.feedback_storage.get_feedback_stats()
    
    st.subheader("Feedback Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Reviews", stats['total'])
    with col2:
        st.metric("Good Ratings", stats['ratings']['Good'])
    with col3:
        st.metric("Bad Ratings", stats['ratings']['Bad'])
    
    # Show percentages
    st.write("Rating Distribution:")
    for rating, percentage in stats['percentages'].items():
        st.progress(percentage / 100, text=f"{rating}: {percentage:.1f}%")

def main():
    st.title("Symptom Tagging System")
    
    initialize_session_state()
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file:
        text_column = st.text_input("Enter the name of the text column")
        if text_column:
            if load_data(uploaded_file, text_column):
                st.success("Data loaded successfully!")
    
    if st.session_state.data_processor is not None and st.session_state.data_processor.df is not None:
        # Show current text and tags
        current_text = st.session_state.data_processor.df.iloc[st.session_state.current_index][st.session_state.data_processor.text_column]
        tags = process_text(current_text)
        
        st.subheader("Current Text")
        st.write(current_text)
        
        st.subheader("Extracted Tags")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Component:", tags['component'])
        with col2:
            st.write("Condition:", tags['condition'])
        
        # Feedback buttons
        st.subheader("Rate the Tags")
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("Bad"):
                submit_feedback("Bad")
        with col2:
            if st.button("OK"):
                submit_feedback("OK")
        with col3:
            if st.button("Good"):
                submit_feedback("Good")
        
        # Progress
        st.progress(st.session_state.current_index / len(st.session_state.data_processor.df))
        st.write(f"Progress: {st.session_state.current_index + 1}/{len(st.session_state.data_processor.df)}")
        
        # Show statistics
        show_feedback_stats()
        
        # Export feedback
        if st.button("Export Feedback"):
            feedback_data = st.session_state.feedback_storage.get_feedback()
            df = pd.DataFrame(feedback_data)
            csv = df.to_csv(index=False)
            st.download_button(
                label="Download Feedback CSV",
                data=csv,
                file_name="feedback.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main() 