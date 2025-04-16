import pandas as pd
from typing import List, Optional
import logging
import chardet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    def __init__(self, csv_path: str, text_column: str):
        """
        Initialize the data processor.
        
        Args:
            csv_path (str): Path to the CSV file
            text_column (str): Name of the column containing free text data
        """
        self.csv_path = csv_path
        self.text_column = text_column
        self.df = None
        
    def detect_encoding(self) -> str:
        """Detect the encoding of the CSV file."""
        with open(self.csv_path, 'rb') as file:
            raw_data = file.read()
            result = chardet.detect(raw_data)
            encoding = result['encoding']
            logger.info(f"Detected encoding: {encoding}")
            return encoding if encoding else 'utf-8'
        
    def load_data(self) -> pd.DataFrame:
        """Load and validate the CSV data."""
        try:
            # Detect file encoding
            encoding = self.detect_encoding()
            
            # Try to read the CSV with the detected encoding
            try:
                self.df = pd.read_csv(self.csv_path, encoding=encoding)
            except UnicodeDecodeError:
                # If the detected encoding fails, try common encodings
                for enc in ['latin1', 'iso-8859-1', 'cp1252']:
                    try:
                        self.df = pd.read_csv(self.csv_path, encoding=enc)
                        logger.info(f"Successfully read file with {enc} encoding")
                        break
                    except UnicodeDecodeError:
                        continue
                else:
                    raise ValueError(f"Could not read file with any supported encoding")
            
            if self.text_column not in self.df.columns:
                raise ValueError(f"Text column '{self.text_column}' not found in CSV")
            return self.df
        except Exception as e:
            logger.error(f"Error loading CSV file: {str(e)}")
            raise
            
    def get_text_samples(self, n_samples: Optional[int] = None) -> List[dict]:
        """
        Get text samples for processing.
        
        Args:
            n_samples (int, optional): Number of samples to return. If None, returns all.
            
        Returns:
            List[dict]: List of dictionaries containing text and metadata
        """
        if self.df is None:
            self.load_data()
            
        samples = self.df[[self.text_column]].copy()
        if n_samples:
            samples = samples.sample(n=min(n_samples, len(samples)))
            
        return samples.to_dict('records')
    
    def add_tags(self, tags: List[dict]) -> None:
        """
        Add extracted tags to the dataframe.
        
        Args:
            tags (List[dict]): List of dictionaries containing extracted tags
        """
        if self.df is None:
            self.load_data()
            
        # Add new columns for tags
        self.df['symptom_component'] = None
        self.df['symptom_condition'] = None
        
        # Update with extracted tags
        for idx, tag in enumerate(tags):
            if idx < len(self.df):
                self.df.at[idx, 'symptom_component'] = tag.get('component')
                self.df.at[idx, 'symptom_condition'] = tag.get('condition')
                
    def save_results(self, output_path: str) -> None:
        """
        Save the processed data with tags to a new CSV file.
        
        Args:
            output_path (str): Path to save the output CSV
        """
        if self.df is None:
            raise ValueError("No data to save. Process data first.")
            
        try:
            self.df.to_csv(output_path, index=False)
            logger.info(f"Results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise 