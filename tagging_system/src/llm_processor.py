from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import List, Dict, Optional
import logging
import json
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMProcessor:
    def __init__(self, model_name: str = "meta-llama/Llama-2-7b-chat-hf"):
        """
        Initialize the LLM processor.
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not self.hf_token:
            raise ValueError("HUGGINGFACE_TOKEN not found in environment variables")
        
    def load_model(self):
        """Load the model and tokenizer."""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token,
                torch_dtype=torch.float16,  # Use half precision for efficiency
                device_map="auto"  # Automatically handle model placement
            )
            logger.info(f"Model loaded successfully on {self.device}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
            
    def extract_entities(self, text: str) -> Dict[str, str]:
        """
        Extract symptom component and condition from text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict[str, str]: Dictionary containing extracted entities
        """
        if self.model is None:
            self.load_model()
            
        # Prepare the prompt in Llama 2 chat format
        prompt = f"""<s>[INST] Extract the symptom component (what's causing the problem) and symptom condition (the problem being caused) from the following text. Format the response as JSON with keys 'component' and 'condition'.

Text: {text} [/INST]"""
        
        # Tokenize and generate
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        try:
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                top_p=0.9,
                repetition_penalty=1.1
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse the response
            try:
                # Extract JSON from response
                json_str = response.split('{')[1].split('}')[0]
                json_str = '{' + json_str + '}'
                result = json.loads(json_str)
                
                return {
                    'component': result.get('component', ''),
                    'condition': result.get('condition', '')
                }
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse JSON from response: {response}")
                return {'component': '', 'condition': ''}
                
        except Exception as e:
            logger.error(f"Error during entity extraction: {str(e)}")
            return {'component': '', 'condition': ''}
            
    def process_batch(self, texts: List[str]) -> List[Dict[str, str]]:
        """
        Process a batch of texts.
        
        Args:
            texts (List[str]): List of texts to process
            
        Returns:
            List[Dict[str, str]]: List of extracted entities
        """
        results = []
        for text in texts:
            entities = self.extract_entities(text)
            results.append(entities)
        return results
        
    def save_model(self, output_dir: str):
        """
        Save the fine-tuned model.
        
        Args:
            output_dir (str): Directory to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        try:
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            logger.info(f"Model saved to {output_dir}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise 