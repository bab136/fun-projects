# Symptom Tagging System

A system for extracting symptom components and conditions from free text data using LLMs, with a feedback loop for continuous improvement through supervised fine-tuning.

## Features

- CSV file processing with pandas
- Entity extraction using LLMs
- Interactive GUI for feedback collection
- Feedback storage and statistics
- Export functionality for collected feedback

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tagging_system
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit application:
```bash
streamlit run src/app.py
```

2. Upload your CSV file containing the text data
3. Enter the name of the column containing the text to analyze
4. Review the extracted tags and provide feedback using the rating buttons
5. Export the collected feedback when ready

## Project Structure

```
tagging_system/
├── src/
│   ├── app.py              # Streamlit GUI application
│   ├── data_processor.py   # CSV processing module
│   ├── llm_processor.py    # LLM integration module
│   └── feedback_storage.py # Feedback storage module
├── data/                   # Directory for input data
├── models/                 # Directory for saved models
├── feedback/              # Directory for feedback database
└── requirements.txt       # Project dependencies
```

## Feedback Collection

The system collects feedback in three categories:
- Bad: Tags are incorrect or missing
- OK: Tags are partially correct
- Good: Tags are correct and complete

## Model Fine-tuning

The collected feedback can be used to fine-tune the LLM model for better performance. The feedback data is stored in a SQLite database and can be exported for further analysis or model training.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 