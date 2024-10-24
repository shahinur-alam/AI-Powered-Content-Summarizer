# AI-Powered Content Summarizer

This project features a Flask-based web application that integrates various NLP models to generate text summaries. Users can input text and receive summaries in formats like paragraphs or bullet points, utilizing models such as BART, Llama 2, and Ollama Llama 3, with support for fine-tuning using LoRA.

## Features
- Summarization using BART, Llama 2, and Ollama Llama 3.
- Choose between paragraph or bullet points format.
- Adjustable summary lengths.
- Efficient fine-tuning with LoRA.

## Installation

### Prerequisites

- Python 3.7+
- pip


### Clone and Install

```bash
git clone https://github.com/shahinur-alam/AI-Powered-Content-Summarizer.git
cd AI-Powered-Content-Summarizer
pip install -r requirements.txt
```


### Run the Application
```bash
python content_summarizer.py
```

The app will run at http://127.0.0.1:5000.

## Usage
- Go to http://127.0.0.1:5000.
- Input your text.
- Select summary type (short/long) and format (paragraph/bullet points).
- Submit to get the summary.

## File Structure
```bash
.
├── content_summarizer.py          # Main app
├── content_summarizer_Llama3.py
├── content_summarizer_Llama27_finetune.py  
└── templates/
    └── index.html  # Web interface
```

## Customization
- GET /: Render input form.
- POST /: Summarize input text.

## Customization
Modify the summarize_text() function in app.py to adjust summary length or format.


