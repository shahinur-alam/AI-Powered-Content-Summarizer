# AI Powered Content Summarization App using BART

This Flask app uses the BART model to summarize text, with options for short or long summaries in paragraph or bullet-point format.

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
├── app.py          # Main app
└── templates/
    └── index.html  # Web interface
```

## Customization
- GET /: Render input form.
- POST /: Summarize input text.

## Customization
Modify the summarize_text() function in app.py to adjust summary length or format.


