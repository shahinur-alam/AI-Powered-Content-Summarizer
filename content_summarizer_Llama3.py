import os
from flask import Flask, render_template, request
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

app = Flask(__name__)

# Initialize Ollama with Llama 3 model
llm = Ollama(model="llama3")

# Define a prompt template for summarization
summarize_template = """Summarize the following text in {format_type} format:

{text}

Summary:"""

prompt = PromptTemplate(
    input_variables=["text", "format_type"],
    template=summarize_template
)

# Create an LLMChain for summarization
summarize_chain = LLMChain(llm=llm, prompt=prompt)


def summarize_text(text, max_length=150, min_length=50, format_type="paragraph"):
    # Adjust format type for the prompt
    format_type = "bullet points" if format_type == "bullet_points" else "paragraph"

    # Generate summary
    summary = summarize_chain.run(text=text, format_type=format_type)

    # Truncate summary if it exceeds max_length
    if len(summary) > max_length:
        summary = summary[:max_length].rsplit('.', 1)[0] + '.'

    return summary


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        text = request.form['text']
        summary_type = request.form['summary_type']
        format_type = request.form['format_type']

        max_length = 150 if summary_type == 'short' else 300
        min_length = 50 if summary_type == 'short' else 100

        summary = summarize_text(text, max_length, min_length, format_type)
        return render_template('index.html', summary=summary)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)