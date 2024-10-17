import os
from flask import Flask, render_template, request
from transformers import BartForConditionalGeneration, BartTokenizer

app = Flask(__name__)

# Load the BART model and tokenizer
model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


def summarize_text(text, max_length=150, min_length=50, format_type="paragraph"):
    # Encode the text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=1024, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=max_length, min_length=min_length, length_penalty=2.0, num_beams=4,
                                 early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    if format_type == "bullet_points":
        # Split the summary into sentences and format as bullet points
        sentences = summary.split('. ')
        summary = "\n".join([f"• {sentence.strip()}" for sentence in sentences if sentence])

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