from flask import Flask, request, jsonify
from transformers import T5Tokenizer, T5ForConditionalGeneration
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello, World! main.py'

@app.route('/ap', methods=['POST'])
def summarize():
    try:
        data = request.get_json()
        long_text = data.get('long_text', '')
        summary = summarize_text(long_text)
        return jsonify({'summary': summary})
    except Exception as e:
        return jsonify({'error': 'Error processing the request'})

def summarize_text(long_text, max_words=75):
    model_name = 't5-base'
    tokenizer = T5Tokenizer.from_pretrained(model_name, model_max_length=1024)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Tokenize the long text
    inputs = tokenizer.encode("summarize: " + long_text, truncation=True, return_tensors='pt')

    # Generate the summary
    summary_ids = model.generate(inputs, num_beams=4, max_length=75, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Check if the generated summary has fewer words than the maximum allowed
    # summary_word_count = len(summary.split())
    # if summary_word_count <= max_words:
    return summary
    # else:
    #     # If the summary exceeds the maximum word limit, return the original text
    #     return long_text

if __name__ == '__main__':
    app.run()
