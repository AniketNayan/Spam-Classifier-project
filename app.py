from flask import Flask, render_template, request
import pickle
import string
import os
import logging
import nltk

# Ensure the app uses a local nltk_data if present
nltk.data.path.append(os.path.abspath("./nltk_data"))

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ps = PorterStemmer()

def transform_text(text):
    """
    Transform input text: lowercase, tokenize, remove non-alnum, remove stopwords, stem.
    Uses a safe fallback if NLTK tokenizers are missing.
    """
    text = text.lower()
    try:
        tokens = nltk.word_tokenize(text)
    except LookupError:
        # Fallback: very simple whitespace split if punkt/punkt_tab missing
        logger.warning("NLTK tokenizer missing; falling back to simple split.")
        tokens = text.split()

    # keep only alphanumeric tokens
    tokens = [t for t in tokens if t.isalnum()]
    # filter stopwords (safe guard: ensure stopwords corpus exists)
    try:
        stop_words = set(stopwords.words('english'))
    except LookupError:
        logger.warning("NLTK stopwords missing; treating as empty set.")
        stop_words = set()

    filtered = [ps.stem(t) for t in tokens if t not in stop_words and t not in string.punctuation]
    return " ".join(filtered)

# Load CountVectorizer & model
MODEL_PATH = 'model.pkl'
VECT_PATH = 'vectorizer.pkl'

cv = None
model = None

def load_models():
    global cv, model
    try:
        with open(VECT_PATH, 'rb') as f:
            cv = pickle.load(f)
        logger.info("Loaded vectorizer.")
    except Exception as e:
        logger.exception(f"Failed to load vectorizer from {VECT_PATH}: {e}")
        cv = None

    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        logger.info("Loaded model.")
    except Exception as e:
        logger.exception(f"Failed to load model from {MODEL_PATH}: {e}")
        model = None

load_models()

app = Flask(__name__)

@app.route('/health')
def health():
    return 'ok', 200

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    message = ""

    if request.method == 'POST':
        message = request.form.get('message', '')  # get user message
        if message.strip():
            try:
                transformed = transform_text(message)
                if cv is None or model is None:
                    logger.error("Model or vectorizer not loaded.")
                    prediction = "Error: model not available"
                else:
                    vector = cv.transform([transformed])
                    result = model.predict(vector)[0]
                    prediction = "Spam" if int(result) == 1 else "Not Spam"
            except Exception as e:
                logger.exception("Error during prediction:")
                prediction = "Error processing message"

    return render_template('index.html', prediction=prediction, message=message)

if __name__ == '__main__':
    # Local dev only. For production, use gunicorn (start.sh)
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
