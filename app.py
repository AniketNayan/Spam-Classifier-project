from flask import Flask, render_template, request
import pickle
import string
import nltk
nltk.data.path.append("./nltk_data") 
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Ensure nltk resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir="./nltk_data")
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', download_dir="./nltk_data")

ps = PorterStemmer()

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = [i for i in text if i.isalnum()]
    y = [ps.stem(i) for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    return " ".join(y)

# Load CountVectorizer & model
cv = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    message = ""  # default empty

    if request.method == 'POST':
        message = request.form.get('message', '')  # get user message
        if message.strip():  # only process if not empty
            transformed = transform_text(message)
            vector = cv.transform([transformed])
            result = model.predict(vector)[0]
            prediction = "Spam" if result == 1 else "Not Spam"

    # send both prediction and message to HTML
    return render_template('index.html', prediction=prediction, message=message)

if __name__ == '__main__':
    app.run(debug=True)
