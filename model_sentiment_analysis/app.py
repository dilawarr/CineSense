from flask import Flask, render_template, request
import joblib
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

app = Flask(__name__)
model = joblib.load('models/sentiment_model.sav')  # Load your sentiment analysis model
port_stem = PorterStemmer()
vectorizer = joblib.load('models/vectorizer.pkl')

def stemming(content):
    stemmed_content = re.sub('[^a-zA-Z]', ' ', content)
    stemmed_content = stemmed_content.lower()
    stemmed_content = stemmed_content.split()
    stemmed_content = [port_stem.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
    stemmed_content = ' '.join(stemmed_content)

    return stemmed_content 

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])

def predict():
    review = request.form['review']

    # pangey
    review = stemming(review)
    review = [review]
    review = vectorizer.transform(review)
    # 

    # sentiment = model.predict([review])[0]
    sentiment = model.predict(review)
    return render_template('result.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
