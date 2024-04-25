from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load('models/sentiment_model.sav')  # Load your sentiment analysis model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    review = request.form['review']
    sentiment = model.predict([review])[0]
    return render_template('result.html', sentiment=sentiment)

if __name__ == '__main__':
    app.run(debug=True)
