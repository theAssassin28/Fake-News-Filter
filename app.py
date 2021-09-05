from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
vectorizer = pickle.load(open('vectorizer.pkl','rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods = ['POST'])
def predict():
    features = request.form.values()
    vectorized_features = vectorizer.transform(features)
    if model.predict(vectorized_features) == 1:
        prediction = 'True News'
    else:
        prediction = 'Fake News'

    return render_template('index.html', prediction_text = prediction)

if __name__ == "__main__":
    app.run(debug = True)
