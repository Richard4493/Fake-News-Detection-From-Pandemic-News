import numpy as np
from flask import Flask, request, jsonify, render_template
from FakeNewsDetection import fake_news_detection
import pickle

model = fake_news_detection("abc")


app = Flask(__name__, template_folder='templates')



@app.route('/')
def man():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():

    data1 = request.form['a']
    data2 = request.form['b']

    pred = model.predict(str(data1),str(data2))
    return render_template('after.html', data=pred)

if __name__ == "__main__":
    app.run(debug=True)