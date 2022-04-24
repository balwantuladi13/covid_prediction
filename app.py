from flask import Flask
from model import grouped_country
from svm import get_predictions

app = Flask(__name__)

@app.route('/get_grouped_country')
def get_grouped_country():
    return grouped_country()

@app.route('/get_predictions')
def get_covid_predictions():
    return get_predictions()

@app.route('/')
def hello():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run()