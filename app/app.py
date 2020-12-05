from functools import wraps

import jsonschema
import werkzeug
from flask import Flask, jsonify, request
from flask_pymongo import PyMongo

from app.model import AnomalyDetector, Classifier
from toolbox.utils import dict_to_time_series

app = Flask(__name__)
app.config["MONGO_URI"] = "mongodb://localhost:27017/kistler_kv"
mongo = PyMongo(app)

clf = Classifier()
anodet = AnomalyDetector()

schema = schema = {
    "type": "object",
    "properties": {
        "pen_id": {
            "type": "string",
        },
        "time": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1

        },
        "position": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1
        },
        "froce": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 1

        }},
    "required": ["pen_id", "force", "position"]

}


def validate_json(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            jsonschema.validate(request.json, schema=schema)
        except jsonschema.exceptions.ValidationError as e:
            return jsonify({'error': e.message})
        return f(*args, **kwargs)
    return wrapper


@app.route('/predict/', methods=['POST'])
@validate_json
def predict():
    data = request.get_json()
    ts_data = dict_to_time_series(data)
    predictions = {
        'classifier': clf.predict(ts_data),
        'anomaly_detector': anodet.predict(ts_data)}
    mongo.db.posts.insert_one({
        'data': data,
        'predictions': predictions})

    return predictions


@app.errorhandler(werkzeug.exceptions.BadRequest)
def handle_bad_request(e):
    return jsonify({"error": "bad request!"}), 400
