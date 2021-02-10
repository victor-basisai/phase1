"""
Script for serving.
"""
import pickle
import numpy as np
import json
from flask import Flask, Response, current_app, request
from bedrock_client.bedrock.metrics.service import ModelMonitoringService

OUTPUT_MODEL_PATH = "/artefact/model.pkl"

FEATURES = [ 
    'LIMIT_BAL',
    'SEX',
    'EDUCATION',
    'MARRIAGE',
    'AGE',
    'PAY_1',
    'PAY_2',
    'PAY_3',
    'PAY_4',
    'PAY_5',
    'PAY_6',
    'BILL_AMT1',
    'BILL_AMT2',
    'BILL_AMT3',
    'BILL_AMT4',
    'BILL_AMT5',
    'BILL_AMT6',
    'PAY_AMT1',
    'PAY_AMT2',
    'PAY_AMT3',
    'PAY_AMT4',
    'PAY_AMT5',
    'PAY_AMT6'
]

def predict_prob(request_json,
                 model=pickle.load(open(OUTPUT_MODEL_PATH, "rb"))):
    """Predict credit risk score given features.
    Args:
        request_json (dict)
        model
    Returns:
        score_prob (float): credit risk probability
    """
    # Parse request_json
    features = list()
    for col in FEATURES:
        features.append(request_json[col])
    
    if features is not None:
        # Score
        score_prob = (
            model
            .predict_proba(np.array(features).reshape(1, -1))[:, 1]
            .item()
        )

        return features, score_prob
    return np.NaN


# pylint: disable=invalid-name
app = Flask(__name__)

@app.before_first_request
def init_background_threads():
    """Global objects with daemon threads will be stopped by gunicorn --preload flag.
    So instantiate them here instead.
    """
    current_app.monitor = ModelMonitoringService()


@app.route("/metrics", methods=["GET"])
def get_metrics():
    """Returns real time feature values recorded by prometheus
    """
    body, content_type = current_app.monitor.export_http(
        params=request.args.to_dict(flat=False),
        headers=request.headers,
    )
    return Response(body, content_type=content_type)


@app.route("/infer", methods=["POST"])
def get_inference():
    """Returns the model inference score given some features in JSON
    """    
    # Perform inference
    features, score_prob = predict_prob(request.json)
    
    # Log the prediction
    current_app.monitor.log_prediction(
        request_body=request.json,
        features=features,
        output=score_prob,
    )

    # Return the result
    result = {
        "inference": score_prob
    }
    return result


@app.route("/", methods=["GET"])
def get_score():
    """Returns the help page"""
    return "<h1 style='color:blue'>Hello Model Server!</h1>"


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()