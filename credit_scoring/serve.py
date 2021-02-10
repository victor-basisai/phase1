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

def predict_prob(features,
                 model=pickle.load(open(OUTPUT_MODEL_PATH, "rb"))):
    """Predict credit risk score given features.
    Args:
        features (dict)
        model
    Returns:
        score_prob (float): credit risk probability
    """
    row_feats = list()
    for col in FEATURES:
        row_feats.append(features[col])
    
    if row_feats is not None:
        # Score
        score_prob = (
            model
            .predict_proba(np.array(row_feats).reshape(1, -1))[:, 1]
            .item()
        )

        # Log the prediction
        current_app.monitor.log_prediction(
            request_body=json.dumps(features),
            features=row_feats,
            output=score_prob,
        )

        return score_prob
    return np.NaN


# pylint: disable=invalid-name
app = Flask(__name__)

@app.before_first_request
def init_background_threads():
    """Global objects with daemon threads will be stopped by gunicorn --preload flag.
    So instantiate them here instead.
    """
    current_app.monitor = ModelMonitoringService()

@app.route("/metrics", methods=["POST"])
def get_metrics():
    """Returns real time feature values recorded by prometheus
    """
    body, content_type = current_app.monitor.export_http(
        params=request.args.to_dict(flat=False),
        headers=request.headers,
    )
    return Response(body, content_type=content_type)


@app.route("/", methods=["GET"])
def get_score():
    """Returns the help page"""
    return "<h1 style='color:blue'>Hello Model Server!</h1>"


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()