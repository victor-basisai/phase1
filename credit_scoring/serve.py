"""
Script for serving.
"""
import pickle
import numpy as np
from flask import Flask, request

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

OUTPUT_MODEL_PATH = "/artefact/model.pkl"


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
    
    # Score
    score_prob = (
        model
        .predict_proba(np.array(row_feats).reshape(1, -1))[:, 1]
        .item()
    )

    return score_prob


# pylint: disable=invalid-name
app = Flask(__name__)


@app.route("/", methods=["POST"])
def get_score():
    """Returns the `score_prob` given the some features"""

    features = request.json
    result = {
        "score_prob": predict_prob(features)
    }
    return result


def main():
    """Starts the Http server"""
    app.run()


if __name__ == "__main__":
    main()