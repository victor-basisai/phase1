# Core Packages
import os
import json

# Third Party
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score
from sklearn.calibration import calibration_curve
from sklearn import metrics
import utils.credit as utils

# Bedrock
from bedrock_client.bedrock.analyzer.model_analyzer import ModelAnalyzer
from bedrock_client.bedrock.analyzer import ModelTypes
from bedrock_client.bedrock.api import BedrockApi
from bedrock_client.bedrock.metrics.service import ModelMonitoringService
import pickle
import logging

OUTPUT_MODEL_PATH = "/artefact/model.pkl"
FEATURE_COLS_PATH = "/artefact/feature_cols.pkl"

CONFIG_FAI = {
    'SEX': {
        'privileged_attribute_values': [1],
        'privileged_group_name': 'Male',  # privileged group name corresponding to values=[1]
        'unprivileged_attribute_values': [2],
        'unprivileged_group_name': 'Female',  # unprivileged group name corresponding to values=[0]
    }
}

def compute_log_metrics(model, x_train, x_val, y_val, model_name="tree_model", model_type=ModelTypes.TREE):
    """Compute and log metrics."""
    y_prob = model.predict_proba(x_val)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    acc = metrics.accuracy_score(y_val, y_pred)
    precision = metrics.precision_score(y_val, y_pred)
    recall = metrics.recall_score(y_val, y_pred)
    f1_score = metrics.f1_score(y_val, y_pred)
    roc_auc = metrics.roc_auc_score(y_val, y_prob)
    avg_prc = metrics.average_precision_score(y_val, y_prob)
    print("Evaluation\n"
          f"  Accuracy          = {acc:.4f}\n"
          f"  Precision         = {precision:.4f}\n"
          f"  Recall            = {recall:.4f}\n"
          f"  F1 score          = {f1_score:.4f}\n"
          f"  ROC AUC           = {roc_auc:.4f}\n"
          f"  Average precision = {avg_prc:.4f}")

    # Log metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("ROC AUC", roc_auc)
    bedrock.log_metric("Avg precision", avg_prc)
    bedrock.log_chart_data(y_val.astype(int).tolist(),
                           y_prob.flatten().tolist())

    # Calculate and upload xafai metrics
    analyzer = ModelAnalyzer(model, model_name=model_name, model_type=model_type).train_features(x_train).test_features(x_val)
    analyzer.fairness_config(CONFIG_FAI).test_labels(y_val).test_inference(y_pred)
    return analyzer.analyze()

def main():
    # Extraneous columns (as might be determined through feature selection)
    drop_cols = ['ID']

    # Load into Dataframes
    X_train, y_train = utils.load_dataset(os.path.join('data', 'creditdata', 'creditdata_train_v2.csv'), drop_columns=drop_cols)
    X_test, y_test = utils.load_dataset(os.path.join('data', 'creditdata', 'creditdata_test_v2.csv'), drop_columns=drop_cols)

    # Use best parameters from a model selection and threshold tuning process
    best_regularizer = 1e-1
    best_th = 0.43
    model = utils.train_log_reg_model(X_train, y_train, seed=0, C=best_regularizer, upsample=True, verbose=True)

    # If model is in an sklearn pipeline, extract it
    (
        shap_values, 
        base_shap_values, 
        global_explainability, 
        fairness_metrics,
    ) = compute_log_metrics(model[1], X_train, X_test, y_test, model_name="logreg_model", model_type=ModelTypes.LINEAR)

    # IMPORTANT TO SHOW ON UI
    print("Saving model!")

    with open(OUTPUT_MODEL_PATH, "wb") as model_file:
        pickle.dump(model[1], model_file)

    # Save feature names
    # with open(FEATURE_COLS_PATH, "wb") as file:
    #     pickle.dump(feature_cols, file)

    print("Done!")

if __name__ == "__main__":
    try:
        print("Hello world")
        main()
    except Exception as e:
        print(e)
        print("Hmm something went wrong...")
        print("What?!")