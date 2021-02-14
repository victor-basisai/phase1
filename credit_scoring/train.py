"""
Python script for training a model version
"""
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

# ---------------------------------
# Constants
# ---------------------------------

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

# ---------------------------------
# Bedrock functions
# ---------------------------------

def compute_log_metrics(model, x_train, 
                        x_test, y_test, 
                        best_th=0.5,
                        model_name="tree_model", 
                        model_type=ModelTypes.TREE):
    """Compute and log metrics."""
    test_prob = model.predict_proba(x_test)[:, 1]
    test_pred = np.where(test_prob > best_th, 1, 0)

    acc = metrics.accuracy_score(y_test, test_pred)
    precision = metrics.precision_score(y_test, test_pred)
    recall = metrics.recall_score(y_test, test_pred)
    f1_score = metrics.f1_score(y_test, test_pred)
    roc_auc = metrics.roc_auc_score(y_test, test_prob)
    avg_prc = metrics.average_precision_score(y_test, test_prob)
    print("Evaluation\n"
          f"  Accuracy          = {acc:.4f}\n"
          f"  Precision         = {precision:.4f}\n"
          f"  Recall            = {recall:.4f}\n"
          f"  F1 score          = {f1_score:.4f}\n"
          f"  ROC AUC           = {roc_auc:.4f}\n"
          f"  Average precision = {avg_prc:.4f}")

    # Bedrock Logger: captures model metrics
    bedrock = BedrockApi(logging.getLogger(__name__))
    bedrock.log_metric("Accuracy", acc)
    bedrock.log_metric("ROC AUC", roc_auc)
    bedrock.log_metric("Avg precision", avg_prc)
    bedrock.log_chart_data(y_test.astype(int).tolist(),
                           test_prob.flatten().tolist())

    # Bedrock Model Analyzer: generates model explainability and fairness metrics
    # Requires model object from pipeline to be passed in
    analyzer = ModelAnalyzer(model[1], model_name=model_name, model_type=model_type)\
                    .train_features(x_train)\
                    .test_features(x_test)
    
    analyzer.fairness_config(CONFIG_FAI)\
        .test_labels(y_test)\
        .test_inference(test_pred)
    
    return analyzer.analyze()

def main():
    # Extraneous columns (as might be determined through feature selection)
    drop_cols = ['ID']

    # Load into Dataframes
    # x_<name> : features
    # y_<name> : labels
    x_train, y_train = utils.load_dataset(os.path.join('data', 'creditdata', 'creditdata_train_v2.csv'), drop_columns=drop_cols)
    x_test, y_test = utils.load_dataset(os.path.join('data', 'creditdata', 'creditdata_test_v2.csv'), drop_columns=drop_cols)

    # # MODEL 1: LOGISTIC REGRESSION
    # # Use best parameters from a model selection and threshold tuning process
    # best_regularizer = 1e-1
    # best_th = 0.43
    # model = utils.train_log_reg_model(x_train, y_train, seed=0, C=best_regularizer, upsample=True, verbose=True)
    # model_name = "logreg_model"
    # model_type = ModelTypes.LINEAR

    # MODEL 2: RANDOM FOREST
    # Uses default threshold of 0.5 and model parameters
    best_th = 0.5
    model = utils.train_rf_model(x_train, y_train, seed=0, upsample=True, verbose=True)
    model_name = "randomforest_model"
    model_type = ModelTypes.TREE

    # If model is in an sklearn pipeline, extract it
    (
        shap_values, 
        base_shap_values, 
        global_explainability, 
        fairness_metrics,
    ) = compute_log_metrics(model=model, x_train=x_train, 
                            x_test=x_test, y_test=y_test, 
                            best_th=best_th,
                            model_name=model_name, model_type=model_type)

    # IMPORTANT: EVALUATE MODEL ON UI
    print("Saving model pipeline!")
    with open(OUTPUT_MODEL_PATH, "wb") as model_file:
        pickle.dump(model, model_file)

    # Save feature names
    # with open(FEATURE_COLS_PATH, "wb") as file:
    #     pickle.dump(feature_cols, file)

    # IMPORTANT: LOG TRAINING MODEL ON UI to compare to DEPLOYED MODEL
    train_prob = model.predict_proba(x_train)[:, 1]
    train_pred = np.where(train_prob > best_th, 1, 0)

    ModelMonitoringService.export_text(
        features=x_train.iteritems(),
        inference=train_pred.tolist(),
    )

    print("Done!")

if __name__ == "__main__":
    try:
        print("Hello world")
        main()
    except Exception as e:
        print(e)
        print("Hmm something went wrong...")
        print("What?!")