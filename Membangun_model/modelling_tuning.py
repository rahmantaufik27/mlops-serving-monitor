import os
import warnings
import sys
from dotenv import load_dotenv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import mlflow
from mlflow.client import MlflowClient
from mlflow.models.signature import infer_signature
import mlflow.sklearn
import logging
logging.basicConfig(filename='app.log', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")
load_dotenv()

def run_rf_hypertuning_model_mlflow(df):
    # set dagshub
    uri_dagshub = "https://dagshub.com/rahmantaufik27/mlflow-sml-rtaufik27.mlflow"
    mlflow.set_tracking_uri(uri_dagshub)
    os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv('USERNAME')
    os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv('TOKEN')

    # set mlflow
    experiment_name = "attrition_prediction"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = None
    client = MlflowClient()
    if experiment is None:
        experiment_id = client.create_experiment(name = experiment_name)
        print(f"Eksperimen '{experiment_name}' berhasil dibuat dengan ID: {experiment_id}")
        experiment = mlflow.get_experiment(experiment_id)
    else:
        print(f"Eksperimen '{experiment_name}' sudah ada")
    # print(experiment)

    # set data
    y = df['Attrition']
    X = df.drop(['Attrition'], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # set rentang hyperparameter
    param_grid = {
        'n_estimators': [50, 100, 150], # Jumlah pohon dalam hutan
        'max_depth': [None, 10, 20],     # Kedalaman maksimum pohon
        'min_samples_split': [2, 5, 10], # Jumlah sampel minimum yang dibutuhkan untuk membagi sebuah node
        'min_samples_leaf': [1, 2, 4]    # Jumlah sampel minimum yang dibutuhkan di leaf node
    }
    n_splits = 5
    random_state = 42

    # running mlflow
    mlflow.sklearn.autolog()
    
    with mlflow.start_run(run_name="rf-tuning-model", experiment_id = experiment.experiment_id) as run:

        # training model with hyperparameter tuning
        cv_strategy = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

        model_rf_tuning = GridSearchCV(
            estimator = RandomForestClassifier(random_state=random_state),
            param_grid = param_grid,
            cv = cv_strategy,
            scoring = 'accuracy', # Metrik yang dioptimalkan oleh GridSearch
            n_jobs = -1,          # Gunakan semua core CPU yang tersedia
            verbose = 2,          # Output detail proses
            return_train_score = False # Tidak perlu mengembalikan train score untuk tuning
        )
        model_rf_tuning.fit(X_train, y_train)
        y_pred = model_rf_tuning.best_estimator_.predict(X_test)

        test_accuracy = accuracy_score(y_test, y_pred)
        test_precision = precision_score(y_test, y_pred, zero_division=0)
        test_recall = recall_score(y_test, y_pred, zero_division=0)
        test_f1_score = f1_score(y_test, y_pred, zero_division=0)

        # logging parameter
        best_params = model_rf_tuning.best_params_
        print(f"Hyperparameter terbaik ditemukan: {best_params}")
        for param_name, param_value in best_params.items():
            mlflow.log_param(f"tuned_{param_name}", param_value)

        # logging metrics
        mlflow.log_metric("test accuracy", test_accuracy)
        mlflow.log_metric("test precision", test_precision)
        mlflow.log_metric("test recall", test_recall)
        mlflow.log_metric("test f1-score", test_f1_score)

        # logging model
        model_signature = infer_signature(model_input=X_train, model_output=y_train)

        tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme
        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                sk_model = model_rf_tuning,
                artifact_path = "model-tuning",
                signature = model_signature,
                input_example = X_train.head(1),
                registered_model_name = "rf_model_tuning"
                )
        else:
            mlflow.sklearn.log_model(
                sk_model = model_rf_tuning,
                artifact_path = "model-tuning",
                signature = model_signature,
                input_example = X_train.head(1),
                )
            
        # print run info    
        print("run_id: {}".format(run.info.run_id))
        print("experiment_id: {}".format(run.info.experiment_id))
        print("status: {}".format(run.info.status))
        print("start_time: {}".format(run.info.start_time))
        print("end_time: {}".format(run.info.end_time))
        print("lifecycle_stage: {}".format(run.info.lifecycle_stage))

if __name__ == "__main__":
    # load dataset
    dataset = "employee_preprocessing.csv"
    df = pd.read_csv(dataset)
    # proses tuning model with mlflow and dagshub
    run_rf_hypertuning_model_mlflow(df)