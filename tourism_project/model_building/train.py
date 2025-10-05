"""
Tourism Package Prediction - Model Training Script
--------------------------------------------------
This script:
 - Loads preprocessed training and test data from the Hugging Face Hub
 - Builds a preprocessing + XGBoost pipeline
 - Performs hyperparameter tuning via GridSearchCV
 - Logs metrics & artifacts to MLflow
 - Uploads the trained model to Hugging Face Model Hub
"""

# ========== Imports ==========
import os
import pandas as pd
import xgboost as xgb
import joblib
import mlflow

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError


# ========== Main Function ==========
def main():
    print("🚀 Starting model training pipeline...")

    # -------------------------
    # MLflow Setup
    # -------------------------
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("visit-with-us-production-training")

    api = HfApi()

    # -------------------------
    # Load Data from Hugging Face Hub
    # -------------------------
    print("📥 Loading datasets from Hugging Face Hub...")
    repo_base = "hf://datasets/Yash0204/prediction-tourism-mlops/processed"
    Xtrain = pd.read_csv(f"{repo_base}/Xtrain.csv")
    Xtest  = pd.read_csv(f"{repo_base}/Xtest.csv")
    ytrain_df = pd.read_csv(f"{repo_base}/ytrain.csv")
    ytest_df  = pd.read_csv(f"{repo_base}/ytest.csv")

    def to_series(yframe):
        """Ensure y is a Series with integer values."""
        if isinstance(yframe, pd.Series):
            return yframe.astype(int)
        if yframe.shape[1] == 1:
            return yframe.iloc[:, 0].astype(int)
        return yframe['ProdTaken'].astype(int)

    ytrain = to_series(ytrain_df)
    ytest  = to_series(ytest_df)

    print(f"✅ Data loaded: {Xtrain.shape[0]} train samples, {Xtest.shape[0]} test samples")

    # -------------------------
    # Define Features
    # -------------------------
    numeric_features = [c for c in [
        'Age', 'NumberOfPersonVisiting', 'PreferredPropertyStar', 'NumberOfTrips',
        'NumberOfChildrenVisiting', 'MonthlyIncome', 'PitchSatisfactionScore',
        'NumberOfFollowups', 'DurationOfPitch', 'Passport', 'OwnCar'
    ] if c in Xtrain.columns]

    categorical_features = [c for c in [
        'TypeofContact', 'CityTier', 'Occupation', 'Gender',
        'MaritalStatus', 'Designation', 'ProductPitched'
    ] if c in Xtrain.columns]

    # -------------------------
    # Handle Class Imbalance
    # -------------------------
    value_counts = ytrain.value_counts()
    neg, pos = int(value_counts.get(0, 0)), int(value_counts.get(1, 0))
    if pos == 0:
        raise ValueError("No positive class in ytrain; cannot compute scale_pos_weight.")
    class_weight = neg / pos
    print(f"⚖️ Class balance → Neg: {neg}, Pos: {pos}, scale_pos_weight={class_weight:.2f}")

    # -------------------------
    # Preprocessing
    # -------------------------
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown='ignore'), categorical_features)
    )

    # -------------------------
    # XGBoost Model + Grid Search
    # -------------------------
    xgb_model = xgb.XGBClassifier(
        scale_pos_weight=class_weight,
        random_state=42,
        objective='binary:logistic',
        eval_metric='logloss',
        tree_method='hist',
        n_jobs=-1
    )

    param_grid = {
        'xgbclassifier__n_estimators': [25, 50, 75],
        'xgbclassifier__max_depth': [2, 3, 4],
        'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
        'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
        'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
        'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
    }

    model_pipeline = make_pipeline(preprocessor, xgb_model)

    # -------------------------
    # MLflow Run
    # -------------------------
    with mlflow.start_run():
        print("🧠 Training model with GridSearchCV...")
        grid_search = GridSearchCV(
            model_pipeline,
            param_grid,
            cv=5,
            n_jobs=-1,
            scoring='f1',
            verbose=0
        )
        grid_search.fit(Xtrain, ytrain)

        print("✅ Grid search complete.")
        mlflow.log_params(grid_search.best_params_)

        # Evaluate Best Model
        best_model = grid_search.best_estimator_
        classification_threshold = 0.45

        y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
        y_pred_test_proba  = best_model.predict_proba(Xtest)[:, 1]

        y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)
        y_pred_test  = (y_pred_test_proba  >= classification_threshold).astype(int)

        train_report = classification_report(ytrain, y_pred_train, output_dict=True, zero_division=0)
        test_report  = classification_report(ytest,  y_pred_test,  output_dict=True, zero_division=0)

        # Log Metrics
        mlflow.log_metrics({
            "train_accuracy": train_report['accuracy'],
            "train_precision": train_report['1']['precision'],
            "train_recall":    train_report['1']['recall'],
            "train_f1-score":  train_report['1']['f1-score'],
            "test_accuracy":   test_report['accuracy'],
            "test_precision":  test_report['1']['precision'],
            "test_recall":     test_report['1']['recall'],
            "test_f1-score":   test_report['1']['f1-score'],
            "threshold":       classification_threshold
        })

        print("\n📊 Model Performance Summary:")
        print(f"Train F1: {train_report['1']['f1-score']:.3f}")
        print(f"Test  F1: {test_report['1']['f1-score']:.3f}")
        print(f"Accuracy: {test_report['accuracy']:.3f}")

        # -------------------------
        # Save & Log Model
        # -------------------------
        os.makedirs("artifacts", exist_ok=True)
        model_path = "artifacts/best_tourism_wellness_model_v1.joblib"
        joblib.dump(best_model, model_path)
        mlflow.log_artifact(model_path, artifact_path="model")

        print(f"💾 Model saved locally at: {model_path}")

        # -------------------------
        # Upload to Hugging Face Hub
        # -------------------------
        repo_id = "Yash0204/prediction-tourism-mlops"
        repo_type = "model"

        try:
            api.repo_info(repo_id=repo_id, repo_type=repo_type)
            print(f"📡 Model repo '{repo_id}' exists. Uploading...")
        except RepositoryNotFoundError:
            print(f"🆕 Repo '{repo_id}' not found. Creating...")
            create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
            print(f"✅ Repo '{repo_id}' created.")

        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo=os.path.basename(model_path),
            repo_id=repo_id,
            repo_type=repo_type,
        )
        print(f"✅ Uploaded '{model_path}' to Hugging Face Hub under '{repo_id}'.")

    print("\n🎉 Training pipeline completed successfully!")


# ========== Entry Point ==========
if __name__ == "__main__":
    main()
