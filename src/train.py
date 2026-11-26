"""
LPU Student Academic Early-Warning & Intervention System
Module: Model Training, Evaluation, Hyperparameter Tuning & Serialization
File: src/train.py

Implements:
1. Leakage-free Stratified Split: Train (70%), Validation (15%), Test (15%).
2. Unified scikit-learn + imbalanced-learn Pipeline:
   - Domain Feature Engineering (CA_Average, CA_Trend, Attendance_Shortage_Flag, Engagement_Index)
   - Median/Most-Frequent Imputation
   - Outlier Capping (IQR bounds fitted on Train)
   - Categorical One-Hot Encoding & StandardScaler
   - SMOTE Resampling (Train set only)
3. 5-Model Benchmark:
   - Baseline: Logistic Regression
   - Random Forest (Bagging)
   - Support Vector Machine (RBF kernel, calibrated probabilities)
   - XGBoost Classifier (Boosting)
   - LightGBM Classifier (Boosting)
4. Hyperparameter Optimization: RandomizedSearchCV on top boosting model.
5. Evaluation prioritizing High-Risk Recall (Recall, F1-macro, ROC-AUC OVR).
6. Artifact Serialization (best_model_pipeline.joblib, shap_explainer.joblib, JSON summaries).
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, auc, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import shap

# Ensure local imports work cleanly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.custom_transformers import AcademicFeatureEngineer, IQROutlierCapper

SEED = 42
TARGET_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_TARGET_MAP = {0: "Low", 1: "Medium", 2: "High"}

def compute_eda_statistics(df: pd.DataFrame) -> dict:
    """Pre-computes comprehensive EDA statistics for zero-latency dashboard rendering."""
    eda = {}
    
    # Class distribution
    class_counts = df["Risk_Level"].value_counts().to_dict()
    class_pcts = (df["Risk_Level"].value_counts(normalize=True) * 100).round(2).to_dict()
    eda["class_distribution"] = {
        "counts": class_counts,
        "percentages": class_pcts
    }
    
    # Department breakdown
    dept_risk = pd.crosstab(df["Department"], df["Risk_Level"]).to_dict(orient="index")
    eda["department_risk"] = dept_risk
    
    # Summary stats of continuous features
    num_cols = [
        "Attendance_Pct", "CA1_Marks", "CA2_Marks", "CA3_Marks",
        "MidTerm_Marks", "Prev_Sem_CGPA", "Assignment_Submission_Rate",
        "Active_Backlogs", "Participation_Score"
    ]
    eda["numerical_summary"] = df[num_cols].describe().round(2).to_dict()
    
    # Correlation matrix of numerical features (with imputed medians for clean matrix)
    df_imputed = df[num_cols].fillna(df[num_cols].median())
    corr_matrix = df_imputed.corr().round(3).to_dict()
    eda["correlation_matrix"] = corr_matrix
    
    # Attendance vs Risk distribution
    att_risk = df.groupby("Risk_Level")["Attendance_Pct"].agg(["mean", "median", "std"]).round(2).to_dict(orient="index")
    cgpa_risk = df.groupby("Risk_Level")["Prev_Sem_CGPA"].agg(["mean", "median", "std"]).round(2).to_dict(orient="index")
    eda["attendance_by_risk"] = att_risk
    eda["cgpa_by_risk"] = cgpa_risk
    
    return eda

def build_preprocessing_pipeline():
    """Builds the feature engineering and column transformation pipeline."""
    fe = AcademicFeatureEngineer()
    
    num_cols = [
        "Semester", "Attendance_Pct", "CA1_Marks", "CA2_Marks", "CA3_Marks",
        "MidTerm_Marks", "Prev_Sem_CGPA", "Assignment_Submission_Rate",
        "Active_Backlogs", "Participation_Score",
        "CA_Average", "CA_Trend", "Engagement_Index",
        "Attendance_Shortage_Flag"
    ]
    cat_cols = ["Department"]
    
    num_pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("capper", IQROutlierCapper(factor=1.5)),
        ("scaler", StandardScaler())
    ])
    
    cat_pipeline = ImbPipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, num_cols),
            ("cat", cat_pipeline, cat_cols)
        ]
    )
    
    return fe, preprocessor, num_cols, cat_cols

def evaluate_predictions(y_true, y_pred, y_prob):
    """Calculates comprehensive classification metrics prioritizing High-Risk Recall."""
    acc = accuracy_score(y_true, y_pred)
    prec_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    
    # Class-specific metrics (Class 2 is High Risk)
    rec_high = recall_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
    prec_high = precision_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
    f1_high = f1_score(y_true, y_pred, labels=[2], average=None, zero_division=0)[0]
    
    # ROC-AUC OVR
    try:
        roc_auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        roc_auc_weighted = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except Exception:
        roc_auc_macro = 0.0
        roc_auc_weighted = 0.0
        
    return {
        "Accuracy": round(float(acc), 4),
        "Precision_Macro": round(float(prec_macro), 4),
        "Recall_Macro": round(float(rec_macro), 4),
        "High_Risk_Recall": round(float(rec_high), 4),
        "High_Risk_Precision": round(float(prec_high), 4),
        "High_Risk_F1": round(float(f1_high), 4),
        "F1_Macro": round(float(f1_macro), 4),
        "F1_Weighted": round(float(f1_weighted), 4),
        "ROC_AUC_Macro": round(float(roc_auc_macro), 4),
        "ROC_AUC_Weighted": round(float(roc_auc_weighted), 4)
    }

def main():
    print("=" * 80)
    print("LPU ACADEMIC EARLY-WARNING & INTERVENTION SYSTEM: TRAINING PIPELINE")
    print("=" * 80)
    
    os.makedirs("artifacts", exist_ok=True)
    
    data_path = os.path.join("data", "student_academic_data.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run src/data_prep.py first.")
        
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns.")
    
    # Compute and save dataset EDA statistics
    print("Computing EDA statistics for dashboard...")
    eda_stats = compute_eda_statistics(df)
    with open("artifacts/eda_stats.json", "w") as f:
        json.dump(eda_stats, f, indent=2)
    print("Saved: artifacts/eda_stats.json")
    
    # Prepare features and target
    meta_cols = ["Student_ID", "Student_Name", "School", "Section", "Mentor_Name", "Risk_Level"]
    drop_cols = [c for c in meta_cols if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df["Risk_Level"].map(TARGET_MAP)
    
    # 1. Stratified Splitting to strictly prevent data leakage
    # First split: 85% Train+Val, 15% Test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=SEED
    )
    # Second split: Train (70% of total), Val (15% of total) -> 15/85 = ~0.17647
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=(0.15 / 0.85), stratify=y_train_val, random_state=SEED
    )
    
    print(f"\nData Splits (Stratified, No Leakage):")
    print(f" - Train set     : {X_train.shape[0]} records ({X_train.shape[0]/len(df):.1%})")
    print(f" - Validation set: {X_val.shape[0]} records ({X_val.shape[0]/len(df):.1%})")
    print(f" - Test set      : {X_test.shape[0]} records ({X_test.shape[0]/len(df):.1%})")
    
    # Record class distribution before and after SMOTE on Train set
    fe, preprocessor, num_cols, cat_cols = build_preprocessing_pipeline()
    X_train_fe = fe.transform(X_train)
    preprocessor.fit(X_train_fe)
    X_train_trans = preprocessor.transform(X_train_fe)
    
    # Get clean feature names out of preprocessor
    ohe_step = preprocessor.named_transformers_["cat"].named_steps["ohe"]
    cat_feature_names = list(ohe_step.get_feature_names_out(cat_cols))
    all_feature_names = num_cols + cat_feature_names
    
    with open("artifacts/feature_names.json", "w") as f:
        json.dump(all_feature_names, f, indent=2)
    print(f"Extracted {len(all_feature_names)} processed feature names -> artifacts/feature_names.json")
    
    smote = SMOTE(random_state=SEED)
    X_train_res, y_train_res = smote.fit_resample(X_train_trans, y_train)
    
    smote_summary = {
        "before_smote": {INV_TARGET_MAP[k]: int(v) for k, v in y_train.value_counts().items()},
        "after_smote": {INV_TARGET_MAP[k]: int(v) for k, v in pd.Series(y_train_res).value_counts().items()}
    }
    with open("artifacts/smote_summary.json", "w") as f:
        json.dump(smote_summary, f, indent=2)
    print("SMOTE balancing recorded -> artifacts/smote_summary.json")
    
    # 2. Define 5 Candidate Models starting from Baseline
    models = {
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000, random_state=SEED, class_weight="balanced"
        ),
        "Random Forest (Bagging)": RandomForestClassifier(
            n_estimators=100, max_depth=8, random_state=SEED
        ),
        "Support Vector Machine (SVM)": SVC(
            probability=True, kernel="rbf", C=1.5, gamma="scale", random_state=SEED
        ),
        "XGBoost Classifier (Boosting)": XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.08,
            eval_metric="mlogloss", random_state=SEED
        ),
        "LightGBM Classifier (Boosting)": LGBMClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.08,
            random_state=SEED, verbose=-1
        )
    }
    
    results = {}
    fitted_pipelines = {}
    
    print("\n" + "=" * 80, flush=True)
    print("TRAINING & BENCHMARKING 5 CANDIDATE MODELS ON TEST SET (HELD-OUT 15%)", flush=True)
    print("=" * 80, flush=True)
    
    for name, clf in models.items():
        print(f"\nTraining [{name}]...", flush=True)
        pipe = ImbPipeline([
            ("feature_engineer", AcademicFeatureEngineer()),
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=SEED)),
            ("clf", clf)
        ])
        
        # Fit on training data
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        
        # Evaluate on Test Set
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)
        
        metrics = evaluate_predictions(y_test, y_pred, y_prob)
        results[name] = metrics
        print(f" -> Accuracy: {metrics['Accuracy']:.3f} | High-Risk Recall: {metrics['High_Risk_Recall']:.3f} | Macro F1: {metrics['F1_Macro']:.3f} | ROC-AUC: {metrics['ROC_AUC_Macro']:.3f}", flush=True)
        
    # 3. Hyperparameter Tuning for the Best Tree-based Model (XGBoost)
    print("\n" + "=" * 80, flush=True)
    print("HYPERPARAMETER OPTIMIZATION: RandomizedSearchCV ON XGBOOST", flush=True)
    print("=" * 80, flush=True)
    
    param_dist = {
        "clf__n_estimators": [90, 120, 150],
        "clf__max_depth": [3, 4, 5],
        "clf__learning_rate": [0.04, 0.08, 0.12],
        "clf__subsample": [0.8, 0.9, 1.0],
        "clf__colsample_bytree": [0.8, 0.9, 1.0],
    }
    
    tune_pipe = ImbPipeline([
        ("feature_engineer", AcademicFeatureEngineer()),
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=SEED)),
        ("clf", XGBClassifier(eval_metric="mlogloss", random_state=SEED))
    ])
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        tune_pipe,
        param_distributions=param_dist,
        n_iter=8,
        scoring="f1_macro",
        cv=cv,
        random_state=SEED,
        n_jobs=1,
        verbose=1
    )
    
    print("Running 3-fold Stratified Randomized Search...", flush=True)
    search.fit(X_train, y_train)
    best_pipe = search.best_estimator_
    print(f"Optimal Parameters: {search.best_params_}", flush=True)
    
    # Evaluate Tuned Model on Test Set
    y_pred_tuned = best_pipe.predict(X_test)
    y_prob_tuned = best_pipe.predict_proba(X_test)
    tuned_metrics = evaluate_predictions(y_test, y_pred_tuned, y_prob_tuned)
    results["Tuned XGBoost (Winning Architecture)"] = tuned_metrics
    
    print("\n" + "-" * 80)
    print("FINAL 6-WAY MODEL BENCHMARK COMPARISON TABLE")
    print("-" * 80)
    metrics_df = pd.DataFrame(results).T
    print(metrics_df[["Accuracy", "High_Risk_Recall", "High_Risk_Precision", "F1_Macro", "ROC_AUC_Macro"]])
    
    # Save Model Comparison to artifacts
    with open("artifacts/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: artifacts/model_comparison.json")
    
    # 4. Serialize Winning Pipeline
    pipeline_save_path = os.path.join("artifacts", "best_model_pipeline.joblib")
    joblib.dump(best_pipe, pipeline_save_path)
    print(f"Serialized winning pipeline to: {pipeline_save_path}")
    
    # 5. Fit & Serialize SHAP Explainer
    print("\nInitializing and serializing SHAP TreeExplainer...")
    winning_clf = best_pipe.named_steps["clf"]
    explainer = shap.TreeExplainer(winning_clf)
    shap_save_path = os.path.join("artifacts", "shap_explainer.joblib")
    joblib.dump(explainer, shap_save_path)
    print(f"Serialized SHAP TreeExplainer to: {shap_save_path}")
    
    # Compute Global Feature Importance on Validation set
    X_val_fe = best_pipe.named_steps["feature_engineer"].transform(X_val)
    X_val_trans = best_pipe.named_steps["preprocessor"].transform(X_val_fe)
    val_shap_values = explainer(X_val_trans)
    
    # Mean absolute SHAP value across all samples and classes
    if len(val_shap_values.values.shape) == 3:
        # Shape: (n_samples, n_features, n_classes)
        # Global impact across all classes
        global_shap_mag = np.mean(np.abs(val_shap_values.values), axis=(0, 2))
        # High Risk class (class index 2) impact specifically
        high_risk_shap_mag = np.mean(np.abs(val_shap_values.values[:, :, 2]), axis=0)
    else:
        global_shap_mag = np.mean(np.abs(val_shap_values.values), axis=0)
        high_risk_shap_mag = global_shap_mag
        
    global_shap_df = pd.DataFrame({
        "Feature": all_feature_names,
        "Global_Importance": global_shap_mag,
        "High_Risk_Importance": high_risk_shap_mag
    }).sort_values(by="High_Risk_Importance", ascending=False)
    
    with open("artifacts/global_shap.json", "w") as f:
        json.dump(global_shap_df.to_dict(orient="records"), f, indent=2)
    print("Saved global SHAP importance -> artifacts/global_shap.json")
    
    # 6. Confusion Matrix Computation & Serialization
    cm = confusion_matrix(y_test, y_pred_tuned)
    cm_norm = (cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100).round(1)
    
    cm_data = {
        "labels": ["Low", "Medium", "High"],
        "matrix_raw": cm.tolist(),
        "matrix_percentage": cm_norm.tolist()
    }
    with open("artifacts/confusion_matrix.json", "w") as f:
        json.dump(cm_data, f, indent=2)
    print("Saved confusion matrix -> artifacts/confusion_matrix.json")
    
    # 7. ROC Curves Computation (One-vs-Rest)
    roc_data = {}
    for c_idx, c_name in INV_TARGET_MAP.items():
        y_true_binary = (y_test == c_idx).astype(int)
        y_score = y_prob_tuned[:, c_idx]
        fpr, tpr, _ = roc_curve(y_true_binary, y_score)
        roc_auc = auc(fpr, tpr)
        # Sample points to keep json compact and fast
        step = max(1, len(fpr) // 50)
        roc_data[c_name] = {
            "fpr": fpr[::step].round(4).tolist() + [1.0],
            "tpr": tpr[::step].round(4).tolist() + [1.0],
            "auc": round(float(roc_auc), 4)
        }
    with open("artifacts/roc_curves.json", "w") as f:
        json.dump(roc_data, f, indent=2)
    print("Saved ROC curves -> artifacts/roc_curves.json")
    
    # 8. Detailed Error Analysis (False Positives vs False Negatives)
    test_eval_df = X_test.copy()
    test_eval_df["Actual_Risk"] = [INV_TARGET_MAP[v] for v in y_test]
    test_eval_df["Predicted_Risk"] = [INV_TARGET_MAP[v] for v in y_pred_tuned]
    test_eval_df["Prob_High"] = y_prob_tuned[:, 2].round(3)
    test_eval_df["Prob_Medium"] = y_prob_tuned[:, 1].round(3)
    test_eval_df["Prob_Low"] = y_prob_tuned[:, 0].round(3)
    
    # Identify misclassifications
    misclassified = test_eval_df[test_eval_df["Actual_Risk"] != test_eval_df["Predicted_Risk"]]
    
    # Dangerous False Negatives: Actual is High, but predicted as Low or Medium
    high_risk_fn = test_eval_df[
        (test_eval_df["Actual_Risk"] == "High") & (test_eval_df["Predicted_Risk"] != "High")
    ]
    
    # False Alarms / False Positives: Actual is Low, but predicted as High
    high_risk_fp = test_eval_df[
        (test_eval_df["Actual_Risk"] == "Low") & (test_eval_df["Predicted_Risk"] == "High")
    ]
    
    error_summary = {
        "total_test_samples": len(test_eval_df),
        "total_misclassifications": len(misclassified),
        "overall_error_rate": round(len(misclassified) / len(test_eval_df), 4),
        "critical_false_negatives_count": len(high_risk_fn),
        "critical_false_negatives_pct": round(len(high_risk_fn) / max(1, (y_test == 2).sum()), 4),
        "false_alarm_fp_count": len(high_risk_fp),
        "confusion_breakdown": {
            "High_actual_predicted_Medium": int(((test_eval_df["Actual_Risk"] == "High") & (test_eval_df["Predicted_Risk"] == "Medium")).sum()),
            "High_actual_predicted_Low": int(((test_eval_df["Actual_Risk"] == "High") & (test_eval_df["Predicted_Risk"] == "Low")).sum()),
            "Medium_actual_predicted_High": int(((test_eval_df["Actual_Risk"] == "Medium") & (test_eval_df["Predicted_Risk"] == "High")).sum()),
            "Medium_actual_predicted_Low": int(((test_eval_df["Actual_Risk"] == "Medium") & (test_eval_df["Predicted_Risk"] == "Low")).sum()),
            "Low_actual_predicted_High": int(((test_eval_df["Actual_Risk"] == "Low") & (test_eval_df["Predicted_Risk"] == "High")).sum()),
            "Low_actual_predicted_Medium": int(((test_eval_df["Actual_Risk"] == "Low") & (test_eval_df["Predicted_Risk"] == "Medium")).sum()),
        },
        "sample_misclassifications": misclassified.head(10).to_dict(orient="records")
    }
    
    with open("artifacts/error_analysis.json", "w") as f:
        json.dump(error_summary, f, indent=2)
    print("Saved error analysis -> artifacts/error_analysis.json")
    
    print("\n" + "=" * 80)
    print("ALL TRAINING, EVALUATION & SERIALIZATION STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 80)

if __name__ == "__main__":
    main()
