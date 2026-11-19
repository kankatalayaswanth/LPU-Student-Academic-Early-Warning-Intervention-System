"""
LPU Student Academic Early-Warning & Intervention System
Module: Explainable AI (XAI) & Actionable Intervention Engine
File: src/explain.py

Computes SHAP feature attributions (TreeExplainer) and converts
local risk factors into concrete, prioritized faculty/mentor interventions.
Includes safe NaN handling for missing student telemetry.
"""

import numpy as np
import pandas as pd
import shap

def safe_val(row, col, default=0.0):
    """
    Safely retrieves a value from a dictionary, Series, or row object.
    Checks pd.isna() and returns default if the value is NaN, None, or absent.
    """
    if isinstance(row, (dict, pd.Series)):
        val = row.get(col, default)
    elif hasattr(row, "__getitem__"):
        try:
            val = row[col]
        except (KeyError, IndexError):
            val = default
    else:
        val = default

    if val is None or pd.isna(val):
        return default
    return val

def explain_student_risk(student_df: pd.DataFrame, pipeline, explainer, feature_names: list, class_idx: int = 2):
    """
    Computes local SHAP explanation for a single student or batch of students.
    Default class_idx=2 corresponds to 'High Risk'.
    Returns:
        shap_df: DataFrame with Feature, SHAP_Value, Absolute_Impact, Feature_Value
        probs: Array of predicted class probabilities [P(Low), P(Medium), P(High)]
        pred_label: Predicted risk label ('Low', 'Medium', 'High')
    """
    # 1. Apply domain feature engineering
    fe = pipeline.named_steps["feature_engineer"]
    X_fe = fe.transform(student_df)
    
    # 2. Apply preprocessing (imputation, IQR capping, scaling, OHE)
    preprocessor = pipeline.named_steps["preprocessor"]
    X_transformed = preprocessor.transform(X_fe)
    
    # 3. Model predictions
    clf = pipeline.named_steps["clf"]
    probs = clf.predict_proba(X_transformed)[0]
    pred_code = clf.predict(X_transformed)[0]
    code_to_label = {0: "Low", 1: "Medium", 2: "High"}
    pred_label = code_to_label.get(pred_code, "Medium")
    
    # 4. Compute SHAP values
    shap_exp = explainer(X_transformed)
    
    # Check dimensionality: (n_samples, n_features, n_classes) or (n_samples, n_features)
    if len(shap_exp.values.shape) == 3:
        # Multiclass: take class_idx (2 = High Risk)
        values = shap_exp.values[0, :, class_idx]
    else:
        values = shap_exp.values[0, :]
        
    # Get the raw/scaled feature values for display
    # We display the readable engineered feature values with safe NaN fallback
    readable_values = []
    first_row = student_df.iloc[0]
    first_row_fe = X_fe.iloc[0]

    for f in feature_names:
        if f.startswith("Department_"):
            dept_name = f.replace("Department_", "")
            readable_values.append(1 if first_row.get("Department") == dept_name else 0)
        elif f in X_fe.columns:
            val = first_row_fe.get(f)
            if pd.isna(val):
                readable_values.append(float(safe_val(first_row, f, 0.0)))
            else:
                readable_values.append(float(val))
        else:
            readable_values.append(0.0)
            
    shap_df = pd.DataFrame({
        "Feature": feature_names,
        "SHAP_Value": values,
        "Abs_SHAP": np.abs(values),
        "Student_Value": readable_values
    })
    
    # Clean feature names for clean UI display
    feature_label_map = {
        "Semester": "Semester",
        "Attendance_Pct": "Attendance %",
        "CA1_Marks": "CA1 Marks",
        "CA2_Marks": "CA2 Marks",
        "CA3_Marks": "CA3 Marks",
        "MidTerm_Marks": "Mid-Term Marks",
        "Prev_Sem_CGPA": "Previous CGPA",
        "Assignment_Submission_Rate": "Assignment Submission Rate %",
        "LMS_Logins_Per_Week": "LMS Logins / Week",
        "Active_Backlogs": "Active Backlogs Count",
        "Study_Hours_Per_Week": "Study Hours / Week",
        "Participation_Score": "Class Participation (1-10)",
        "CA_Average": "Continuous Assessment Avg",
        "CA_Trend": "CA Trajectory (CA3 - CA1)",
        "Engagement_Index": "Blended Engagement Index",
        "Attendance_Shortage_Flag": "Attendance Shortage Flag (<75%)"
    }
    
    shap_df["Display_Feature"] = shap_df["Feature"].apply(
        lambda x: feature_label_map.get(x, x.replace("Department_", "Dept: "))
    )
    
    shap_df = shap_df.sort_values(by="Abs_SHAP", ascending=False).reset_index(drop=True)
    return shap_df, probs, pred_label

def generate_actionable_interventions(student_row: dict) -> list:
    """
    Translates raw student academic metrics and early warning signals
    into concrete, prioritized, rule-based mentor interventions.
    Utilizes safe_val to guarantee graceful handling of missing (NaN) values.
    """
    interventions = []
    
    attendance = float(safe_val(student_row, "Attendance_Pct", 75.0))
    ca1 = float(safe_val(student_row, "CA1_Marks", 50.0))
    ca2 = float(safe_val(student_row, "CA2_Marks", 50.0))
    ca3 = float(safe_val(student_row, "CA3_Marks", 50.0))
    ca_avg = (ca1 + ca2 + ca3) / 3.0
    ca_trend = ca3 - ca1
    midterm = float(safe_val(student_row, "MidTerm_Marks", 50.0))
    cgpa = float(safe_val(student_row, "Prev_Sem_CGPA", 7.0))
    backlogs = int(safe_val(student_row, "Active_Backlogs", 0))
    assignments = float(safe_val(student_row, "Assignment_Submission_Rate", 75.0))
    lms_logins = float(safe_val(student_row, "LMS_Logins_Per_Week", 14.0))
    study_hours = float(safe_val(student_row, "Study_Hours_Per_Week", 15.0))
    
    # 1. Debarment & Attendance Non-Compliance (LPU Policy Section 14.1)
    if attendance < 75.0:
        interventions.append({
            "priority": "CRITICAL",
            "priority_color": "#DC2626",
            "category": "Attendance & Compliance",
            "title": "Mandatory Debarment Warning",
            "action": f"Current attendance ({attendance:.1f}%) is below the mandatory 75.0% threshold. Issue official intimation under Academic Section 14.1. Conduct mandatory parent-mentor conference within 48 hours and restrict non-academic event duty leaves.",
            "metric": f"{attendance:.1f}% Attendance"
        })
    elif attendance < 80.0:
        interventions.append({
            "priority": "MEDIUM",
            "priority_color": "#D97706",
            "category": "Attendance Monitoring",
            "title": "Attendance Margin Alert",
            "action": f"Attendance ({attendance:.1f}%) is nearing critical zone. Direct student to maintain 100% attendance over next 3 weeks to prevent debarment in upcoming exams.",
            "metric": f"{attendance:.1f}% Attendance"
        })

    # 2. Active Backlogs Clearance
    if backlogs > 0:
        interventions.append({
            "priority": "CRITICAL",
            "priority_color": "#DC2626",
            "category": "Degree Progression",
            "title": f"Backlog Recovery Plan ({backlogs} Active Backlog{'s' if backlogs>1 else ''})",
            "action": f"Student has {backlogs} un-cleared course backlog(s). Cap registration for supplementary elective credits. Assign curated 5-year solved question banks and enroll in Summer/Weekend remedial coaching sessions.",
            "metric": f"{backlogs} Backlog(s)"
        })

    # 3. Continuous Assessment Deterioration
    if ca_trend < -5.0 or ca_avg < 45.0:
        interventions.append({
            "priority": "HIGH",
            "priority_color": "#DC2626",
            "category": "Continuous Assessment",
            "title": "Remedial Peer-Tutoring & Concept Recovery",
            "action": f"CA trajectory dropped by {abs(ca_trend):.1f} marks across assessments (CA Average: {ca_avg:.1f}/100). Pair student with Departmental Peer-Tutoring Cell for bi-weekly doubt-clearing sessions before End-Term exams.",
            "metric": f"CA Avg {ca_avg:.1f} (Trend: {ca_trend:+.1f})"
        })

    # 4. Mid-Term Examination Deficit
    if midterm < 45.0:
        interventions.append({
            "priority": "HIGH",
            "priority_color": "#DC2626",
            "category": "Mid-Term Performance",
            "title": "Faculty Office-Hour Guided Problem Sessions",
            "action": f"Mid-Term score ({midterm:.1f}/100) reflects core concept gaps. Require student to attend 2 hours of weekly faculty office hours and complete structured revision modules.",
            "metric": f"{midterm:.1f}/100 Mid-Term"
        })

    # 5. Low Digital & Assignment Engagement
    if assignments < 65.0 or lms_logins < 6:
        interventions.append({
            "priority": "MEDIUM",
            "priority_color": "#D97706",
            "category": "Digital Engagement",
            "title": "LMS Access Audit & Assignment Grace Protocol",
            "action": f"Low digital participation (LMS logins: {lms_logins:.0f}/week, Submissions: {assignments:.1f}%). IT/Mentor must verify student LMS credentials and extend a 5-day supervised grace submission window for overdue assignments.",
            "metric": f"{assignments:.1f}% Submissions | {lms_logins:.0f} Logins/Wk"
        })

    # 6. Low Cumulative Grade Point Average (Probation)
    if cgpa < 6.0:
        interventions.append({
            "priority": "HIGH",
            "priority_color": "#DC2626",
            "category": "Academic Standing",
            "title": "Academic Probation Mentorship Protocol",
            "action": f"Cumulative GPA ({cgpa:.2f}/10.0) is at probation risk. Assign senior faculty mentor for bi-weekly academic progress reviews submitted directly to Department Head.",
            "metric": f"{cgpa:.2f} CGPA"
        })

    # 7. Low Independent Study Hours
    if study_hours < 10.0:
        interventions.append({
            "priority": "MEDIUM",
            "priority_color": "#D97706",
            "category": "Habit & Study Routine",
            "title": "Structured Study Timetable Restructuring",
            "action": f"Reported study time ({study_hours:.1f} hrs/week) is inadequate for course load. Mentor to structure a 15-hour weekly timetable allocating 2 hours daily to core engineering subjects.",
            "metric": f"{study_hours:.1f} hrs/week"
        })

    # 8. Commendation / Honors if all metrics are healthy
    if len(interventions) == 0:
        interventions.append({
            "priority": "LOW RISK / COMMENDATION",
            "priority_color": "#16A34A",
            "category": "Honors & Enrichment",
            "title": "Advanced Competitive Track & Research Fellowship",
            "action": "Student demonstrates exemplary academic consistency, high attendance, and strong assessment scores. Nominate for departmental honors research projects, national hackathons, and student peer-mentor leadership roles.",
            "metric": f"{attendance:.1f}% Attendance | {cgpa:.2f} CGPA | {ca_avg:.1f} CA Avg"
        })

    return interventions
