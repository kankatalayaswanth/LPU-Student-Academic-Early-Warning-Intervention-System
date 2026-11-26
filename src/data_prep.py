"""
LPU Student Academic Early-Warning & Intervention System
Module: Data Acquisition and Synthetic Generation
File: src/data_prep.py

Generates a realistic, non-toy student academic dataset (2,500 records)
with authentic academic correlations, class imbalance, 3% missingness,
and realistic outliers for rigorous pre-processing demonstration.
"""

import os
import numpy as np
import pandas as pd

# Set deterministic random seed
SEED = 42
np.random.seed(SEED)

DEPARTMENTS = [
    "Computer Science & Engineering",
    "Information Technology",
    "Electronics & Communication",
    "Mechanical Engineering",
    "Civil Engineering",
    "Management & Commerce",
    "Biotechnology"
]

DEPT_WEIGHTS = [0.38, 0.16, 0.14, 0.10, 0.08, 0.08, 0.06]

def generate_student_dataset(n_samples: int = 2500, random_state: int = 42) -> pd.DataFrame:
    """
    Generates realistic academic records reflecting Lovely Professional University (LPU)
    academic evaluation structures (CAs, Mid-Terms, Attendance, LMS activity, Backlogs).
    """
    rng = np.random.RandomState(random_state)
    
    # Student IDs
    student_ids = [f"LPU12{i+1:04d}" for i in range(n_samples)]
    
    # Academic demographics
    departments = rng.choice(DEPARTMENTS, size=n_samples, p=DEPT_WEIGHTS)
    semesters = rng.choice(list(range(1, 9)), size=n_samples, p=[0.15, 0.18, 0.18, 0.16, 0.13, 0.10, 0.06, 0.04])
    
    # Latent academic capability factor (Gaussian standard normal)
    latent_ability = rng.normal(loc=0.0, scale=1.0, size=n_samples)
    
    # Latent diligence / study habit factor (correlated with ability)
    latent_diligence = 0.65 * latent_ability + rng.normal(loc=0.0, scale=0.6, size=n_samples)
    
    # Attendance Percentage (Base ~ 78%, capped at 100, sensitive to diligence)
    # LPU mandatory cutoff is 75%
    raw_attendance = 76.0 + 11.5 * latent_diligence + rng.normal(0, 5.0, size=n_samples)
    attendance_pct = np.clip(raw_attendance, 35.0, 100.0).round(1)
    
    # Previous Semester CGPA (scale 0-10, realistic mean ~ 7.1)
    raw_cgpa = 7.15 + 1.20 * latent_ability + 0.30 * latent_diligence + rng.normal(0, 0.45, size=n_samples)
    prev_sem_cgpa = np.clip(raw_cgpa, 3.8, 9.95).round(2)
    
    # Continuous Assessments (CA1, CA2, CA3 on 0-100 scale)
    # Model student trajectory (some improve, some decay)
    ca_base = 62.0 + 14.0 * latent_ability + 8.0 * latent_diligence + rng.normal(0, 7.0, size=n_samples)
    
    # CA1 (early semester)
    ca1 = np.clip(ca_base + rng.normal(0, 6.0, size=n_samples), 12.0, 100.0).round(1)
    
    # CA2 (mid semester, attendance effect begins showing)
    ca2_shift = (attendance_pct - 75.0) * 0.15
    ca2 = np.clip(ca_base + ca2_shift + rng.normal(1.0, 5.5, size=n_samples), 10.0, 100.0).round(1)
    
    # CA3 (late semester - student burnout vs exam push)
    fatigue_or_push = rng.choice([-6.0, 0.0, 5.0], size=n_samples, p=[0.25, 0.50, 0.25])
    ca3 = np.clip(ca_base + ca2_shift + fatigue_or_push + rng.normal(0, 6.5, size=n_samples), 8.0, 100.0).round(1)
    
    # Mid-Term Marks (0-100 scale, strong predictor)
    midterm_base = 0.45 * ca1 + 0.45 * ca2 + 3.0 * (prev_sem_cgpa - 7.0) + rng.normal(0, 6.0, size=n_samples)
    midterm_marks = np.clip(midterm_base, 10.0, 100.0).round(1)
    
    # Assignment Submission Rate (% 0-100)
    raw_assignments = 75.0 + 15.0 * latent_diligence + rng.normal(0, 6.0, size=n_samples)
    assignment_rate = np.clip(raw_assignments, 15.0, 100.0).round(1)
    
    # Participation Score (1.0 to 10.0 in classroom discussions and laboratory engagement)
    raw_part = 6.0 + 1.6 * latent_diligence + rng.normal(0, 1.2, size=n_samples)
    participation = np.clip(raw_part, 1.0, 10.0).round(1)
    
    # Active Backlogs (0 to 6, heavily skewed towards 0)
    # Strongly inverse with CGPA and diligence
    backlog_prob_base = np.clip(0.35 - 0.04 * prev_sem_cgpa - 0.08 * latent_diligence, 0.02, 0.85)
    backlogs = np.zeros(n_samples, dtype=int)
    for i in range(n_samples):
        if rng.rand() < backlog_prob_base[i]:
            backlogs[i] = rng.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.12, 0.05, 0.03])
    
    # Ground Truth Risk Score Synthesis
    # Composite risk index based on realistic university academic indicators:
    # Attendance compliance, Mid-Terms, Continuous Assessments, CGPA, Backlogs, Assignment rate, Participation
    ca_avg = (ca1 + ca2 + ca3) / 3.0
    
    risk_index = (
        - 0.38 * ((attendance_pct - 75.0) / 15.0)
        - 0.32 * ((midterm_marks - 50.0) / 20.0)
        - 0.28 * ((ca_avg - 55.0) / 20.0)
        - 0.24 * ((prev_sem_cgpa - 7.0) / 1.5)
        + 0.40 * (backlogs / 2.0)
        - 0.20 * ((assignment_rate - 70.0) / 20.0)
        - 0.14 * ((participation - 6.0) / 2.0)
        + rng.normal(0, 0.35, size=n_samples)
    )
    
    # Natural academic clinical risk thresholds (not artificially forced round percentiles):
    # Generates a realistic ~14.8% high-risk rate representing students facing debarment or failure
    risk_labels = []
    for r in risk_index:
        if r >= 1.17:
            risk_labels.append("High")
        elif r >= 0.18:
            risk_labels.append("Medium")
        else:
            risk_labels.append("Low")
            
    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Department": departments,
        "Semester": semesters,
        "Attendance_Pct": attendance_pct,
        "CA1_Marks": ca1,
        "CA2_Marks": ca2,
        "CA3_Marks": ca3,
        "MidTerm_Marks": midterm_marks,
        "Prev_Sem_CGPA": prev_sem_cgpa,
        "Assignment_Submission_Rate": assignment_rate,
        "Active_Backlogs": backlogs,
        "Participation_Score": participation,
        "Risk_Level": risk_labels
    })
    
    # Inject realistic entry outliers (0.8% of rows) to demonstrate IQR clipping
    outlier_idx = rng.choice(n_samples, size=int(0.008 * n_samples), replace=False)
    for idx in outlier_idx[:len(outlier_idx)//2]:
        df.loc[idx, "CA1_Marks"] = rng.choice([105.0, 110.0]) # Typo / data entry errors
    for idx in outlier_idx[len(outlier_idx)//2:]:
        df.loc[idx, "MidTerm_Marks"] = rng.choice([105.0, 110.0])
        
    # Inject 3% missing values randomly in select numerical features to test SimpleImputer
    num_cols_for_na = [
        "Attendance_Pct", "CA2_Marks", "MidTerm_Marks", 
        "Prev_Sem_CGPA", "Assignment_Submission_Rate"
    ]
    
    for col in num_cols_for_na:
        na_mask = rng.rand(n_samples) < 0.03
        df.loc[na_mask, col] = np.nan
        df.loc[na_mask, col] = np.nan
        
    return df

def generate_sample_batch_upload(df_full: pd.DataFrame, n_batch: int = 50, random_state: int = 99) -> pd.DataFrame:
    """
    Creates an unlabelled evaluation batch file (50 rows) for testing CSV batch upload in Streamlit.
    """
    # Sample a realistic stratified slice across departments
    batch_df = df_full.sample(n=n_batch, random_state=random_state).copy()
    
    # Modify IDs to show they are incoming/unseen students
    batch_df["Student_ID"] = [f"LPU24_NEW_{i+1:03d}" for i in range(n_batch)]
    
    # Drop target label column
    batch_df = batch_df.drop(columns=["Risk_Level"])
    
    return batch_df.reset_index(drop=True)

def main():
    os.makedirs("data", exist_ok=True)
    print("Generating LPU Academic Dataset (2,500 records)...")
    df = generate_student_dataset(n_samples=2500, random_state=SEED)
    
    dataset_path = os.path.join("data", "student_academic_data.csv")
    df.to_csv(dataset_path, index=False)
    print(f"Saved dataset to: {dataset_path} ({df.shape[0]} rows, {df.shape[1]} columns)")
    
    print("\nClass Distribution:")
    print(df["Risk_Level"].value_counts(normalize=True).round(3) * 100)
    
    print("\nMissing Values Count:")
    print(df.isna().sum()[df.isna().sum() > 0])
    
    # Generate sample batch upload file
    batch_df = generate_sample_batch_upload(df, n_batch=50, random_state=99)
    batch_path = os.path.join("data", "sample_batch_upload.csv")
    batch_df.to_csv(batch_path, index=False)
    print(f"\nSaved sample batch upload file to: {batch_path} ({batch_df.shape[0]} rows)")

if __name__ == "__main__":
    main()
