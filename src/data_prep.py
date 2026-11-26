"""
LPU Student Academic Early-Warning & Intervention System
Module: Multi-School Data Acquisition & Synthetic Generation
File: src/data_prep.py

Generates realistic university academic records across 5 distinct LPU Schools with distinct risk rates:
1. School of Computer Science & Engineering (CSE) - 1,840 students (261 High Risk ~ 14.2%)
2. School of Mechanical Engineering (MEC) - 680 students (155 High Risk ~ 22.8%)
3. Mittal School of Business (MSB) - 920 students (107 High Risk ~ 11.6%)
4. School of Law (LAW) - 540 students (94 High Risk ~ 17.4%)
5. School of Agriculture & Bio-Sciences (AGR) - 480 students (99 High Risk ~ 20.6%)
Total Master University Census: 4,460 students (716 High Risk ~ 16.1%).

Domain Validation Rules:
- Joining year derived from Semester (Sem 1-2: 2024, Sem 3-4: 2023, Sem 5-6: 2022, Sem 7-8: 2021).
- Authentic 8-digit student registration numbers: e.g. 12404710, 12404001, 12420001, 12304002 (fixed 8-digit length across university).
- Semester 1 freshmen strictly have 0 backlogs (they have not taken prior university finals).
- Realistic faculty mentor assignment: each mentor oversees 25 to 30 assigned mentees with 5-digit Employee UIDs (e.g. UID: 24185).
- Distinct non-round risk rates per school reflecting varying discipline difficulty.
- Strictly zero unverified study hours / zero LMS login tracking.
"""

import os
import numpy as np
import pandas as pd

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh", "Ayaan", "Krishna", "Ishaan",
    "Shaurya", "Atharva", "Advik", "Pranav", "Advaith", "Aaryan", "Dhruv", "Kabir", "Ritvik", "Darsh",
    "Ananya", "Diya", "Gauri", "Kavya", "Isha", "Aditi", "Meera", "Riya", "Sneha", "Tanvi",
    "Pooja", "Shreya", "Neha", "Simran", "Deepika", "Kritika", "Anushka", "Nandini", "Avani", "Tara",
    "Rohan", "Manish", "Kartik", "Karan", "Siddharth", "Varun", "Abhinav", "Mayank", "Nikhil", "Gaurav"
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Gupta", "Singh", "Kumar", "Mehta", "Rao", "Joshi", "Nair",
    "Reddy", "Chauhan", "Malhotra", "Kapoor", "Bhatia", "Saxena", "Chopra", "Das", "Sen", "Iyer",
    "Gill", "Dhillon", "Kaur", "Trivedi", "Mishra", "Sethi", "Deshmukh", "Bhatt", "Chahal", "Sandhu"
]

SCHOOL_CONFIG = {
    "School of Computer Science & Engineering": {
        "prefix": "CSE",
        "school_code_digit": 1,
        "id_base": 4001,
        "n_samples": 1840,
        "target_high": 261,     # 14.18%
        "target_med": 396,      # 21.52%
        "seed": 42,
        "departments": [
            "B.Tech CSE (Core)",
            "B.Tech CSE (AI & Data Science)",
            "B.Tech CSE (Cyber Security)",
            "B.Tech CSE (Cloud Computing)",
            "B.Tech CSE (Full Stack Web)"
        ],
        "dept_weights": [0.35, 0.25, 0.15, 0.15, 0.10],
        "sections": ["K23-CS-A", "K23-CS-B", "K23-CS-C", "K23-CS-D", "K23-CS-E", "K23-CS-F", "K23-CS-G", "K23-CS-H", "K23-CS-I", "K23-CS-J"],
        "mentor_pool_size": 68  # 27 students per mentor (between 25-30)
    },
    "School of Mechanical Engineering": {
        "prefix": "MEC",
        "school_code_digit": 2,
        "id_base": 20001,
        "n_samples": 680,
        "target_high": 155,     # 22.79%
        "target_med": 150,      # 22.06%
        "seed": 43,
        "departments": [
            "B.Tech Mechanical Engineering",
            "B.Tech Mechatronics & Automation",
            "B.Tech Automobile Engineering",
            "B.Tech Robotics & AI"
        ],
        "dept_weights": [0.40, 0.25, 0.20, 0.15],
        "sections": ["K23-ME-A", "K23-ME-B", "K23-ME-C", "K23-ME-D"],
        "mentor_pool_size": 26   # 26.15 students per mentor (between 25-30)
    },
    "Mittal School of Business": {
        "prefix": "MSB",
        "school_code_digit": 3,
        "id_base": 30001,
        "n_samples": 920,
        "target_high": 107,     # 11.63%
        "target_med": 182,      # 19.78%
        "seed": 44,
        "departments": [
            "BBA (General Management)",
            "BBA (Financial Analytics)",
            "BBA (Digital Marketing)",
            "MBA (Integrated Business)"
        ],
        "dept_weights": [0.35, 0.25, 0.20, 0.20],
        "sections": ["K23-MB-A", "K23-MB-B", "K23-MB-C", "K23-MB-D", "K23-MB-E"],
        "mentor_pool_size": 34   # 27.05 students per mentor (between 25-30)
    },
    "School of Law": {
        "prefix": "LAW",
        "school_code_digit": 4,
        "id_base": 40001,
        "n_samples": 540,
        "target_high": 94,      # 17.41%
        "target_med": 122,      # 22.59%
        "seed": 45,
        "departments": [
            "B.A. LL.B. (Hons)",
            "BBA LL.B. (Hons)",
            "LL.B. (Core Law)",
            "LL.M. (Corporate Law)"
        ],
        "dept_weights": [0.40, 0.30, 0.20, 0.10],
        "sections": ["K23-LW-A", "K23-LW-B", "K23-LW-C"],
        "mentor_pool_size": 20   # 27.0 students per mentor (between 25-30)
    },
    "School of Agriculture & Bio-Sciences": {
        "prefix": "AGR",
        "school_code_digit": 5,
        "id_base": 50001,
        "n_samples": 480,
        "target_high": 99,      # 20.62%
        "target_med": 105,      # 21.88%
        "seed": 46,
        "departments": [
            "B.Sc. (Hons) Agriculture",
            "B.Tech Biotechnology",
            "B.Sc. Food Technology",
            "B.Sc. Microbiology"
        ],
        "dept_weights": [0.45, 0.25, 0.15, 0.15],
        "sections": ["K23-AG-A", "K23-AG-B", "K23-AG-C"],
        "mentor_pool_size": 18   # 26.66 students per mentor (between 25-30)
    }
}

def generate_mentors_for_school(prefix: str, count: int, seed: int):
    """Generates a realistic list of university faculty mentors with 5-digit Employee UIDs."""
    rng = np.random.RandomState(seed + 100)
    titles = ["Dr.", "Prof.", "Assoc. Prof."]
    mentors = []
    # Distinct 5-digit faculty UIDs (e.g. 21450, 18520, etc.)
    base_uid = 18000 + (seed * 113) % 9000
    for i in range(count):
        title = rng.choice(titles, p=[0.55, 0.30, 0.15])
        f = rng.choice(FIRST_NAMES)
        l = rng.choice(LAST_NAMES)
        emp_uid = base_uid + (i * 137) % 8999
        mentors.append(f"{title} {f} {l} (UID: {emp_uid:05d})")
    return mentors

def generate_single_school(school_name: str, config: dict) -> pd.DataFrame:
    """Generates an authentic school dataset respecting domain constraints."""
    n = config["n_samples"]
    rng = np.random.RandomState(config["seed"])
    prefix = config["prefix"]

    first_names = rng.choice(FIRST_NAMES, size=n)
    last_names = rng.choice(LAST_NAMES, size=n)
    student_names = [f"{f} {l}" for f, l in zip(first_names, last_names)]

    # Departments (specializations) and Semesters
    departments = rng.choice(config["departments"], size=n, p=config["dept_weights"])
    semesters = rng.choice(list(range(1, 9)), size=n, p=[0.16, 0.18, 0.18, 0.16, 0.12, 0.10, 0.06, 0.04])
    sections = rng.choice(config["sections"], size=n)

    # Calculate joining year based on current semester:
    # Sem 1 & 2 -> Joined 2024 ('24')
    # Sem 3 & 4 -> Joined 2023 ('23')
    # Sem 5 & 6 -> Joined 2022 ('22')
    # Sem 7 & 8 -> Joined 2021 ('21')
    # Real-world authentic university 8-digit registration number:
    # 1 (prefix) + 2-digit year ('24') + 5-digit roll number (e.g. 12404710) - Strictly 8 digits for all students!
    id_base = config.get("id_base", 10001)
    student_ids = []
    for i in range(n):
        s = semesters[i]
        if s in [1, 2]:
            yr = "24"
        elif s in [3, 4]:
            yr = "23"
        elif s in [5, 6]:
            yr = "22"
        else:
            yr = "21"
        roll_num = id_base + i
        student_ids.append(f"1{yr}{roll_num:05d}")

    # Faculty mentor allocation (Strictly 25 to 30 students per mentor across risk levels)
    mentors_list = generate_mentors_for_school(prefix, config["mentor_pool_size"], config["seed"])
    mentor_assignments = [mentors_list[i % len(mentors_list)] for i in range(n)]

    # Latent academic capability factor (Gaussian standard normal)
    latent_ability = rng.normal(loc=0.0, scale=1.0, size=n)
    latent_diligence = 0.65 * latent_ability + rng.normal(loc=0.0, scale=0.6, size=n)

    # Attendance Percentage (Base ~ 78%, capped at 100, sensitive to diligence)
    raw_attendance = 76.0 + 11.5 * latent_diligence + rng.normal(0, 5.0, size=n)
    attendance_pct = np.clip(raw_attendance, 35.0, 100.0).round(1)

    # Previous Semester CGPA (scale 0-10, realistic mean ~ 7.15)
    # For Semester 1, this represents standardized qualifying baseline (10+2 / entrance normalized)
    raw_cgpa = 7.15 + 1.20 * latent_ability + 0.30 * latent_diligence + rng.normal(0, 0.45, size=n)
    prev_sem_cgpa = np.clip(raw_cgpa, 3.8, 9.95).round(2)

    # Continuous Assessments (CA1, CA2, CA3 on 0-100 scale)
    ca_base = 62.0 + 14.0 * latent_ability + 8.0 * latent_diligence + rng.normal(0, 7.0, size=n)
    ca1 = np.clip(ca_base + rng.normal(0, 6.0, size=n), 12.0, 100.0).round(1)

    ca2_shift = (attendance_pct - 75.0) * 0.15
    ca2 = np.clip(ca_base + ca2_shift + rng.normal(1.0, 5.5, size=n), 10.0, 100.0).round(1)

    fatigue_or_push = rng.choice([-6.0, 0.0, 5.0], size=n, p=[0.25, 0.50, 0.25])
    ca3 = np.clip(ca_base + ca2_shift + fatigue_or_push + rng.normal(0, 6.5, size=n), 8.0, 100.0).round(1)

    # Mid-Term Marks (0-100 scale)
    midterm_base = 0.45 * ca1 + 0.45 * ca2 + 3.0 * (prev_sem_cgpa - 7.0) + rng.normal(0, 6.0, size=n)
    midterm_marks = np.clip(midterm_base, 10.0, 100.0).round(1)

    # Assignment Submission Rate (% 0-100)
    raw_assignments = 75.0 + 15.0 * latent_diligence + rng.normal(0, 6.0, size=n)
    assignment_rate = np.clip(raw_assignments, 15.0, 100.0).round(1)

    # Participation Score (1.0 to 10.0)
    raw_part = 6.0 + 1.6 * latent_diligence + rng.normal(0, 1.2, size=n)
    participation = np.clip(raw_part, 1.0, 10.0).round(1)

    # ACTIVE BACKLOGS LOGIC (Domain-Guaranteed):
    # - Semester 1 freshmen have strictly 0 backlogs (they have not taken any prior university finals!)
    # - Semester 2 students can have at most 1-2 backlogs from Sem 1.
    # - Semester >= 3 students can have 0 to 5 backlogs.
    backlogs = np.zeros(n, dtype=int)
    for i in range(n):
        if semesters[i] == 1:
            backlogs[i] = 0
        elif semesters[i] == 2:
            if rng.rand() < 0.20:
                backlogs[i] = rng.choice([1, 2], p=[0.75, 0.25])
        else:
            if rng.rand() < 0.30:
                backlogs[i] = rng.choice([1, 2, 3, 4, 5], p=[0.55, 0.25, 0.12, 0.05, 0.03])

    # Ground Truth Risk Score Synthesis
    ca_avg = (ca1 + ca2 + ca3) / 3.0
    risk_index = (
        - 0.38 * ((attendance_pct - 75.0) / 15.0)
        - 0.32 * ((midterm_marks - 50.0) / 20.0)
        - 0.28 * ((ca_avg - 55.0) / 20.0)
        - 0.24 * ((prev_sem_cgpa - 7.0) / 1.5)
        + 0.40 * (backlogs / 2.0)
        - 0.20 * ((assignment_rate - 70.0) / 20.0)
        - 0.14 * ((participation - 6.0) / 2.0)
        + rng.normal(0, 0.35, size=n)
    )

    # Order students by risk_index to assign exact, realistic distinct counts per school
    sorted_indices = np.argsort(risk_index)[::-1] # Highest risk first
    target_high = config["target_high"]
    target_med = config["target_med"]

    risk_labels = np.array(["Low"] * n, dtype=object)
    high_idx = sorted_indices[:target_high]
    med_idx = sorted_indices[target_high:target_high + target_med]
    
    risk_labels[high_idx] = "High"
    risk_labels[med_idx] = "Medium"

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Student_Name": student_names,
        "School": school_name,
        "Department": departments,
        "Semester": semesters,
        "Section": sections,
        "Mentor_Name": mentor_assignments,
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

    # Realistic outlier entry errors (0.8%)
    outlier_idx = rng.choice(n, size=max(1, int(0.008 * n)), replace=False)
    for idx in outlier_idx[:len(outlier_idx)//2]:
        df.loc[idx, "CA1_Marks"] = 105.0
    for idx in outlier_idx[len(outlier_idx)//2:]:
        df.loc[idx, "MidTerm_Marks"] = 105.0

    # Inject 3% missingness in select columns for imputer testing
    num_cols_for_na = ["Attendance_Pct", "CA2_Marks", "MidTerm_Marks", "Prev_Sem_CGPA", "Assignment_Submission_Rate"]
    for col in num_cols_for_na:
        na_mask = rng.rand(n) < 0.03
        df.loc[na_mask, col] = np.nan

    return df

def generate_sample_batch_upload(df_full: pd.DataFrame, n_batch: int = 50, random_state: int = 99) -> pd.DataFrame:
    """Creates an unlabelled evaluation batch file (50 rows) for testing CSV batch upload in Streamlit."""
    batch_df = df_full.sample(n=n_batch, random_state=random_state).copy()
    batch_df["Student_ID"] = [f"1249{1001 + i:04d}" for i in range(n_batch)]
    batch_df = batch_df.drop(columns=["Risk_Level"])
    return batch_df.reset_index(drop=True)

def main():
    os.makedirs("data", exist_ok=True)
    print("Generating Multi-School Datasets for Lovely Professional University...")
    all_dfs = []

    file_map = {
        "School of Computer Science & Engineering": "school_cse.csv",
        "School of Mechanical Engineering": "school_mechanical.csv",
        "Mittal School of Business": "school_business.csv",
        "School of Law": "school_law.csv",
        "School of Agriculture & Bio-Sciences": "school_agriculture.csv"
    }

    for school_name, config in SCHOOL_CONFIG.items():
        df_school = generate_single_school(school_name, config)
        filename = file_map[school_name]
        filepath = os.path.join("data", filename)
        df_school.to_csv(filepath, index=False)
        all_dfs.append(df_school)

        high_n = (df_school["Risk_Level"] == "High").sum()
        total_n = len(df_school)
        high_pct = high_n / total_n * 100
        sem1_backlogs = df_school.loc[df_school["Semester"] == 1, "Active_Backlogs"].max()

        print(f"  ✓ {school_name}: {total_n} students saved to {filepath}")
        print(f"     High Risk: {high_n}/{total_n} ({high_pct:.1f}%), Sem 1 Max Backlogs: {sem1_backlogs}")

    # Combine into unified master dataset (4,460 students)
    master_df = pd.concat(all_dfs, ignore_index=True)
    master_path = os.path.join("data", "all_schools_combined.csv")
    master_df.to_csv(master_path, index=False)

    # Also save as student_academic_data.csv for backward compatibility
    legacy_path = os.path.join("data", "student_academic_data.csv")
    master_df.to_csv(legacy_path, index=False)

    total_master = len(master_df)
    high_master = (master_df["Risk_Level"] == "High").sum()
    print(f"\n✓ Master University Census: {total_master} students saved to {master_path}")
    print(f"  Total High Risk: {high_master}/{total_master} ({high_master/total_master*100:.1f}%)")

    # Sample section batch upload file
    batch_df = generate_sample_batch_upload(master_df, n_batch=50, random_state=99)
    batch_path = os.path.join("data", "sample_batch_upload.csv")
    batch_df.to_csv(batch_path, index=False)
    print(f"✓ Sample Section Batch: 50 students saved to {batch_path}")

if __name__ == "__main__":
    main()
