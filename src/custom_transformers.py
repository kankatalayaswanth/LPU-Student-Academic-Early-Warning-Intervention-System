"""
LPU Student Academic Early-Warning & Intervention System
Module: Custom Transformers for Preprocessing & Feature Engineering
File: src/custom_transformers.py
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class AcademicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Engineers university-domain features:
    1. CA_Average: Mean performance across Continuous Assessments CA1, CA2, CA3.
    2. CA_Trend: Trajectory (CA3 - CA1), capturing academic burnout vs exam push.
    3. Attendance_Shortage_Flag: Binary indicator for LPU debarment threshold (< 75.0%).
    4. Engagement_Index: Blended metric of digital LMS activity (40%) and homework diligence (60%).
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            df = X.copy()
        else:
            raise TypeError("AcademicFeatureEngineer expects a pandas DataFrame input.")

        # Continuous Assessment metrics
        ca1 = df["CA1_Marks"].fillna(50.0)
        ca2 = df["CA2_Marks"].fillna(50.0)
        ca3 = df["CA3_Marks"].fillna(50.0)

        df["CA_Average"] = (ca1 + ca2 + ca3) / 3.0
        df["CA_Trend"] = ca3 - ca1

        # Attendance compliance indicator (< 75% threshold in LPU)
        att = df["Attendance_Pct"].fillna(75.0)
        df["Attendance_Shortage_Flag"] = (att < 75.0).astype(float)

        # Digital & Assignment Engagement Index (0 - 100 scale)
        lms = df["LMS_Logins_Per_Week"].fillna(14.0)
        asn = df["Assignment_Submission_Rate"].fillna(75.0)
        
        # LMS logins normalized to 35, assignments normalized to 100
        lms_norm = np.clip(lms, 0, 35) / 35.0
        asn_norm = np.clip(asn, 0, 100) / 100.0
        df["Engagement_Index"] = (lms_norm * 40.0 + asn_norm * 60.0).round(2)

        return df

class IQROutlierCapper(BaseEstimator, TransformerMixin):
    """
    Outlier handler using Interquartile Range (IQR) clipping.
    Caps values strictly using bounds fitted on training data:
    [Q1 - factor * IQR, Q3 + factor * IQR]
    """
    def __init__(self, factor=1.5):
        self.factor = factor
        self.lower_caps_ = None
        self.upper_caps_ = None

    def fit(self, X, y=None):
        X_arr = np.asarray(X, dtype=float)
        # Compute quartiles column-wise ignoring NaNs
        q25 = np.nanpercentile(X_arr, 25, axis=0)
        q75 = np.nanpercentile(X_arr, 75, axis=0)
        iqr = q75 - q25
        self.lower_caps_ = q25 - self.factor * iqr
        self.upper_caps_ = q75 + self.factor * iqr
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        if self.lower_caps_ is None or self.upper_caps_ is None:
            raise RuntimeError("IQROutlierCapper is not fitted yet.")
        return np.clip(X_arr, self.lower_caps_, self.upper_caps_)
