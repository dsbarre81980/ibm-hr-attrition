# features.py
import pandas as pd

def add_age_group(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    bins_age = [18, 25, 35, 45, 55, 70]
    labels_age = ["18-25", "26-35", "36-45", "46-55", "56-70"]
    dfx["AgeGroup"] = pd.cut(
        dfx["Age"], bins=bins_age, labels=labels_age, include_lowest=True, right=True
    )
    return dfx

def add_tenure_group(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    bins_tenure = [-1, 2, 5, 10, 20, 50]
    labels_tenure = ["≤2", "3-5", "6-10", "11-20", ">20"]
    dfx["TenureGroup"] = pd.cut(
        dfx["YearsAtCompany"], bins=bins_tenure, labels=labels_tenure, right=True
    )
    return dfx

def add_income_quartile(df: pd.DataFrame) -> pd.DataFrame:
    dfx = df.copy()
    dfx["IncomeQuartile"] = pd.qcut(
        dfx["MonthlyIncome"], 4, labels=["Q1 (bajo)", "Q2", "Q3", "Q4 (alto)"]
    )
    return dfx

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df1 = add_age_group(df)
    df2 = add_tenure_group(df1)
    df3 = add_income_quartile(df2)
    return df3
