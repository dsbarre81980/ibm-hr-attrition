from pathlib import Path
import pandas as pd

from data_prep import load_csv, clean_text_cols, drop_irrelevant_cols, drop_duplicates, quick_quality_report, save_csv
from features import build_features
from utils import check_non_negative, iqr_summary
from encode_ml import to_ml_dataset
from viz import hist, bar_counts, stacked_attrition_by

# === Config rutas ===
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw" / "WA_Fn-UseC_-HR-Employee-Attrition.csv"
DATA_PROCESSED_DIR = BASE_DIR / "data" / "processed"
ANALYTICS_OUT = DATA_PROCESSED_DIR / "hr_clean_analytics.csv"
ML_OUT = DATA_PROCESSED_DIR / "hr_clean_ml.csv"

# Ejecuta todo el pipeline de limpieza y preparación de datos
def main():
    # 1) Carga
    df = load_csv(DATA_RAW)
    print("Dimensiones (original):", df.shape)
    print(df.head(3), "\n")

    # 2) EDA rápida
    quick_quality_report(df)

    # 3) Limpieza
    dfc = clean_text_cols(df)
    dfc = drop_irrelevant_cols(dfc)
    dfc = drop_duplicates(dfc)

    print("\n=== Tras limpieza ===")
    print("Dimensiones:", dfc.shape)

    # 4) Chequeos de calidad y outliers (solo informe, no capamos)
    check_non_negative(
        dfc,
        ["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "TrainingTimesLastYear", "DistanceFromHome", "Age"],
    )
    iqr_summary(dfc, ["MonthlyIncome", "TotalWorkingYears", "YearsAtCompany", "Age"])

    # 5) Features derivadas
    dfa = build_features(dfc)

    # 6) Export ANALYTICS (texto legible)
    save_csv(dfa, ANALYTICS_OUT)

    # 7) Export ML (codificado)
    df_ml = to_ml_dataset(dfa)
    save_csv(df_ml, ML_OUT)

    # 8) Evidencias gráficas simples
    try:
        hist(dfa["Age"], "Distribución de Edad", "Edad")
        bar_counts(dfa["Attrition"].astype(str), "Distribución de Attrition (Yes/No)", "Attrition")
        stacked_attrition_by(dfa, "Department")
    except Exception as e:
        print("Gráficos omitidos:", e)

    print("\n✅ Pipeline completado.")

if __name__ == "__main__":
    main()
