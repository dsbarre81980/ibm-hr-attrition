import pandas as pd

# Verifica si hay valores negativos en columnas numéricas
def check_non_negative(df: pd.DataFrame, cols) -> None:
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            if (df[col] < 0).any():
                print(f"{col} tiene valores negativos. Revisa valores o fuentes.")

# Calcula máscara de outliers usando método IQR
def iqr_outliers_mask(s: pd.Series):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (s < low) | (s > high), (low, high)

# Muestra resumen de outliers por columna usando IQR
def iqr_summary(df: pd.DataFrame, cols) -> None:
    for col in cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            mask, (low, high) = iqr_outliers_mask(df[col])
            print(f"{col}: outliers={mask.sum()} | rango~({low:.1f}, {high:.1f})")
