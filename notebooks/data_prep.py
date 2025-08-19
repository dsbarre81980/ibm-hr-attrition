from pathlib import Path
import pandas as pd

DROP_COLS_DEFAULT = ["EmployeeCount", "StandardHours", "Over18", "EmployeeNumber"]

# Carga un archivo CSV y verifica que exista
def load_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {path}")
    return pd.read_csv(path)

# Limpia espacios en blanco de todas las columnas de texto
def clean_text_cols(df: pd.DataFrame) -> pd.DataFrame:
    dfc = df.copy()
    for c in dfc.select_dtypes(include="object").columns:
        dfc[c] = dfc[c].astype(str).str.strip()
    return dfc

# Elimina columnas que no aportan información útil
def drop_irrelevant_cols(df: pd.DataFrame, cols_to_drop=None) -> pd.DataFrame:
    dfc = df.copy()
    cols = cols_to_drop if cols_to_drop is not None else DROP_COLS_DEFAULT
    cols = [c for c in cols if c in dfc.columns]
    return dfc.drop(columns=cols, errors="ignore")

# Elimina filas duplicadas del dataset
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates()

# Genera un reporte rápido de calidad de datos
def quick_quality_report(df: pd.DataFrame) -> None:
    print("=== Calidad de datos ===")
    print("Dimensiones:", df.shape)
    print("\nNulos por columna:\n", df.isna().sum())
    print("\nDuplicados:", df.duplicated().sum())
    print("\nCardinalidad (valores únicos):\n", df.nunique().sort_values())
    print("\nTipos:\n")
    print(df.dtypes)

# Guarda el DataFrame como CSV creando carpetas si es necesario
def save_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"Guardado: {path}")
