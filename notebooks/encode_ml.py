import pandas as pd

# Convierte variables categóricas a numéricas para machine learning
def to_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df_ml = df.copy()

    # Binarias Yes/No a 1/0 si existen
    bin_map = {"Yes": 1, "No": 0}
    for b in ["Attrition", "OverTime"]:
        if b in df_ml.columns and df_ml[b].dtype == object:
            df_ml[b] = df_ml[b].map(bin_map)

    # One-hot para categóricas restantes de texto
    cat_cols = df_ml.select_dtypes(include="object").columns.tolist()
    df_ml = pd.get_dummies(df_ml, columns=cat_cols, drop_first=True)
    return df_ml
