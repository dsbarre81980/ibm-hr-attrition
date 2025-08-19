import matplotlib.pyplot as plt
import pandas as pd

# Crea un histograma de una serie numérica
def hist(series: pd.Series, title: str, xlabel: str):
    series.plot(kind="hist", bins=10)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frecuencia")
    plt.show()

# Crea gráfico de barras con conteos de valores únicos
def bar_counts(series: pd.Series, title: str, xlabel: str):
    series.value_counts().plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Cantidad")
    plt.show()

# Crea gráfico de barras apiladas de attrition por categoría
def stacked_attrition_by(df: pd.DataFrame, by_col: str):
    pivot = df.groupby([by_col, "Attrition"]).size().unstack(fill_value=0)
    pivot.plot(kind="bar", stacked=True)
    plt.title(f"Attrition por {by_col}")
    plt.xlabel(by_col)
    plt.ylabel("Número de empleados")
    plt.tight_layout()
    plt.show()
