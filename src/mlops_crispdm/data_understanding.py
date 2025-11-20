import pandas as pd
from pathlib import Path
# ======================================================
# 2 Carga del dataset traducido
# ======================================================
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "spam_dataset_es.csv"

df = pd.read_csv(CSV_PATH)

# # usamos solo la etiqueta y el texto traducido
df = df[["Target", "Text_es"]].dropna().rename(columns={"Text_es": "Texto"})
df.head()

def run():
  content = ("""\
========================================================================
2. Data Understanding — Análisis del Dataset de SMS Spam
========================================================================
  """)
  print(content)

  print("Tamaño del dataset:", df.shape)
  print(df["Target"].value_counts())

  
  

