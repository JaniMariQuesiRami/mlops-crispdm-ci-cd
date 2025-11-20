import pandas as pd

# ======================================================
# 2 Carga del dataset traducido
# ======================================================
df = pd.read_csv("mlops_crispdm/spam_dataset_es.csv")

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

  
  

