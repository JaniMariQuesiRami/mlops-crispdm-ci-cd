import re
import unicodedata
import nltk
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE

from mlops_crispdm.data_understanding import df

spanish_stopwords = set(stopwords.words("spanish"))

def limpiar_texto(text):

    text = str(text)

    # 1 Corregir encoding
    text = text.encode('latin-1', errors='ignore').decode('utf-8', errors='ignore')

    # 2 Normalización unicode
    text = unicodedata.normalize("NFKC", text)

    # 3 Reemplazar URLs por <url>
    text = re.sub(r"http\S+|www\.\S+", " <url> ", text)

    # 4 Mantener letras, números y símbolos útiles
    text = re.sub(r"[^0-9a-zA-ZáéíóúÁÉÍÓÚñÑüÜ¿¡!?,.%$:/\-+\s]", "", text)

    # 5 Minúsculas
    text = text.lower()

    # 6 Tokenizar
    tokens = word_tokenize(text, language="spanish")

    # 7 OPCIONAL: quitar stopwords
    tokens = [t for t in tokens if t not in spanish_stopwords]

    # 8 Unir
    return " ".join(tokens)

# df = df[['Target', 'Text_es']].dropna() # Asegurarse de mantener las dos columnas y eliminar filas con NaNs
# df = df.rename(columns={'Target': 'label', 'Text_es': 'Texto'})

# Mapear la columna 'label' a binario
# df['label'] = df['label'].map({'ham': 0, 'spam': 1})

df["texto_limpio"] = df["Texto"].apply(limpiar_texto)

# ======================================================
# 4 Vectorización con TF-IDF
# ======================================================
X = df["texto_limpio"]
y = df["Target"].map({"ham": 0, "spam": 1})

vectorizer = TfidfVectorizer(max_features=5000)
X_vec = vectorizer.fit_transform(X)

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_vec, y, test_size=0.2, random_state=42, stratify=y
)

# Balanceo de clases (spam < ham)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# ======================================================
# 5 Modelo baseline 1: SVM lineal
# ======================================================
svm = LinearSVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)

def run():
  content = ("""\
========================================================================
3. Data Preparation — Preprocesamiento del Dataset de SMS Spam
========================================================================
  """)
  print(content)
  print("Texto limpiado de ejemplo:")
  print(df.head(10))
  

  print("=== Resultados SVM ===")
  print("Accuracy:", accuracy_score(y_test, y_pred_svm))
  print(classification_report(y_test, y_pred_svm, target_names=["No Spam", "Spam"]))

  # ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svm, display_labels=["No Spam","Spam"])
  # plt.title("Matriz de Confusión - SVM")
  # plt.show()

