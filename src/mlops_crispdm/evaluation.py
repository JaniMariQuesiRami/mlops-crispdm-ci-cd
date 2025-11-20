import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from mlops_crispdm.modeling import X_test2, y_test2, y_pred_svm_consistent, y_pred_mlp_consistent, y_pred_lstm, y_pred_lstm_proba

# Tabla comparativa actualizada con predicciones consistentes
results = pd.DataFrame({
    'Modelo': ['SVM (Baseline)', 'FFNN (Baseline)', 'LSTM'],
    'Accuracy': [
        accuracy_score(y_test2, y_pred_svm_consistent),
        accuracy_score(y_test2, y_pred_mlp_consistent),
        accuracy_score(y_test2, y_pred_lstm)
    ],
    'Precision': [
        precision_score(y_test2, y_pred_svm_consistent),
        precision_score(y_test2, y_pred_mlp_consistent),
        precision_score(y_test2, y_pred_lstm)
    ],
    'Recall': [
        recall_score(y_test2, y_pred_svm_consistent),
        recall_score(y_test2, y_pred_mlp_consistent),
        recall_score(y_test2, y_pred_lstm)
    ],
    'F1-Score': [
        f1_score(y_test2, y_pred_svm_consistent),
        f1_score(y_test2, y_pred_mlp_consistent),
        f1_score(y_test2, y_pred_lstm)
    ]
})


def run():
  content = ("""\
========================================================================
5. Evaluation — Evaluación de Modelos de Clasificación de Spam
========================================================================
  """)
  print(content)

  print("\n" + "="*80)
  print("COMPARACIÓN DE TODOS LOS MODELOS (con test set unificado)")
  print("="*80)
  print(results.to_string(index=False))
  print("="*80)

  # Análisis de ejemplos donde los modelos fallan
  print("\n" + "="*80)
  print("ANÁLISIS DE ERRORES")
  print("="*80)

  # Falsos Positivos (clasificados como spam siendo no-spam)
  fp_lstm = (y_pred_lstm == 1) & (y_test2 == 0)

  # Falsos Negativos (clasificados como no-spam siendo spam)
  fn_lstm = (y_pred_lstm == 0) & (y_test2 == 1)

  print(f"\nLSTM:")
  print(f"  Falsos Positivos: {fp_lstm.sum()}")
  print(f"  Falsos Negativos: {fn_lstm.sum()}")

  # Mostrar ejemplos de errores
  print("\n" + "="*80)
  print("EJEMPLOS DE FALSOS POSITIVOS (LSTM)")
  print("="*80)
  fp_indices = np.where(fp_lstm)[0][:5]  # Primeros 5
  for i, idx in enumerate(fp_indices, 1):
      print(f"\n{i}. Texto: {X_test2[idx][:200]}...")
      print(f"   Probabilidad: {y_pred_lstm_proba[idx]:.4f}")

  print("\n" + "="*80)
  print("EJEMPLOS DE FALSOS NEGATIVOS (LSTM)")
  print("="*80)
  fn_indices = np.where(fn_lstm)[0][:5]  # Primeros 5
  for i, idx in enumerate(fn_indices, 1):
      print(f"\n{i}. Texto: {X_test2[idx][:200]}...")
      print(f"   Probabilidad: {y_pred_lstm_proba[idx]:.4f}")