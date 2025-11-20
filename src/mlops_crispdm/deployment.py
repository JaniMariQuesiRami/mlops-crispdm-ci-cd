import torch
from mlops_crispdm.modeling import text_to_sequence, lstm_model, word2idx, LSTM_CONFIG, device

# Frases de ejemplo para probar
example_phrases = [
    "Felicitaciones! Has ganado un premio de 1000 euros. Reclama en link.ly/premio",
    "Hola, ¿cómo estás? Quería saber si tenías tiempo para tomar un café mañana.",
    "URGENTE: Tu cuenta bancaria ha sido comprometida. Confirma tus datos en www.banco.com/seguridad",
    "No olvides nuestra reunión el viernes a las 10 AM en la sala principal.",
    "Tu número ha sido seleccionado para un sorteo exclusivo. Llama al 902123456 ahora mismo.",
    "Me gustaria que me dieras feedback sobre el proyecto que te envié ayer."
]

def predict_with_lstm(text, model, word2idx, max_length, device):
    model.eval()
    sequence = text_to_sequence(text, word2idx, max_length)
    input_tensor = torch.LongTensor(sequence).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = (output > 0.5).item()
    return "Spam" if prediction == 1 else "No Spam", output.item()

def run():
  content = ("""\
========================================================================
6. Deployment — Despliegue de Modelos de Detección de Spam en SMS
========================================================================
  """)
  print(content)

  print("\n" + "="*80)
  print("PREDICCIONES PARA NUEVAS FRASES")
  print("="*80)

  for i, phrase in enumerate(example_phrases):
      print(f"\n--- Frase {i+1} ---")
      print(f"Texto: {phrase}")

      # Predicción LSTM
      lstm_pred_label, lstm_pred_proba = predict_with_lstm(phrase, lstm_model, word2idx, LSTM_CONFIG['max_length'], device)
      print(f"LSTM -> Clasificación: {lstm_pred_label} (Prob: {lstm_pred_proba:.4f})")

