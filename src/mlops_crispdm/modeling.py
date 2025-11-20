import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from collections import Counter
import re

from mlops_crispdm.data_preparation import X_train, X_test, limpiar_texto, y_train, y_test, y_pred_svm, df, svm, vectorizer

# Usar GPU si está disponible
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mlp = MLPClassifier(hidden_layer_sizes=(128,), activation="relu", max_iter=20, random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)

df = df.rename(columns={'Target': 'label', 'Text_es': 'Texto'})
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# División de datos
X = df['Texto'].values
y = df['label'].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test2, y_val, y_test2 = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# ==================== HIPERPARÁMETROS LSTM ====================
LSTM_CONFIG = {
    # Tokenización
    'vocab_size': 10000,        # Tamaño del vocabulario
    'max_length': 100,          # Longitud máxima de secuencia

    # Arquitectura
    'embedding_dim': 128,       # Dimensión de embeddings
    'hidden_dim': 256,          # Dimensión de capa oculta LSTM
    'num_layers': 2,            # Número de capas LSTM
    'bidirectional': True,      # LSTM bidireccional
    'dropout': 0.3,             # Dropout para regularización

    # Entrenamiento
    'batch_size': 64,           # Tamaño de batch
    'learning_rate': 0.001,     # Tasa de aprendizaje
    'epochs': 1,               # Número de épocas
    'weight_decay': 1e-5,       # L2 regularization
}

def tokenize(text):
    """Tokenización básica"""
    return re.findall(r'\b\w+\b', text.lower())

# Construir vocabulario
all_words = []
for text in X_train:
    all_words.extend(tokenize(text))

word_counts = Counter(all_words)
most_common = word_counts.most_common(LSTM_CONFIG['vocab_size'] - 2)  # -2 para PAD y UNK

# Crear diccionarios
word2idx = {'<PAD>': 0, '<UNK>': 1}
for idx, (word, _) in enumerate(most_common, start=2):
    word2idx[word] = idx

idx2word = {idx: word for word, idx in word2idx.items()}

def text_to_sequence(text, word2idx, max_length):
    """Convierte texto a secuencia de índices"""
    tokens = tokenize(text)
    sequence = [word2idx.get(word, word2idx['<UNK>']) for word in tokens]

    # Padding o truncado
    if len(sequence) < max_length:
        sequence = sequence + [word2idx['<PAD>']] * (max_length - len(sequence))
    else:
        sequence = sequence[:max_length]

    return sequence

# Convertir datos
X_train_seq = np.array([text_to_sequence(text, word2idx, LSTM_CONFIG['max_length']) for text in X_train])
X_val_seq = np.array([text_to_sequence(text, word2idx, LSTM_CONFIG['max_length']) for text in X_val])
X_test_seq = np.array([text_to_sequence(text, word2idx, LSTM_CONFIG['max_length']) for text in X_test2])

# Dataset personalizado
class SpamDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = torch.LongTensor(sequences)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

# Crear dataloaders
train_dataset = SpamDataset(X_train_seq, y_train)
val_dataset = SpamDataset(X_val_seq, y_val)
test_dataset = SpamDataset(X_test_seq, y_test2)

train_loader = DataLoader(train_dataset, batch_size=LSTM_CONFIG['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=LSTM_CONFIG['batch_size'])
test_loader = DataLoader(test_dataset, batch_size=LSTM_CONFIG['batch_size'])

# Modelo LSTM
class LSTMClassifier(nn.Module):
    def __init__(self, config):
        super(LSTMClassifier, self).__init__()

        self.embedding = nn.Embedding(
            config['vocab_size'],
            config['embedding_dim'],
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            config['embedding_dim'],
            config['hidden_dim'],
            config['num_layers'],
            batch_first=True,
            bidirectional=config['bidirectional'],
            dropout=config['dropout'] if config['num_layers'] > 1 else 0
        )

        self.dropout = nn.Dropout(config['dropout'])

        # Ajustar según bidireccionalidad
        lstm_output_dim = config['hidden_dim'] * 2 if config['bidirectional'] else config['hidden_dim']

        self.fc = nn.Linear(lstm_output_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Usar último estado oculto
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        else:
            hidden = hidden[-1,:,:]

        hidden = self.dropout(hidden)
        output = self.fc(hidden)
        output = self.sigmoid(output)

        return output.squeeze()

# Instanciar modelo
lstm_model = LSTMClassifier(LSTM_CONFIG).to(device)

# Entrenamiento LSTM
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(
    lstm_model.parameters(),
    lr=LSTM_CONFIG['learning_rate'],
    weight_decay=LSTM_CONFIG['weight_decay']
)

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for sequences, labels in tqdm(loader, desc="Entrenando"):
        sequences, labels = sequences.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / len(loader), correct / total

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for sequences, labels in loader:
            sequences, labels = sequences.to(device), labels.to(device)

            outputs = model(sequences)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader), correct / total

# Evaluación LSTM en test set
lstm_model.eval()
y_pred_lstm = []
y_pred_lstm_proba = []

with torch.no_grad():
    for sequences, labels in test_loader:
        sequences = sequences.to(device)
        outputs = lstm_model(sequences)
        predicted = (outputs > 0.5).float().cpu().numpy()
        y_pred_lstm.extend(predicted)
        y_pred_lstm_proba.extend(outputs.cpu().numpy())

y_pred_lstm = np.array(y_pred_lstm)
y_pred_lstm_proba = np.array(y_pred_lstm_proba)


# 1. Limpiar X_test (el raw text utilizado por LSTM y BERT)
X_test_cleaned = [limpiar_texto(text) for text in X_test2]

# 2. Vectorizar este X_test limpio usando el vectorizer ya entrenado
X_test_vec_consistent = vectorizer.transform(X_test_cleaned)

# 3. Generar nuevas predicciones para SVM y FFNN
y_pred_svm_consistent = svm.predict(X_test_vec_consistent)
y_pred_mlp_consistent = mlp.predict(X_test_vec_consistent)





def run():
  content = ("""\
========================================================================
4. Modeling — Entrenamiento de Modelos para Detección de Spam en SMS
========================================================================
  """)
  print(content)

  # ======================================================
  # 6 Modelo baseline 2: Red neuronal simple (FFNN)
  # ======================================================
  
  print("=== Resultados FFNN ===")
  print("Accuracy:", accuracy_score(y_test, y_pred_mlp))
  print(classification_report(y_test, y_pred_mlp, target_names=["No Spam", "Spam"]))

  # ConfusionMatrixDisplay.from_predictions(y_test, y_pred_mlp, display_labels=["No Spam","Spam"])
  # plt.title("Matriz de Confusión - FFNN")
  # plt.show()

  # ======================================================
  # 7 Conclusiones
  # ======================================================
  print("Resumen de desempeño:")
  print(f"SVM Accuracy:  {accuracy_score(y_test, y_pred_svm):.3f}")
  print(f"FFNN Accuracy: {accuracy_score(y_test, y_pred_mlp):.3f}")

  # Verificar estructura
  print(f"Forma del dataset: {df.shape}")
  print(f"\nPrimeras filas:")
  print(df.head())

  # Distribución de clases
  print(f"\nDistribución de clases:")
  print(df['label'].value_counts())
  print(f"\nPorcentaje de spam: {df['label'].mean()*100:.2f}%")

 
  print(f"Datos de entrenamiento: {len(X_train)}")
  print(f"Datos de validación: {len(X_val)}")
  print(f"Datos de prueba: {len(X_test2)}")

  print("Configuración del modelo LSTM:")
  for key, value in LSTM_CONFIG.items():
    print(f"  {key}: {value}")

  print(f"Tamaño del vocabulario: {len(word2idx)}")
  print(f"Palabras más comunes: {list(word2idx.keys())[2:12]}")

  print(f"Forma de X_train_seq: {X_train_seq.shape}")


  print(f"Usando dispositivo: {device}")

  print(f"\nModelo LSTM creado:")
  print(lstm_model)
  print(f"\nTotal de parámetros: {sum(p.numel() for p in lstm_model.parameters()):,}")

  # Entrenamiento
  print("\n" + "="*60)
  print("ENTRENANDO MODELO LSTM")
  print("="*60)

  train_losses, val_losses = [], []
  train_accs, val_accs = [], []

  for epoch in range(LSTM_CONFIG['epochs']):
    train_loss, train_acc = train_epoch(lstm_model, train_loader, criterion, optimizer, device)
    val_loss, val_acc = evaluate(lstm_model, val_loader, criterion, device)

    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accs.append(train_acc)
    val_accs.append(val_acc)

    print(f"Época {epoch+1}/{LSTM_CONFIG['epochs']}")
    print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print()



  print("\n" + "="*60)
  print("RESULTADOS LSTM EN TEST SET")
  print("="*60)
  print(f"Accuracy:  {accuracy_score(y_test2, y_pred_lstm):.4f}")
  print(f"Precision: {precision_score(y_test2, y_pred_lstm):.4f}")
  print(f"Recall:    {recall_score(y_test2, y_pred_lstm):.4f}")
  print(f"F1-Score:  {f1_score(y_test2, y_pred_lstm):.4f}")
  print("\nReporte de Clasificación:")
  print(classification_report(y_test2, y_pred_lstm, target_names=['No Spam', 'Spam']))

  print("Predicciones para SVM (tamaño):", y_pred_svm_consistent.shape)
  print("Predicciones para FFNN (tamaño):", y_pred_mlp_consistent.shape)
  print("y_test (tamaño):", y_test2.shape)

