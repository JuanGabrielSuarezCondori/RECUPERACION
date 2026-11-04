"""
train.py
Módulo de entrenamiento para el traductor Quechua -> Español con SimpleRNN Many-to-Many.
Incluye:
- Construcción y serialización de vocabularios sin sesgo de UNK.
- DataLoader de alto rendimiento con secuencias pre-numericalizadas.
- SimpleRNN Encoder con descarte de padding via pack_padded_sequence.
- SimpleRNN Decoder condicionado en el vector de contexto en cada paso temporal.
- Función de pérdida CrossEntropyLoss ignorando el token <pad>.
- Recorte de gradientes (Gradient Clipping) para mitigar explosión de gradientes.
- Monitoreo de curvas de aprendizaje (Loss y Perplejidad).
- Guardado del mejor punto de control (Checkpoint) según pérdida de validación.
- Soporte para CLI e integración con Streamlit.
"""

import os
import sys
import json
import time
import math
import argparse
from typing import Optional, Callable, Dict, Any

# Configurar salida UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.preprocessor import (
        Vocabulary, TranslationDataset, translation_collate_fn, PAD_IDX
    )
    from src.rnn_model import build_model, Seq2SeqSimpleRNN
except ImportError:
    from preprocessor import (
        Vocabulary, TranslationDataset, translation_collate_fn, PAD_IDX
    )
    from rnn_model import build_model, Seq2SeqSimpleRNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_epoch(model: Seq2SeqSimpleRNN, iterator: DataLoader,
                optimizer: torch.optim.Optimizer, criterion: nn.Module,
                clip: float, teacher_forcing_ratio: float, device: torch.device) -> float:
    """Ejecuta una época de entrenamiento."""
    model.train()
    epoch_loss = 0.0

    for src, trg, src_lens, _, _ in iterator:
        src = src.to(device)
        trg = trg.to(device)
        src_lens = src_lens.to(device)

        optimizer.zero_grad()

        # Forward pass Many-to-Many
        output = model(src, trg, src_lengths=src_lens, teacher_forcing_ratio=teacher_forcing_ratio)

        # Dimensiones:
        # output: [batch_size, trg_len, output_dim]
        # trg:    [batch_size, trg_len]
        # Se ignora el primer token (<sos>) en el cálculo de pérdida
        output_dim = output.shape[-1]
        output = output[:, 1:].reshape(-1, output_dim)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()

        # Recorte de gradientes para evitar explosión en RNN Simple
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate_epoch(model: Seq2SeqSimpleRNN, iterator: DataLoader,
                   criterion: nn.Module, device: torch.device) -> float:
    """
    Evalúa la pérdida en el conjunto de validación calculando la probabilidad
    condicional exacta de la secuencia objetivo.
    """
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for src, trg, src_lens, _, _ in iterator:
            src = src.to(device)
            trg = trg.to(device)
            src_lens = src_lens.to(device)

            output = model(src, trg, src_lengths=src_lens, teacher_forcing_ratio=1.0)

            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            trg = trg[:, 1:].reshape(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def train_model(
    epochs: int = 20,
    batch_size: int = 32,
    emb_dim: int = 128,
    hidden_dim: int = 256,
    lr: float = 0.003,
    teacher_forcing_ratio: float = 0.6,
    min_freq: int = 1,
    clip: float = 1.0,
    dropout: float = 0.1,
    device_name: Optional[str] = None,
    progress_callback: Optional[Callable[[int, int, Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Función principal de entrenamiento.
    Permite llamadas programáticas (e.g. desde Streamlit) con callbacks de progreso.
    """
    if device_name is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    print(f"Dispositivo de entrenamiento seleccionado: {device}")

    # 1. Cargar datos
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Archivos de entrenamiento no encontrados en {DATA_DIR}. Ejecuta dataset_builder.py primero."
        )

    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)

    # 2. Construir Vocabularios
    src_vocab = Vocabulary("quechua")
    trg_vocab = Vocabulary("espanol")

    src_vocab.build_vocab(train_df["quechua"].tolist(), min_freq=min_freq)
    trg_vocab.build_vocab(train_df["espanol"].tolist(), min_freq=min_freq)

    print(f"Tamaño vocabulario Quechua (fuente): {len(src_vocab)}")
    print(f"Tamaño vocabulario Español (objetivo): {len(trg_vocab)}")

    # Guardar vocabularios
    src_vocab_path = os.path.join(MODELS_DIR, "vocab_que.json")
    trg_vocab_path = os.path.join(MODELS_DIR, "vocab_es.json")
    src_vocab.save(src_vocab_path)
    trg_vocab.save(trg_vocab_path)

    # 3. Crear Datasets y DataLoaders
    train_dataset = TranslationDataset(
        train_df["quechua"].tolist(),
        train_df["espanol"].tolist(),
        src_vocab, trg_vocab
    )
    val_dataset = TranslationDataset(
        val_df["quechua"].tolist(),
        val_df["espanol"].tolist(),
        src_vocab, trg_vocab
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        collate_fn=translation_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        collate_fn=translation_collate_fn
    )

    # 4. Construir Modelo
    model = build_model(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        emb_dim=emb_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        device=device
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    # Ignorar token <pad> en la función de pérdida
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    best_val_loss = float("inf")
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_ppl": [],
        "val_ppl": [],
        "epochs": [],
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "emb_dim": emb_dim,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "teacher_forcing_ratio": teacher_forcing_ratio,
            "min_freq": min_freq,
            "src_vocab_size": len(src_vocab),
            "trg_vocab_size": len(trg_vocab)
        }
    }

    model_save_path = os.path.join(MODELS_DIR, "simple_rnn_que_es.pt")

    print("\nIniciando entrenamiento del modelo SimpleRNN Many-to-Many...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion,
            clip=clip, teacher_forcing_ratio=teacher_forcing_ratio,
            device=device
        )
        val_loss = evaluate_epoch(model, val_loader, criterion, device=device)
        scheduler.step(val_loss)
        epoch_duration = time.time() - epoch_start

        # Perplejidad = exp(loss)
        train_ppl = math.exp(min(train_loss, 20))
        val_ppl = math.exp(min(val_loss, 20))

        history["epochs"].append(epoch)
        history["train_loss"].append(round(train_loss, 4))
        history["val_loss"].append(round(val_loss, 4))
        history["train_ppl"].append(round(train_ppl, 2))
        history["val_ppl"].append(round(val_ppl, 2))

        # Guardar mejor modelo según pérdida de validación
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "hyperparameters": history["hyperparameters"]
            }, model_save_path)
            saved_marker = " [* Mejor Modelo Guardado]"
        else:
            saved_marker = ""

        print(
            f"Época {epoch:02d}/{epochs:02d} | "
            f"Tiempo: {epoch_duration:.1f}s | "
            f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.1f}) | "
            f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.1f}){saved_marker}"
        )

        # Notificar callback de Streamlit si existe
        if progress_callback is not None:
            progress_callback(epoch, epochs, {
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_ppl": train_ppl,
                "val_ppl": val_ppl,
                "best_val_loss": best_val_loss
            })

    total_time = time.time() - start_time
    history["total_time_seconds"] = round(total_time, 2)
    print(f"\nEntrenamiento completado en {total_time:.1f} segundos.")
    print(f"Mejor pérdida de validación alcanzada: {best_val_loss:.4f}")

    # Guardar historial en JSON
    history_path = os.path.join(MODELS_DIR, "train_history.json")
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenar traductor SimpleRNN Quechua-Español")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas")
    parser.add_argument("--batch_size", type=int, default=32, help="Tamaño de batch")
    parser.add_argument("--emb_dim", type=int, default=128, help="Dimensión de embeddings")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Dimensión de estado oculto RNN")
    parser.add_argument("--lr", type=float, default=0.003, help="Tasa de aprendizaje (learning rate)")
    parser.add_argument("--teacher_forcing", type=float, default=0.6, help="Ratio de teacher forcing")
    parser.add_argument("--min_freq", type=int, default=1, help="Frecuencia mínima de palabra")

    args = parser.parse_args()
    train_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        teacher_forcing_ratio=args.teacher_forcing,
        min_freq=args.min_freq
    )
