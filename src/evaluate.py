"""
evaluate.py
Módulo de evaluación para el modelo Seq2Seq SimpleRNN Quechua -> Español.
Calcula métricas estándar de traducción automática:
- BLEU-1, BLEU-2, BLEU-3, BLEU-4 y Cumulative BLEU (con SmoothingFunction de NLTK).
- Generación de predicciones sobre el conjunto de prueba (test.csv).
- Exportación de reporte detallado en JSON para visualización en Streamlit.
"""

import os
import sys
import json
from typing import List, Dict, Any, Tuple, Optional

# Configurar salida UTF-8
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.preprocessor import Vocabulary, tokenize, detokenize
    from src.rnn_model import build_model, Seq2SeqSimpleRNN
except ImportError:
    from preprocessor import Vocabulary, tokenize, detokenize
    from rnn_model import build_model, Seq2SeqSimpleRNN

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

def load_trained_model(device: Optional[torch.device] = None) -> Tuple[Seq2SeqSimpleRNN, Vocabulary, Vocabulary, Dict[str, Any]]:
    """Carga los vocabularios y el modelo entrenado desde el disco."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    src_vocab_path = os.path.join(MODELS_DIR, "vocab_que.json")
    trg_vocab_path = os.path.join(MODELS_DIR, "vocab_es.json")
    model_path = os.path.join(MODELS_DIR, "simple_rnn_que_es.pt")

    if not os.path.exists(src_vocab_path) or not os.path.exists(trg_vocab_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            "No se encontraron los artefactos del modelo en 'models/'. "
            "Es necesario ejecutar el entrenamiento previamente (train.py)."
        )

    src_vocab = Vocabulary.load(src_vocab_path)
    trg_vocab = Vocabulary.load(trg_vocab_path)

    checkpoint = torch.load(model_path, map_location=device)
    hyperparams = checkpoint.get("hyperparameters", {})

    model = build_model(
        src_vocab_size=len(src_vocab),
        trg_vocab_size=len(trg_vocab),
        emb_dim=hyperparams.get("emb_dim", 128),
        hidden_dim=hyperparams.get("hidden_dim", 256),
        dropout=0.0,
        device=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, src_vocab, trg_vocab, hyperparams

def translate_sentence(sentence: str, model: Seq2SeqSimpleRNN,
                       src_vocab: Vocabulary, trg_vocab: Vocabulary,
                       max_len: int = 25, mask_unk: bool = True) -> Tuple[str, List[str]]:
    """
    Traduce una oración individual de Quechua a Español utilizando el modelo.
    Retorna la oración traducida y la lista de tokens generados.
    """
    model.eval()
    src_indices = src_vocab.numericalize(sentence, add_sos=True, add_eos=True)
    predicted_indices = model.translate(src_indices, max_len=max_len, mask_unk=mask_unk)
    predicted_tokens = [trg_vocab.idx2word.get(idx, "<unk>") for idx in predicted_indices]
    translated_text = detokenize(predicted_tokens)
    return translated_text, predicted_tokens

def evaluate_test_set(limit: Optional[int] = None) -> Dict[str, Any]:
    """
    Evalúa el modelo en el conjunto de prueba test.csv calculando BLEU scores
    y generando un reporte detallado.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, src_vocab, trg_vocab, hyperparams = load_trained_model(device)

    test_path = os.path.join(DATA_DIR, "test.csv")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Archivo de prueba no encontrado en {test_path}")

    test_df = pd.read_csv(test_path)
    if limit is not None:
        test_df = test_df.head(limit)

    smoothing = SmoothingFunction().method1

    references_corpus = []
    hypotheses_corpus = []
    sample_results = []

    bleu1_scores = []
    bleu2_scores = []
    bleu3_scores = []
    bleu4_scores = []

    print(f"Evaluando {len(test_df)} oraciones de prueba...")

    for idx, row in test_df.iterrows():
        src_text = str(row["quechua"])
        ref_text = str(row["espanol"])

        # Generar traducción
        pred_text, pred_tokens = translate_sentence(src_text, model, src_vocab, trg_vocab, mask_unk=True)
        ref_tokens = tokenize(ref_text)

        # Listas para corpus BLEU
        references_corpus.append([ref_tokens])
        hypotheses_corpus.append(pred_tokens)

        # Cálculo de BLEU por oración
        b1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
        b2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
        b3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
        b4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

        bleu1_scores.append(b1)
        bleu2_scores.append(b2)
        bleu3_scores.append(b3)
        bleu4_scores.append(b4)

        sample_results.append({
            "id": int(idx),
            "quechua": src_text,
            "referencia": ref_text,
            "prediccion": pred_text,
            "bleu1": round(b1 * 100, 2),
            "bleu2": round(b2 * 100, 2),
            "bleu4": round(b4 * 100, 2),
            "coincidencia_exacta": (ref_text.strip().lower() == pred_text.strip().lower())
        })

    # Corpus BLEU acumulativo
    corpus_b1 = corpus_bleu(references_corpus, hypotheses_corpus, weights=(1, 0, 0, 0), smoothing_function=smoothing)
    corpus_b2 = corpus_bleu(references_corpus, hypotheses_corpus, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
    corpus_b4 = corpus_bleu(references_corpus, hypotheses_corpus, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)

    avg_b1 = (sum(bleu1_scores) / len(bleu1_scores)) * 100
    avg_b2 = (sum(bleu2_scores) / len(bleu2_scores)) * 100
    avg_b3 = (sum(bleu3_scores) / len(bleu3_scores)) * 100
    avg_b4 = (sum(bleu4_scores) / len(bleu4_scores)) * 100
    exact_matches = sum(1 for s in sample_results if s["coincidencia_exacta"])

    summary = {
        "num_test_samples": len(test_df),
        "corpus_bleu_1": round(corpus_b1 * 100, 2),
        "corpus_bleu_2": round(corpus_b2 * 100, 2),
        "corpus_bleu_4": round(corpus_b4 * 100, 2),
        "avg_bleu_1": round(avg_b1, 2),
        "avg_bleu_2": round(avg_b2, 2),
        "avg_bleu_3": round(avg_b3, 2),
        "avg_bleu_4": round(avg_b4, 2),
        "exact_matches": exact_matches,
        "exact_match_ratio": round((exact_matches / len(test_df)) * 100, 2),
        "samples": sample_results
    }

    eval_output_path = os.path.join(MODELS_DIR, "evaluation_results.json")
    with open(eval_output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n================ RESULTADOS DE EVALUACIÓN ================")
    print(f"Total de muestras de prueba evaluadas: {summary['num_test_samples']}")
    print(f"Corpus BLEU-1: {summary['corpus_bleu_1']}%")
    print(f"Corpus BLEU-2: {summary['corpus_bleu_2']}%")
    print(f"Corpus BLEU-4: {summary['corpus_bleu_4']}%")
    print(f"Promedio BLEU-1 (Unigramas): {summary['avg_bleu_1']}%")
    print(f"Promedio BLEU-4 (Estándar):  {summary['avg_bleu_4']}%")
    print(f"Coincidencias exactas:       {summary['exact_matches']} ({summary['exact_match_ratio']}%)")
    print(f"Reporte guardado en: {eval_output_path}")

    return summary

if __name__ == "__main__":
    evaluate_test_set()
