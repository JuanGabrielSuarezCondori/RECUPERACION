"""
preprocessor.py
Preprocesamiento para el traductor Quechua -> Español:
- Tokenización con separación de signos de puntuación y preservación morfológica.
- Clase Vocabulary con tokens especiales (<pad>, <unk>, <sos>, <eos>).
- Transformación a secuencias numéricas (numericalization) y padding.
- Dataset y Collate function para PyTorch DataLoader.
"""

import re
import json
import os
from typing import List, Dict, Optional, Tuple
import torch
from torch.utils.data import Dataset

# Tokens especiales universales
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"

PAD_IDX = 0
UNK_IDX = 1
SOS_IDX = 2
EOS_IDX = 3

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, SOS_TOKEN, EOS_TOKEN]

def tokenize(text: str) -> List[str]:
    """
    Tokeniza una oración en palabras y signos de puntuación independientes.
    Preserva apóstrofes internos típicos del Quechua (e.g. p'unchaw, mawk'a).
    """
    if not isinstance(text, str):
        return []
    text = text.lower().strip()
    
    # Separar signos de puntuación con espacios
    for punct in [r'\?', r'¿', r'!', r'¡', r'\.', r',', r':', r';', r'"']:
        text = re.sub(f"({punct})", r" \1 ", text)
        
    # Reemplazar múltiples espacios por uno solo
    tokens = text.split()
    return tokens

def detokenize(tokens: List[str]) -> str:
    """Reconstruye una oración fluida a partir de una lista de tokens."""
    text = " ".join(tokens)
    # Pegar signos de puntuación correspondientes
    text = re.sub(r'\s+([\.,\?!:;])', r'\1', text)
    text = re.sub(r'([¿¡])\s+', r'\1', text)
    # Capitalizar primera letra si es aplicable
    if len(text) > 0:
        text = text[0].upper() + text[1:]
    return text.strip()

class Vocabulary:
    """Clase para gestionar el mapeo palabra <-> índice numérico."""
    def __init__(self, name: str = "vocab"):
        self.name = name
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Dict[str, int] = {}
        self._init_special_tokens()

    def _init_special_tokens(self):
        for idx, token in enumerate(SPECIAL_TOKENS):
            self.word2idx[token] = idx
            self.idx2word[idx] = token

    def build_vocab(self, sentences: List[str], min_freq: int = 1):
        """Construye el vocabulario a partir de una lista de oraciones."""
        # Contar frecuencias
        for sentence in sentences:
            tokens = tokenize(sentence)
            for token in tokens:
                self.word_counts[token] = self.word_counts.get(token, 0) + 1

        # Asignar índices a palabras que cumplan min_freq
        curr_idx = len(SPECIAL_TOKENS)
        for token, count in self.word_counts.items():
            if count >= min_freq and token not in self.word2idx:
                self.word2idx[token] = curr_idx
                self.idx2word[curr_idx] = token
                curr_idx += 1

    def numericalize(self, sentence: str, add_sos: bool = True, add_eos: bool = True) -> List[int]:
        """Convierte una oración en una lista de índices numéricos con SOS y EOS opcionales."""
        tokens = tokenize(sentence)
        indices = []
        if add_sos:
            indices.append(SOS_IDX)
        for token in tokens:
            indices.append(self.word2idx.get(token, UNK_IDX))
        if add_eos:
            indices.append(EOS_IDX)
        return indices

    def denumericalize(self, indices: List[int], strip_special: bool = True) -> List[str]:
        """Convierte una lista de índices a lista de palabras."""
        words = []
        for idx in indices:
            # Si el índice es un tensor o numpy
            if hasattr(idx, 'item'):
                idx = idx.item()
            word = self.idx2word.get(idx, UNK_TOKEN)
            if strip_special and word in SPECIAL_TOKENS:
                if word == EOS_TOKEN:
                    break
                continue
            words.append(word)
        return words

    def __len__(self) -> int:
        return len(self.word2idx)

    def save(self, filepath: str):
        """Guarda el vocabulario en un archivo JSON."""
        data = {
            "name": self.name,
            "word2idx": self.word2idx,
            "idx2word": {str(k): v for k, v in self.idx2word.items()},
            "word_counts": self.word_counts
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> "Vocabulary":
        """Carga un vocabulario desde un archivo JSON."""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = cls(name=data.get("name", "vocab"))
        vocab.word2idx = data["word2idx"]
        vocab.idx2word = {int(k): v for k, v in data["idx2word"].items()}
        vocab.word_counts = data.get("word_counts", {})
        return vocab

class TranslationDataset(Dataset):
    """
    Dataset para pares de traducción Quechua-Español.
    Pre-numericaliza todas las secuencias en el constructor para máximo rendimiento en memoria.
    """
    def __init__(self, quechua_sentences: List[str], espanol_sentences: List[str],
                 src_vocab: Vocabulary, trg_vocab: Vocabulary):
        assert len(quechua_sentences) == len(espanol_sentences), "Las listas deben tener la misma longitud"
        self.quechua_raw = quechua_sentences
        self.espanol_raw = espanol_sentences
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab

        # Pre-numericalizar en memoria para acelerar el entrenamiento 10x
        self.src_tensors = [
            torch.tensor(src_vocab.numericalize(s, add_sos=True, add_eos=True), dtype=torch.long)
            for s in quechua_sentences
        ]
        self.trg_tensors = [
            torch.tensor(trg_vocab.numericalize(t, add_sos=True, add_eos=True), dtype=torch.long)
            for t in espanol_sentences
        ]

    def __len__(self) -> int:
        return len(self.quechua_raw)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str, str]:
        return (
            self.src_tensors[idx],
            self.trg_tensors[idx],
            self.quechua_raw[idx],
            self.espanol_raw[idx]
        )

def translation_collate_fn(batch):
    """
    Función de agrupamiento (collate) para DataLoader con padding dinámico y tracking de longitudes reales.
    Retorna:
      src_padded:  [batch_size, max_src_len]
      trg_padded:  [batch_size, max_trg_len]
      src_lengths: [batch_size] (longitudes efectivas sin padding)
      src_raws:    lista de textos fuente originales
      trg_raws:    lista de textos destino originales
    """
    src_list, trg_list, src_raws, trg_raws = zip(*batch)

    src_lengths = torch.tensor([len(s) for s in src_list], dtype=torch.long)
    max_src_len = max(len(s) for s in src_list)
    src_padded = torch.full((len(src_list), max_src_len), PAD_IDX, dtype=torch.long)
    for i, s in enumerate(src_list):
        src_padded[i, :len(s)] = s

    max_trg_len = max(len(t) for t in trg_list)
    trg_padded = torch.full((len(trg_list), max_trg_len), PAD_IDX, dtype=torch.long)
    for i, t in enumerate(trg_list):
        trg_padded[i, :len(t)] = t

    return src_padded, trg_padded, src_lengths, src_raws, trg_raws

if __name__ == "__main__":
    # Test rápido de tokenización y vocabulario
    test_que = "Allillanchu, imataq sutiyki? Ñuqa yachachiq kani."
    test_es = "¡Hola! ¿Cómo estás? Mi nombre es Juan."
    
    print("Tokens Quechua:", tokenize(test_que))
    print("Tokens Español:", tokenize(test_es))
    
    v = Vocabulary("test")
    v.build_vocab([test_que])
    num = v.numericalize(test_que)
    print("Numericalized:", num)
    print("Denumericalized:", v.denumericalize(num))
    print("Detokenized:", detokenize(v.denumericalize(num)))
