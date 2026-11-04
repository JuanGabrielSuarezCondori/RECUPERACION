"""
rnn_model.py
Arquitectura de RNN Simple (SimpleRNN) Many-to-Many para Traducción Automática.
Implementa una arquitectura Seq2Seq con celdas recurrentes básicas (nn.RNN con activación tanh):
- Encoder: Embedding -> SimpleRNN (con pack_padded_sequence para descartar padding) -> Vector de Contexto
- Decoder: Embedding + Vector de Contexto -> SimpleRNN -> Linear Projection (vocabulario destino)
- Many-to-Many: Traducción de secuencia a secuencia (T_x -> T_y) con Teacher Forcing.
"""

import os
import sys
import random
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Tuple, List, Optional

# Asegurar importación compatible
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.preprocessor import PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX
except ImportError:
    from preprocessor import PAD_IDX, SOS_IDX, EOS_IDX, UNK_IDX

class SimpleRNNEncoder(nn.Module):
    """
    Encoder basado en una Red Neuronal Recurrente Simple (SimpleRNN / nn.RNN).
    Procesa la secuencia de entrada en Quechua paso a paso y produce un vector de contexto
    (el estado oculto h_t al final de la secuencia real sin contaminación de tokens <pad>).
    
    Ecuación recurrente básica:
      h_t = tanh(W_ih * x_t + b_ih + W_hh * h_{t-1} + b_hh)
    """
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float = 0.1):
        super(SimpleRNNEncoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # Capa 1: Embedding
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)

        # Capa 2: RNN Básica (SimpleRNN con función de activación tanh)
        self.rnn = nn.RNN(
            input_size=emb_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            nonlinearity='tanh'
        )

    def forward(self, src: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        src: [batch_size, src_len]
        lengths: [batch_size] longitudes reales de cada secuencia
        Retorna:
          hidden: [1, batch_size, hidden_dim] (Vector de Contexto final)
        """
        # [batch_size, src_len, emb_dim]
        embedded = self.dropout(self.embedding(src))

        if lengths is not None:
            # Empacar secuencias para que los tokens <pad> no alteren el estado oculto final
            packed = pack_padded_sequence(
                embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, hidden = self.rnn(packed)
        else:
            _, hidden = self.rnn(embedded)

        return hidden

class SimpleRNNDecoder(nn.Module):
    """
    Decoder basado en una Red Neuronal Recurrente Simple (SimpleRNN).
    Genera la secuencia objetivo en Español token por token.
    Para evitar el desvanecimiento del vector de contexto en oraciones de varios pasos,
    el vector de contexto se concatena a la entrada del embedding en cada paso temporal.
    
    Capa de salida: Proyección lineal a la distribución sobre el vocabulario objetivo.
    """
    def __init__(self, vocab_size: int, emb_dim: int, hidden_dim: int, dropout: float = 0.1):
        super(SimpleRNNDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # Capa 1: Embedding del token objetivo
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD_IDX)
        self.dropout = nn.Dropout(dropout)

        # Capa 2: RNN Básica alimentada por [Embedding; Vector de Contexto]
        self.rnn = nn.RNN(
            input_size=emb_dim + hidden_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            nonlinearity='tanh'
        )

        # Capa 3: Proyección Lineal a distribución sobre vocabulario
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_token: torch.Tensor, hidden: torch.Tensor,
                context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_token: [batch_size, 1]
        hidden:      [1, batch_size, hidden_dim]
        context:     [1, batch_size, hidden_dim] (Vector de contexto fijo del encoder)
        Retorna:
          prediction: [batch_size, vocab_size] (logits no normalizados)
          hidden:     [1, batch_size, hidden_dim]
        """
        # [batch_size, 1, emb_dim]
        embedded = self.dropout(self.embedding(input_token))

        # [batch_size, 1, hidden_dim]
        ctx_proj = context.permute(1, 0, 2)

        # Concatenar embedding + vector de contexto: [batch_size, 1, emb_dim + hidden_dim]
        rnn_input = torch.cat([embedded, ctx_proj], dim=2)

        # output: [batch_size, 1, hidden_dim]
        output, hidden = self.rnn(rnn_input, hidden)

        # prediction: [batch_size, vocab_size]
        prediction = self.fc_out(output.squeeze(1))
        return prediction, hidden

class Seq2SeqSimpleRNN(nn.Module):
    """
    Modelo completo Many-to-Many Seq2Seq con RNN Simple para traducción de Quechua a Español.
    Une el Encoder y el Decoder con soporte para Teacher Forcing durante el entrenamiento.
    """
    def __init__(self, encoder: SimpleRNNEncoder, decoder: SimpleRNNDecoder, device: torch.device):
        super(Seq2SeqSimpleRNN, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hidden_dim == decoder.hidden_dim, "hidden_dim del encoder y decoder deben coincidir"

    def forward(self, src: torch.Tensor, trg: torch.Tensor,
                src_lengths: Optional[torch.Tensor] = None,
                teacher_forcing_ratio: float = 0.5) -> torch.Tensor:
        """
        src: [batch_size, src_len]
        trg: [batch_size, trg_len]
        src_lengths: [batch_size]
        teacher_forcing_ratio: Probabilidad de usar la palabra real como siguiente entrada.
        
        Retorna:
          outputs: [batch_size, trg_len, trg_vocab_size]
        """
        batch_size = src.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size

        if src_lengths is None:
            src_lengths = (src != PAD_IDX).sum(dim=1)

        # Tensor para almacenar las predicciones de cada paso temporal
        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size, device=self.device)

        # 1. Pasar la secuencia fuente por el Encoder
        context = self.encoder(src, src_lengths)
        hidden = context

        # 2. Primer token de entrada al decoder es siempre <sos> (trg[:, 0])
        input_token = trg[:, 0].unsqueeze(1) # [batch_size, 1]

        # 3. Decodificación paso a paso (Many-to-Many)
        for t in range(1, trg_len):
            prediction, hidden = self.decoder(input_token, hidden, context)
            outputs[:, t, :] = prediction

            # Decidir si usamos Teacher Forcing
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = prediction.argmax(1).unsqueeze(1) # [batch_size, 1]
            
            # Siguiente entrada
            input_token = trg[:, t].unsqueeze(1) if teacher_force else top1

        return outputs

    def translate(self, src_indices: List[int], max_len: int = 25,
                  mask_unk: bool = True) -> List[int]:
        """
        Inferencia autoregresiva con decodificación greedy.
        Traduce una secuencia de tokens fuente Quechua a índices objetivo en Español.
        """
        self.eval()
        with torch.no_grad():
            src_tensor = torch.tensor(src_indices, dtype=torch.long, device=self.device).unsqueeze(0)
            src_lengths = torch.tensor([len(src_indices)], dtype=torch.long)
            
            context = self.encoder(src_tensor, src_lengths)
            hidden = context

            # Iniciar con token <sos>
            curr_token = torch.tensor([[SOS_IDX]], dtype=torch.long, device=self.device)
            predicted_indices = []

            for _ in range(max_len):
                prediction, hidden = self.decoder(curr_token, hidden, context)
                
                if mask_unk:
                    # Penalizar <unk> para favorecer palabras reales si están disponibles
                    prediction[:, UNK_IDX] = -1e9

                best_token = prediction.argmax(1).item()

                if best_token == EOS_IDX:
                    break
                predicted_indices.append(best_token)
                curr_token = torch.tensor([[best_token]], dtype=torch.long, device=self.device)

        return predicted_indices

def build_model(src_vocab_size: int, trg_vocab_size: int,
                emb_dim: int = 128, hidden_dim: int = 256,
                dropout: float = 0.1,
                device: Optional[torch.device] = None) -> Seq2SeqSimpleRNN:
    """Función de fábrica para instanciar el modelo Seq2Seq SimpleRNN."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    encoder = SimpleRNNEncoder(src_vocab_size, emb_dim, hidden_dim, dropout)
    decoder = SimpleRNNDecoder(trg_vocab_size, emb_dim, hidden_dim, dropout)
    model = Seq2SeqSimpleRNN(encoder, decoder, device).to(device)
    return model

if __name__ == "__main__":
    dev = torch.device("cpu")
    m = build_model(src_vocab_size=100, trg_vocab_size=120, emb_dim=32, hidden_dim=64, device=dev)
    print("Arquitectura Seq2Seq SimpleRNN creada:")
    print(m)
    
    dummy_src = torch.randint(1, 100, (4, 8), dtype=torch.long)
    dummy_trg = torch.randint(1, 120, (4, 10), dtype=torch.long)
    dummy_lens = torch.tensor([8, 7, 5, 4], dtype=torch.long)
    out = m(dummy_src, dummy_trg, dummy_lens, teacher_forcing_ratio=0.5)
    print("Salida forward shape:", out.shape)
