"""
app.py
Aplicación interactiva en Streamlit para el Traductor Quechua -> Español
con Red Neuronal Recurrente Simple (SimpleRNN) con arquitectura Many-to-Many.
"""

import os
import sys
import json
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch

# Asegurar importación de módulos propios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from src.preprocessor import Vocabulary, tokenize, detokenize, UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN
from src.rnn_model import build_model, Seq2SeqSimpleRNN
from src.evaluate import load_trained_model, translate_sentence, evaluate_test_set
from src.train import train_model

DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Configuración de página de Streamlit
st.set_page_config(
    page_title="Traductor Quechua a Español | SimpleRNN Many-to-Many",
    page_icon="🇵🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #7C2D12 0%, #B45309 50%, #D97706 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .main-header h1 {
        color: white;
        margin: 0;
        font-size: 2.2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #FEF3C7;
        margin: 6px 0 0 0;
        font-size: 1.05rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #B45309;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .translation-output {
        background-color: #FEF3C7;
        border-left: 5px solid #D97706;
        padding: 18px 24px;
        border-radius: 8px;
        font-size: 1.4rem;
        font-weight: 600;
        color: #78350F;
        margin-top: 15px;
    }
    .source-box {
        background-color: #F1F5F9;
        border-radius: 8px;
        padding: 14px;
        font-size: 1.1rem;
        color: #1E293B;
    }
    .tag-badge {
        display: inline-block;
        padding: 3px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: #E0E7FF;
        color: #3730A3;
        margin-right: 4px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 18px;
        border-radius: 6px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Funciones auxiliares de carga en caché
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_cached_model():
    """Carga en caché el modelo y los vocabularios."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, src_vocab, trg_vocab, hyperparams = load_trained_model(device)
        return model, src_vocab, trg_vocab, hyperparams, None
    except Exception as e:
        return None, None, None, None, str(e)

@st.cache_data(show_spinner=False)
def load_data_and_history():
    """Carga datos de corpus, historial y evaluación."""
    corpus_path = os.path.join(DATA_DIR, "corpus_quechua_espanol.csv")
    history_path = os.path.join(MODELS_DIR, "train_history.json")
    eval_path = os.path.join(MODELS_DIR, "evaluation_results.json")

    df_corpus = pd.read_csv(corpus_path) if os.path.exists(corpus_path) else None
    
    history = None
    if os.path.exists(history_path):
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

    evaluation = None
    if os.path.exists(eval_path):
        with open(eval_path, "r", encoding="utf-8") as f:
            evaluation = json.load(f)

    return df_corpus, history, evaluation

# Cargar recursos
model, src_vocab, trg_vocab, hyperparams, model_err = get_cached_model()
df_corpus, history, evaluation = load_data_and_history()

# -----------------------------------------------------------------------------
# Encabezado principal
# -----------------------------------------------------------------------------
st.markdown("""
<div class="main-header">
    <h1>🇵🇪 Traductor Quechua ⇄ Español</h1>
    <p>Red Neuronal Recurrente Simple (SimpleRNN) con Arquitectura Many-to-Many (Secuencia a Secuencia)</p>
</div>
""", unsafe_allow_html=True)

# Barra lateral de información del sistema
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cf/Flag_of_Peru.svg/320px-Flag_of_Peru.svg.png", width=70)
    st.title("⚙️ Estado del Modelo")
    
    device_name = "CUDA (GPU)" if torch.cuda.is_available() else "CPU"
    st.info(f"**Dispositivo:** `{device_name}`")

    if model is not None:
        st.success("✅ Modelo RNN Cargado")
        st.markdown(f"""
        - **Vocabulario Quechua:** `{len(src_vocab):,}` tokens
        - **Vocabulario Español:** `{len(trg_vocab):,}` tokens
        - **Dimensión Embedding:** `{hyperparams.get('emb_dim', 128)}`
        - **Dimensión Oculta RNN:** `{hyperparams.get('hidden_dim', 256)}`
        - **Celdas Recurrentes:** `SimpleRNN (tanh)`
        - **Arquitectura:** `Many-to-Many Seq2Seq`
        """)
    else:
        st.error(f"⚠️ Modelo no encontrado: {model_err}")
        st.caption("Ve a la pestaña 'Reentrenamiento' para compilar y entrenar el modelo.")

    st.markdown("---")
    st.markdown("""
    **Proyecto Deep Learning NLP**
    - **Lengua Origen:** Quechua Sureño / Chanka / Cusco
    - **Lengua Destino:** Español
    - **Métrica Principal:** BLEU Score (NLTK)
    - **Framework:** PyTorch & Streamlit
    """)

# -----------------------------------------------------------------------------
# Pestañas de la Aplicación
# -----------------------------------------------------------------------------
tab_trans, tab_eval, tab_train, tab_corpus, tab_theory = st.tabs([
    "🌐 Traductor en Vivo",
    "📊 Evaluación & BLEU",
    "⚙️ Reentrenamiento Interactivo",
    "📚 Explorador del Corpus",
    "🧠 Arquitectura Many-to-Many & Teoría"
])

# =============================================================================
# PESTAÑA 1: TRADUCTOR EN VIVO
# =============================================================================
with tab_trans:
    st.subheader("🌐 Traducción de Quechua a Español en Tiempo Real")
    st.write(
        "Ingresa una oración o frase en Quechua para traducirla al Español utilizando la RNN Simple entrenada. "
        "También puedes presionar los botones de frases cotidianas precargadas:"
    )

    # Frases de ejemplo organizadas por categorías
    st.markdown("**Ejemplos rápidos para probar:**")
    col_ex1, col_ex2, col_ex3, col_ex4 = st.columns(4)

    example_input = ""
    with col_ex1:
        if st.button("🌟 Allin punchaw (Buenos días)"):
            example_input = "Allin punchaw"
        if st.button("🤝 Allillanchu (¿Cómo estás?)"):
            example_input = "Allillanchu"
        if st.button("🙏 Sulpayki (Gracias)"):
            example_input = "Sulpayki"

    with col_ex2:
        if st.button("👤 Imataq sutiyki? (Nombre)"):
            example_input = "Imataq sutiyki?"
        if st.button("🙋 Sutiyqa Juanmi (Me llamo)"):
            example_input = "Sutiyqa Juanmi"
        if st.button("👋 Tupananchiskama (Hasta luego)"):
            example_input = "Tupananchiskama"

    with col_ex3:
        if st.button("🍲 Ñuqa mikuni (Yo como)"):
            example_input = "Ñuqa mikuni"
        if st.button("💧 Yakuta upyani (Bebo agua)"):
            example_input = "Yakuta upyani"
        if st.button("🏠 Wasiyman rini (Voy a casa)"):
            example_input = "Wasiyman rini"

    with col_ex4:
        if st.button("👨 Taytayqa llamkan (Padre)"):
            example_input = "Taytayqa llamkan"
        if st.button("🐱 Misi puñun (Gato duerme)"):
            example_input = "Misi puñun"
        if st.button("🐶 Allqu anchatam anyan (Perro)"):
            example_input = "Allqu anchatam anyan"

    # Campo de texto interactivo
    user_input = st.text_area(
        "Texto en Quechua:",
        value=example_input if example_input else "Allin punchaw, imataq sutiyki?",
        height=100,
        placeholder="Escribe aquí tu frase en Quechua (e.g. allillanchu, ñuqa mikuni, sulpayki)..."
    )

    col_btn, col_opt = st.columns([1, 3])
    with col_btn:
        translate_btn = st.button("🚀 Traducir al Español", type="primary", use_container_width=True)
    with col_opt:
        mask_unk_opt = st.checkbox("Penalizar token <unk> (Recomendado para priorizar palabras léxicas reales)", value=True)

    if translate_btn or user_input:
        if model is None:
            st.error("El modelo no está disponible. Entrénalo primero en la pestaña correspondiente.")
        elif not user_input.strip():
            st.warning("Por favor ingresa una frase en Quechua.")
        else:
            with st.spinner("Procesando con Encoder-Decoder SimpleRNN..."):
                t_start = time.time()
                translated_text, pred_tokens = translate_sentence(
                    user_input, model, src_vocab, trg_vocab,
                    max_len=25, mask_unk=mask_unk_opt
                )
                duration_ms = (time.time() - t_start) * 1000

            st.markdown("### Resultado de la Traducción:")
            st.markdown(f"""
            <div class="translation-output">
                🇪🇸 {translated_text}
            </div>
            """, unsafe_allow_html=True)
            st.caption(f"⏱️ Tiempo de inferencia autoregresiva: **{duration_ms:.2f} ms**")

            # Desglose técnico de la traducción
            with st.expander("🔍 Inspección Técnica del Proceso Many-to-Many paso a paso", expanded=False):
                st.markdown("#### 1. Tokenización y Mapeo en el Encoder (Quechua)")
                input_tokens = tokenize(user_input)
                token_details = []
                for tok in input_tokens:
                    idx = src_vocab.word2idx.get(tok, src_vocab.word2idx.get(UNK_TOKEN))
                    is_known = tok in src_vocab.word2idx
                    token_details.append({
                        "Token": tok,
                        "Índice en Vocabulario": idx,
                        "Estado": "✅ En Vocabulario" if is_known else "⚠️ Fuera de Vocabulario (<unk>)"
                    })
                st.dataframe(pd.DataFrame(token_details), use_container_width=True)

                st.markdown("#### 2. Vector de Contexto y Decodificador (Español)")
                st.markdown(f"""
                - **Secuencia de Entrada:** `{input_tokens}` (longitud: {len(input_tokens)})
                - **Vector de Contexto generado ($h_{{T_x}}$):** Tensor de dimensión `[{hyperparams.get('hidden_dim', 256)}]` que resume toda la frase.
                - **Tokens Generados por el Decoder:** `{pred_tokens}`
                - **Reconstrucción Final:** `{translated_text}`
                """)

# =============================================================================
# PESTAÑA 2: EVALUACIÓN & MÉTRICAS (BLEU)
# =============================================================================
with tab_eval:
    st.subheader("📊 Evaluación del Modelo con Corpus de Prueba (BLEU Score)")
    st.write(
        "El modelo se evalúa sobre el subconjunto de prueba independiente (`test.csv`, 275 oraciones) "
        "mediante la métrica estándar de Traducción Automática **BLEU (Bilingual Evaluation Understudy)** "
        "utilizando la implementación oficial de NLTK con funciones de suavizado (SmoothingFunction)."
    )

    if evaluation is not None:
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        with col_m1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Corpus BLEU-1 (Unigramas)</div>
                <div class="metric-value">{evaluation.get('corpus_bleu_1', 0)}%</div>
                <small>Precisión léxica palabra por palabra</small>
            </div>
            """, unsafe_allow_html=True)

        with col_m2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Corpus BLEU-2 (Bigramas)</div>
                <div class="metric-value">{evaluation.get('corpus_bleu_2', 0)}%</div>
                <small>Cohesión de pares de palabras</small>
            </div>
            """, unsafe_allow_html=True)

        with col_m3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Corpus BLEU-4 (Estándar)</div>
                <div class="metric-value">{evaluation.get('corpus_bleu_4', 0)}%</div>
                <small>Métrica de 4-gramas acumulativa</small>
            </div>
            """, unsafe_allow_html=True)

        with col_m4:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Coincidencias Exactas</div>
                <div class="metric-value">{evaluation.get('exact_match_ratio', 0)}%</div>
                <small>{evaluation.get('exact_matches', 0)} / {evaluation.get('num_test_samples', 0)} oraciones</small>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📋 Muestras del Conjunto de Prueba: Referencia vs Predicción SimpleRNN")
        
        samples = evaluation.get("samples", [])
        if samples:
            df_samples = pd.DataFrame(samples)
            
            # Filtro interactivo
            filter_opt = st.radio(
                "Filtrar muestras:",
                ["Todas las muestras", "Solo coincidencias exactas (100% acierto)", "Solo oraciones con discrepancias"],
                horizontal=True
            )
            
            if filter_opt == "Solo coincidencias exactas (100% acierto)":
                filtered_df = df_samples[df_samples["coincidencia_exacta"] == True]
            elif filter_opt == "Solo oraciones con discrepancias":
                filtered_df = df_samples[df_samples["coincidencia_exacta"] == False]
            else:
                filtered_df = df_samples

            st.write(f"Mostrando **{len(filtered_df)}** oraciones:")
            st.dataframe(
                filtered_df[["id", "quechua", "referencia", "prediccion", "bleu1", "bleu4", "coincidencia_exacta"]].rename(
                    columns={
                        "id": "ID",
                        "quechua": "Quechua (Entrada)",
                        "referencia": "Español (Referencia Ground Truth)",
                        "prediccion": "Predicción SimpleRNN",
                        "bleu1": "BLEU-1 (%)",
                        "bleu4": "BLEU-4 (%)",
                        "coincidencia_exacta": "Acierto Exacto"
                    }
                ),
                use_container_width=True,
                height=350
            )

        # Gráfico de distribución de BLEU scores
        if samples:
            st.markdown("#### 📈 Distribución de BLEU-1 por Oración en el Conjunto de Prueba")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            b1_values = [s["bleu1"] for s in samples]
            ax.hist(b1_values, bins=20, color="#D97706", edgecolor="#78350F", alpha=0.85)
            ax.set_title("Histograma de Frecuencia de Scores BLEU-1 (%)", fontsize=12, fontweight="bold")
            ax.set_xlabel("BLEU-1 Score (%)", fontsize=10)
            ax.set_ylabel("Cantidad de Oraciones", fontsize=10)
            ax.grid(axis="y", linestyle="--", alpha=0.5)
            st.pyplot(fig)
            plt.close(fig)

    else:
        st.info("Aún no se ha generado el reporte de evaluación. Haz clic en el botón inferior para ejecutar la evaluación completa:")
        if st.button("Ejecutar Evaluación en test.csv"):
            with st.spinner("Evaluando conjunto de prueba..."):
                evaluate_test_set()
                st.cache_data.clear()
                st.rerun()

# =============================================================================
# PESTAÑA 3: REENTRENAMIENTO INTERACTIVO
# =============================================================================
with tab_train:
    st.subheader("⚙️ Experimentación y Reentrenamiento Interactivo")
    st.write(
        "Ajusta los hiperparámetros de la Red Neuronal Recurrente Simple y entrena el modelo directamente "
        "desde esta interfaz. Podrás visualizar las curvas de pérdida (Loss) y perplejidad en tiempo real."
    )

    col_h1, col_h2, col_h3 = st.columns(3)
    with col_h1:
        n_epochs = st.slider("Número de Épocas:", min_value=5, max_value=40, value=20, step=5)
        learning_rate = st.select_slider(
            "Tasa de Aprendizaje (LR):",
            options=[0.0005, 0.001, 0.002, 0.003, 0.005, 0.01],
            value=0.003
        )

    with col_h2:
        hidden_dim = st.selectbox("Dimensión Oculta RNN (Hidden Dim):", options=[64, 128, 256], index=2)
        emb_dim = st.selectbox("Dimensión de Embeddings:", options=[32, 64, 128], index=2)

    with col_h3:
        teacher_forcing = st.slider("Teacher Forcing Ratio:", min_value=0.0, max_value=1.0, value=0.6, step=0.1)
        batch_size = st.selectbox("Tamaño de Batch:", options=[16, 32, 64], index=1)

    start_train_btn = st.button("🔥 Iniciar Entrenamiento del Modelo", type="primary")

    # Contenedor para visualización de entrenamiento en vivo
    if start_train_btn:
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()

        train_loss_history = []
        val_loss_history = []
        epoch_numbers = []

        def training_callback(epoch, total_epochs, metrics):
            progress = epoch / total_epochs
            progress_bar.progress(progress)
            status_text.markdown(f"**Entrenando Época {epoch}/{total_epochs}** | Train Loss: `{metrics['train_loss']:.4f}` | Val Loss: `{metrics['val_loss']:.4f}`")

            train_loss_history.append(metrics["train_loss"])
            val_loss_history.append(metrics["val_loss"])
            epoch_numbers.append(epoch)

            # Actualizar gráfico interactivo en vivo
            fig_live, ax_live = plt.subplots(figsize=(9, 3.5))
            ax_live.plot(epoch_numbers, train_loss_history, label="Pérdida de Entrenamiento (Train Loss)", color="#B45309", marker="o", linewidth=2)
            ax_live.plot(epoch_numbers, val_loss_history, label="Pérdida de Validación (Val Loss)", color="#2563EB", marker="s", linewidth=2)
            ax_live.set_xlabel("Época")
            ax_live.set_ylabel("CrossEntropy Loss")
            ax_live.set_title("Curvas de Aprendizaje en Vivo (SimpleRNN Many-to-Many)")
            ax_live.legend()
            ax_live.grid(True, linestyle="--", alpha=0.5)
            chart_placeholder.pyplot(fig_live)
            plt.close(fig_live)

        try:
            with st.spinner("Entrenando red neuronal..."):
                train_model(
                    epochs=n_epochs,
                    batch_size=batch_size,
                    emb_dim=emb_dim,
                    hidden_dim=hidden_dim,
                    lr=learning_rate,
                    teacher_forcing_ratio=teacher_forcing,
                    min_freq=1,
                    progress_callback=training_callback
                )
                
                # Ejecutar evaluación automática post-entrenamiento
                status_text.markdown("🔄 Ejecutando evaluación en el conjunto de prueba...")
                evaluate_test_set()

            st.success("✅ ¡Entrenamiento completado y mejor modelo guardado exitosamente!")
            # Limpiar caché de Streamlit para recargar el nuevo modelo
            st.cache_resource.clear()
            st.cache_data.clear()
            time.sleep(1)
            st.rerun()

        except Exception as err:
            st.error(f"Error durante el entrenamiento: {err}")

    # Mostrar curvas del último entrenamiento guardado
    elif history is not None:
        st.markdown("#### 📈 Curvas de Aprendizaje del Modelo Guardado")
        epochs_x = history.get("epochs", [])
        train_y = history.get("train_loss", [])
        val_y = history.get("val_loss", [])

        if epochs_x and train_y and val_y:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Loss
            ax1.plot(epochs_x, train_y, label="Train Loss", color="#B45309", marker="o")
            ax1.plot(epochs_x, val_y, label="Val Loss", color="#2563EB", marker="s")
            ax1.set_title("Convergencia de Función de Pérdida (Cross-Entropy)")
            ax1.set_xlabel("Época")
            ax1.set_ylabel("Loss")
            ax1.grid(True, linestyle="--", alpha=0.5)
            ax1.legend()

            # Perplejidad
            train_ppl = history.get("train_ppl", [])
            val_ppl = history.get("val_ppl", [])
            if train_ppl and val_ppl:
                ax2.plot(epochs_x, train_ppl, label="Train Perplexity", color="#D97706", marker="o")
                ax2.plot(epochs_x, val_ppl, label="Val Perplexity", color="#3B82F6", marker="s")
                ax2.set_title("Perplejidad por Época (PPL = exp(loss))")
                ax2.set_xlabel("Época")
                ax2.set_ylabel("Perplejidad")
                ax2.grid(True, linestyle="--", alpha=0.5)
                ax2.legend()

            st.pyplot(fig)
            plt.close(fig)

            hp = history.get("hyperparameters", {})
            st.info(f"**Hiperparámetros de este modelo:** Épocas: `{hp.get('epochs')}`, Hidden Dim: `{hp.get('hidden_dim')}`, Emb Dim: `{hp.get('emb_dim')}`, LR: `{hp.get('lr')}`, Teacher Forcing: `{hp.get('teacher_forcing_ratio')}`, Tiempo Total: `{history.get('total_time_seconds', 0)}s`")

# =============================================================================
# PESTAÑA 4: EXPLORADOR DEL CORPUS
# =============================================================================
with tab_corpus:
    st.subheader("📚 Explorador del Corpus Paralelo Quechua - Español")
    st.write(
        "El proyecto utiliza un corpus consolidado compuesto por pares conversacionales y frases cotidianas "
        "curadas de Quechua Chanka/Collao, complementado con oraciones limpias y filtradas de AmericasNLP / Siminchik."
    )

    if df_corpus is not None:
        c1, c2, c3, c4 = st.columns(4)
        n_total = len(df_corpus)
        n_train = int(n_total * 0.8)
        n_val = int(n_total * 0.1)
        n_test = n_total - n_train - n_val

        with c1:
            st.metric("Total de Pares Paralelos", f"{n_total:,}")
        with c2:
            st.metric("Conjunto de Entrenamiento (80%)", f"{n_train:,}")
        with c3:
            st.metric("Conjunto de Validación (10%)", f"{n_val:,}")
        with c4:
            st.metric("Conjunto de Prueba (10%)", f"{n_test:,}")

        st.markdown("---")
        search_query = st.text_input("🔍 Buscar palabra o frase en Quechua o Español:", placeholder="Ejemplo: allin, mikuni, padre, comer, yaku...")

        if search_query.strip():
            filtered_corpus = df_corpus[
                df_corpus["quechua"].str.contains(search_query, case=False, na=False) |
                df_corpus["espanol"].str.contains(search_query, case=False, na=False)
            ]
        else:
            filtered_corpus = df_corpus

        st.dataframe(
            filtered_corpus.rename(columns={"quechua": "Quechua", "espanol": "Español", "fuente": "Origen"}),
            use_container_width=True,
            height=300
        )

        st.markdown("#### 📊 Análisis de Longitud de Oraciones")
        fig_lens, ax_l = plt.subplots(figsize=(10, 3.5))
        que_lens = df_corpus["quechua"].apply(lambda s: len(str(s).split()))
        esp_lens = df_corpus["espanol"].apply(lambda s: len(str(s).split()))

        ax_l.hist(que_lens, bins=12, alpha=0.6, label="Quechua (Palabras)", color="#B45309")
        ax_l.hist(esp_lens, bins=12, alpha=0.6, label="Español (Palabras)", color="#2563EB")
        ax_l.set_xlabel("Número de palabras por oración")
        ax_l.set_ylabel("Frecuencia")
        ax_l.set_title("Distribución de Longitud de Secuencias (Ideal para SimpleRNN)")
        ax_l.legend()
        ax_l.grid(True, linestyle="--", alpha=0.5)
        st.pyplot(fig_lens)
        plt.close(fig_lens)

# =============================================================================
# PESTAÑA 5: ARQUITECTURA MANY-TO-MANY & TEORÍA
# =============================================================================
with tab_theory:
    st.subheader("🧠 Arquitectura de RNN Simple Many-to-Many para Traducción")
    
    st.markdown("""
    ### 1. ¿Por qué Many-to-Many?
    En el aprendizaje profundo para Procesamiento de Lenguaje Natural (NLP), las arquitecturas se clasifican según la dimensionalidad temporal de entrada y salida:
    - **One-to-One:** Red clásica feedforward (vector a etiqueta).
    - **One-to-Many:** Subtitulado de imágenes (imagen a secuencia de texto).
    - **Many-to-One:** Análisis de sentimiento (oración a polaridad positiva/negativa).
    - **Many-to-Many (Secuencia a Secuencia / Seq2Seq):** Traducción Automática.
    
    En la traducción de **Quechua a Español**, la longitud de la oración fuente ($T_x$) y la longitud de la oración traducida ($T_y$) **no coinciden** necesariamente, y el orden sintáctico difiere radicalmente:
    - El Quechua es una lengua aglutinante de orden **SOV** (Sujeto - Objeto - Verbo).
    - El Español es una lengua flexiva de orden **SVO** (Sujeto - Verbo - Objeto).
    
    Por ello, se utiliza una arquitectura **Many-to-Many Asíncrona (Encoder-Decoder)**:
    """)

    # Diagrama ASCII / Mermaid interactivo
    st.code("""
    ┌────────────────────────────────────────────────────────────────────────────────────────┐
    │                         ARQUITECTURA MANY-TO-MANY (SEQ2SEQ)                            │
    └────────────────────────────────────────────────────────────────────────────────────────┘
    
       [Quechua x₁]     [Quechua x₂]     ...     [Quechua xTx]
            │                │                        │
            ▼                ▼                        ▼
       ┌─────────┐      ┌─────────┐              ┌─────────┐
       │Embedding│      │Embedding│              │Embedding│
       └────┬────┘      └────┬────┘              └────┬────┘
            │                │                        │
            ▼                ▼                        ▼
    h₀ ──> ┌───┐   h₁  ──>  ┌───┐   ...  hTx-1 ──>   ┌───┐
           │RNN│ ─────────> │RNN│ ───────────────>  │RNN│ ──> Vector de Contexto (c = hTx)
           └───┘            └───┘                    └───┘                 │
                                                                           │
    ═══════════════════════════════════════════════════════════════════════╪═══════════════════
                                                                           ▼
                                                             ┌───────────────────────────┐
                                                             │ Estado Inicial h₀_dec = c │
                                                             └─────────────┬─────────────┘
                                                                           │
                                                                           ▼
       [Token <sos>]          [Palabra y₁]             [Palabra y₂]     ...
            │                      │                        │
            ▼                      ▼                        ▼
       ┌─────────┐            ┌─────────┐              ┌─────────┐
       │Embedding│ + c        │Embedding│ + c          │Embedding│ + c
       └────┬────┘            └────┬────┘              └────┬────┘
            │                      │                        │
            ▼                      ▼                        ▼
           ┌───┐                  ┌───┐                    ┌───┐
      h₀ ─>│RNN│ ──── h₁ ───────> │RNN│ ──── h₂ ─────────> │RNN│ ──> ...
           └───┘                  └───┘                    └───┘
             │                      │                        │
             ▼                      ▼                        ▼
        ┌─────────┐            ┌─────────┐              ┌─────────┐
        │ Linear  │            │ Linear  │              │ Linear  │
        └────┬────┘            └────┬────┘              └────┬────┘
             │                      │                        │
             ▼                      ▼                        ▼
        [Español y₁]           [Español y₂]             [Token <eos>]
    """, language="text")

    st.markdown("""
    ### 2. Formulaciones Matemáticas de la RNN Simple (SimpleRNN)

    #### Capa Recurrente Simple (SimpleRNN / Elman Cell):
    A diferencia de LSTM o GRU que usan compuertas de olvido, la RNN simple aplica una transformación lineal al vector de entrada y al estado oculto anterior, pasando por la tangente hiperbólica $\\tanh$:
    """)
    st.latex(r"h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})")

    st.markdown("""
    #### Proyección de Salida del Decodificador:
    El estado oculto del decodificador en el paso $t$ se proyecta mediante una capa densa (Linear) sobre todo el vocabulario objetivo español $V_{\\text{esp}}$:
    """)
    st.latex(r"P(y_t = w \mid y_{<t}, x) = \frac{\exp(W_{\text{out}} h_t^{\text{dec}} + b_{\text{out}})_w}{\sum_{v \in V_{\text{esp}}} \exp(W_{\text{out}} h_t^{\text{dec}} + b_{\text{out}})_v}")

    st.markdown("""
    #### Función de Pérdida con Enmascaramiento de Padding:
    Se minimiza la entropía cruzada ignorando rigurosamente los tokens de relleno `<pad>` (índice 0):
    """)
    st.latex(r"\mathcal{L} = - \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^{T_i} \mathbb{I}(y_{i,t} \neq \text{<pad>}) \cdot \log P(y_{i,t} \mid y_{i,<t}, x_i)")

    st.markdown("""
    ### 3. Técnicas Clave de Implementación
    1. **Empaquetado de Secuencias (`pack_padded_sequence`):**
       En una RNN Simple, procesar tokens de relleno `<pad>` al final de la oración sobrescribe el estado oculto y diluye la representación semántica. Al usar `pack_padded_sequence`, el cálculo recurrente se detiene en el último token real de cada oración.
    2. **Persistencia del Vector de Contexto:**
       Para contrarrestar el desvanecimiento del gradiente (vanishing gradient) inherente a las RNN simples, el vector de contexto $c$ se inyecta como entrada a la celda recurrente en cada paso de decodificación $[e(y_t); c]$.
    3. **Teacher Forcing:**
       Durante el entrenamiento, con una probabilidad configurable ($p = 0.6$), el decodificador recibe como entrada el token real objetivo (ground truth) en lugar de su propia predicción previa, acelerando la convergencia.
    4. **Métrica BLEU (Bilingual Evaluation Understudy):**
       Evalúa la calidad comparando la concordancia de $n$-gramas (1-gramas hasta 4-gramas) entre la traducción predicha y la referencia humana, penalizando oraciones excesivamente breves (Brevity Penalty).
    """)

# Pie de página
st.markdown("---")
st.markdown(
    "<center><small>🇵🇪 Proyecto Deep Learning: Traductor Quechua-Español con Red Neuronal Recurrente Many-to-Many | Python & Streamlit</small></center>",
    unsafe_allow_html=True
)

