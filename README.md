# 🇵🇪 RunasimiTrans: Traductor Quechua ⇄ Español con RNN Many-to-Many

Este proyecto implementa un sistema completo de **Traducción Automática (Machine Translation)** de **Quechua a Español** utilizando una **Red Neuronal Recurrente Simple (SimpleRNN)** con arquitectura **Many-to-Many (Secuencia a Secuencia / Seq2Seq)**, desarrollada en **Python**, **PyTorch** y con una interfaz gráfica web interactiva en **Streamlit**.

---

## 📌 Tabla de Contenidos
1. [Descripción del Proyecto](#-descripción-del-proyecto)
2. [Aclaración Teórica: ¿Cuántos Ciclos Convolucionales Tiene?](#-aclaración-teórica-cuántos-ciclos-convolucionales-tiene)
3. [Todo el Proceso Paso a Paso que Sigue el Modelo (Pipeline End-to-End)](#-todo-el-proceso-paso-a-paso-que-sigue-el-modelo-pipeline-end-to-end)
4. [Recopilación y Preprocesamiento de Datos](#-recopilación-y-preprocesamiento-de-datos)
5. [Arquitectura de la Red Neuronal (Many-to-Many)](#-arquitectura-de-la-red-neuronal-many-to-many)
6. [Entrenamiento y Mitigación de Limitaciones de SimpleRNN](#-entrenamiento-y-mitigación-de-limitaciones-de-simplernn)
7. [Evaluación y Métricas (BLEU Score)](#-evaluación-y-métricas-bleu-score)
8. [Estructura del Proyecto](#-estructura-del-proyecto)
9. [Instalación y Ejecución](#-instalación-y-ejecución)
10. [Uso de la Aplicación Streamlit](#-uso-de-la-aplicación-streamlit)

---

## 📖 Descripción del Proyecto

El **Quechua** (Runasimi) es la lengua originaria más hablada de América, con millones de hablantes en la región andina (Perú, Bolivia, Ecuador). Posee una morfología predominantemente **aglutinante** y una estructura sintáctica canónica **SOV** (Sujeto - Objeto - Verbo).

Por su parte, el **Español** es una lengua romance con estructura **SVO** (Sujeto - Verbo - Objeto) y morfología flexiva.

Debido a que el orden de las palabras y la longitud de las oraciones difieren entre ambas lenguas ($T_x \neq T_y$), la traducción automática requiere una formulación **Many-to-Many Asíncrona (Seq2Seq)**:
- Un **Encoder** procesa los $T_x$ tokens en Quechua y comprime su significado en un **vector de contexto**.
- Un **Decoder** utiliza ese vector de contexto para emitir de forma autoregresiva los $T_y$ tokens en Español.

---

## 🔬 Aclaración Teórica: ¿Cuántos Ciclos Convolucionales Tiene?

> [!IMPORTANT]
> **Respuesta Directa y Rigurosa:**
> El modelo **TIENE CERO (0) CICLOS O CAPAS CONVOLUCIONALES**. 
> Este sistema es una **Red Neuronal Recurrente Pura (SimpleRNN Many-to-Many)** y no una Red Neuronal Convolucional (CNN).

### ¿Por qué NO utiliza ciclos convolucionales?
1. **Naturaleza Matemática Diferente:**
   - Una **Red Convolucional (CNN)** aplica filtros espaciales o temporales rígidos ($K \times K$ o 1D kernel $K$) mediante la operación de convolución discreta:
     $$(f * g)(t) = \sum_{\tau} f(\tau) g(t - \tau)$$
     Las CNNs procesan ventanas locales fijas y carecen de un estado oculto recurrente dinámico autoregresivo.
   - En cambio, nuestro traductor utiliza **Celdas Recurrentes de Elman (SimpleRNN)** gobernadas por retroalimentación temporal continua:
     $$h_t = \tanh(W_{ih} x_t + b_{ih} + W_{hh} h_{t-1} + b_{hh})$$

2. **Lo que SÍ tiene: Ciclos Recurrentes Temporales (Time Steps):**
   En lugar de convoluciones, el modelo opera a través de **ciclos recurrentes temporales secuenciales (BPTT - Backpropagation Through Time)**:
   - **Ciclos Recurrentes del Encoder ($T_x$ ciclos):** Se ejecutan exactamente $T_x$ pasos recurrentes, uno por cada palabra de la oración en Quechua (típicamente entre 2 y 9 ciclos según la longitud de la frase).
   - **Ciclos Recurrentes del Decoder ($T_y$ ciclos):** Se ejecutan $T_y$ pasos recurrentes autoregresivos paso a paso en Español, hasta que el modelo decide emitir el token de fin de secuencia `<eos>` o alcanza el límite máximo (`max_len = 25` ciclos).

*(Nota: Aunque existen modelos de traducción basados en convoluciones como ConvS2S de Gehring et al., 2017, este proyecto fue desarrollado estrictamente bajo la especificación formal del requerimiento: **Redes Neuronales Recurrentes SimpleRNN Many-to-Many**).*

---

## 🔄 Todo el Proceso Paso a Paso que Sigue el Modelo (Pipeline End-to-End)

A continuación se detalla exhaustivamente la trayectoria completa que recorre una oración desde que el usuario la ingresa en Quechua hasta que se emite la traducción final en Español:

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             FLUJO COMPLETO DE PROCESAMIENTO (PIPELINE)                           │
└──────────────────────────────────────────────────────────────────────────────────────────────────┘

 [Entrada Quechua: "Allin punchaw, imataq sutiyki?"]
                     │
                     ▼
 [Fase 1: Normalización UTF-8 & Minúsculas] ──> "allin punchaw , imataq sutiyki ?"
                     │
                     ▼
 [Fase 2: Tokenización Léxica] ──> ["allin", "punchaw", ",", "imataq", "sutiyki", "?"]
                     │
                     ▼
 [Fase 3: Inserción de Tokens Especiales] ──> ["<sos>", "allin", "punchaw", ",", "imataq", "sutiyki", "?", "<eos>"]
                     │
                     ▼
 [Fase 4: Numericalización (Vocabulario)] ──> Tensor [2, 14, 28, 5, 42, 67, 8, 3]  (Tx = 8 tokens)
                     │
                     ▼
 [Fase 5: Capa de Embedding del Encoder] ──> Tensor [1, 8, 128] (vectores densos de 128 dimensiones)
                     │
                     ▼
 [Fase 6: Ciclos Recurrentes del Encoder (SimpleRNN con pack_padded_sequence)]
       t=1: h₁ = tanh(W·x₁ + U·h₀ + b)
       t=2: h₂ = tanh(W·x₂ + U·h₁ + b)
       ...
       t=8: h₈ = tanh(W·x₈ + U·h₇ + b)  ──>  Vector de Contexto c = h₈ ∈ ℝ²⁵⁶
                     │
                     ▼
 ══════════════════════════════════════════════════════════════════════════════════════════════════
                          PUENTE DE TRANSFERENCIA MANY-TO-MANY (Context Vector c)
 ══════════════════════════════════════════════════════════════════════════════════════════════════
                     │
                     ▼
 [Fase 7: Inicialización del Decoder SimpleRNN]
       - Estado oculto inicial del decoder: h₀_dec = c ∈ ℝ²⁵⁶
       - Primer token de entrada: "<sos>" (índice 2)
                     │
                     ▼
 [Fase 8: Ciclos Recurrentes Autoregresivos del Decoder (Paso a Paso con Contexto Persistente)]
       Paso t=1:
          • Entrada combinada: [ Embedding("<sos>") ; Contexto c ] ∈ ℝ³⁸⁴
          • Ciclo recurrente: h₁_dec = tanh(W_in·entrada + W_rec·h₀_dec + b)
          • Proyección Lineal: Logits = Linear(h₁_dec) ∈ ℝ²⁶³⁴ (distribución de vocabulario español)
          • Máscara de UNK: Logits[<unk>] = -1e9 (obliga a predecir palabras reales)
          • Selección Greedy: Token predicho = argmax(Logits) ──> "buenos"
       Paso t=2:
          • Entrada combinada: [ Embedding("buenos") ; Contexto c ] ∈ ℝ³⁸⁴
          • Ciclo recurrente: h₂_dec = tanh(W_in·entrada + W_rec·h₁_dec + b)
          • Selección Greedy: Token predicho = argmax(Logits) ──> "días"
       Paso t=3:
          • Entrada combinada: [ Embedding("días") ; Contexto c ] ∈ ℝ³⁸⁴
          • Selección Greedy: Token predicho = argmax(Logits) ──> ","
       Paso t=4:
          • Entrada combinada: [ Embedding(",") ; Contexto c ] ∈ ℝ³⁸⁴
          • Selección Greedy: Token predicho = argmax(Logits) ──> "¿"
       ...
       Paso t=Ty:
          • Selección Greedy: Token predicho = "<eos>" ──> ¡CRITERIO DE PARADA ALCANZADO!
                     │
                     ▼
 [Fase 9: Lista de Tokens Generados] ──> ["buenos", "días", ",", "¿", "cómo", "te", "llamas", "?"]
                     │
                     ▼
 [Fase 10: Detokenización y Post-procesamiento]
       • Unión inteligente de puntuación pegada (?, !)
       • Capitalización de primera letra
                     │
                     ▼
 [Salida Final en Español: "Buenos días, ¿cómo te llamas?"]
```

### Descripción Detallada de cada Fase del Proceso:

#### 1. Fase de Entrada y Normalización (`src/dataset_builder.py` / `src/preprocessor.py`)
- Se recibe el texto en Quechua introducido por el usuario o extraído del corpus.
- Se convierte a minúsculas y se limpian espacios en blanco redundantes.
- Se normaliza la codificación bajo el estándar **UTF-8 estricto**, preservando apóstrofes glotales (`p'unchaw, mawk'a`) y diacríticos.

#### 2. Fase de Tokenización Léxica
- Mediante expresiones regulares especializadas, se separan los signos de puntuación (`?`, `¿`, `!`, `¡`, `.`, `,`) de las palabras contiguas.
- Esto permite que la red neuronal aprenda el valor semántico de la palabra independientemente de si finaliza en coma, punto o signo de interrogación.

#### 3. Fase de Numericalización y Asignación de Tokens Especiales
- Se rodea la secuencia con los delimitadores de control: `<sos>` (*Start of Sequence*, ID 2) al principio y `<eos>` (*End of Sequence*, ID 3) al final.
- Cada palabra se transforma en su ID entero consultando el diccionario `word2idx` del vocabulario Quechua (compuesto por 3,356 palabras). Si una palabra no existe en el vocabulario, se le asigna el índice de `<unk>` (ID 1).

#### 4. Fase de Embeddings del Encoder
- La secuencia de números enteros entra a la matriz `nn.Embedding(3356, 128)`.
- Cada ID se convierte en un vector continuo denso de 128 dimensiones, capturando propiedades semánticas y sintácticas del Quechua.

#### 5. Ciclos Recurrentes del Encoder (SimpleRNN con `pack_padded_sequence`)
- La capa recurrente `nn.RNN(128, 256, batch_first=True, nonlinearity='tanh')` itera temporalmente token por token.
- **Mecanismo Anti-Acolchado:** Se aplica `pack_padded_sequence` con las longitudes reales de la oración. Esto impide que los ceros de relleno (`<pad>`) sean procesados por la RNN al final de la frase, garantizando que el estado oculto devuelto pertenezca exactamente al último token real (`<eos>`).
- El resultado es el **Vector de Contexto** $\mathbf{c} = h_{T_x}^{\text{enc}} \in \mathbb{R}^{256}$.

#### 6. Transferencia Many-to-Many al Decodificador
- El Vector de Contexto $\mathbf{c}$ condensa toda la semántica de la oración Quechua.
- Este vector se transfiere al Decoder como su estado oculto inicial: $h_0^{\text{dec}} = \mathbf{c}$.

#### 7. Ciclos Recurrentes Autoregresivos del Decodificador (SimpleRNN con Contexto Persistente)
- El decoder arranca recibiendo como primer token `<sos>`.
- En cada ciclo temporal $t$, se ejecuta la técnica de **persistencia de contexto** (Cho et al., 2014):
  $$\text{Entrada al RNN}_t = [ \text{Embedding}(y_{t-1}) \; ; \; \mathbf{c} ] \in \mathbb{R}^{128 + 256 = 384}$$
- Al recibir el vector de contexto en cada ciclo, la celda recurrente no olvida la oración en Quechua incluso después de varios pasos de generación.

#### 8. Proyección Lineal a Logits sobre el Vocabulario Español
- El estado oculto $h_t^{\text{dec}} \in \mathbb{R}^{256}$ pasa por una capa densa `nn.Linear(256, 2634)`.
- Produce un vector de 2,634 puntajes numéricos (logits), correspondientes a cada palabra del vocabulario en Español.

#### 9. Decodificación Greedy con Supresión de Tokens Desconocidos
- Se aplica una penalización estricta sobre el token desconocido: `Logits[<unk>] = -1e9`. Esto fuerza al decodificador a elegir palabras reales con contenido semántico.
- Se selecciona el índice de máxima probabilidad mediante $\hat{y}_t = \arg\max(\text{Logits})$.
- La palabra seleccionada se convierte en la entrada del siguiente ciclo temporal $t+1$.

#### 10. Criterio de Parada y Detokenización
- La generación autoregresiva continúa ciclo a ciclo hasta que ocurre una de dos condiciones:
  1. El modelo emite el token especial `<eos>` (*End of Sequence*).
  2. Se alcanza el límite máximo de seguridad (`max_len = 25` ciclos).
- La lista de tokens resultantes se procesa con `detokenize()`: se adhieren los signos de puntuación a sus respectivas palabras (`¿` se pega a la derecha, `?` se pega a la izquierda) y se capitaliza la primera letra.
- Se entrega la traducción fluida y limpia al usuario en la interfaz web de Streamlit.

---

## 🗂️ Recopilación y Preprocesamiento de Datos

### 1. Corpus Paralelo Consolidado
Se conformó un corpus paralelo con **2,745 pares de oraciones**, compuesto por:
1. **Corpus Conversacional Curado:** Saludos, despedidas, preguntas de identidad, familia, actividades diarias, comida, números, animales y naturaleza en Quechua Chanka/Cusco.
2. **Corpus AmericasNLP / Siminchik:** Oraciones breves (3 a 8 palabras) filtradas, normalizadas y depuradas del benchmark de traducción automática para lenguas originarias.

División del dataset:
- **Entrenamiento (Train):** 2,196 muestras (80%)
- **Validación (Val):** 274 muestras (10%)
- **Prueba (Test):** 275 muestras (10%)

### 2. Pipeline de Preprocesamiento (`src/preprocessor.py`)
- **Limpieza de Caracteres:** Manejo riguroso de codificación UTF-8 para preservar acentos en español (`á, é, í, ó, ú, ñ, ¿, ¡`) y apóstrofes glotales en quechua (`p'unchaw, mawk'a`).
- **Tokenización:** Separación de palabras y signos ortográficos en unidades léxicas independientes.
- **Tokens Especiales:**
  - `<pad>` (índice 0): Relleno para igualar longitud en mini-batches.
  - `<unk>` (índice 1): Representación de palabras desconocidas fuera de vocabulario (OOV).
  - `<sos>` (índice 2): Inicio de secuencia (*Start of Sequence*).
  - `<eos>` (índice 3): Fin de secuencia (*End of Sequence*).
- **Vocabularios:** Mapeo bidireccional palabra $\leftrightarrow$ índice.
  - Vocabulario Quechua: **3,356 tokens**
  - Vocabulario Español: **2,634 tokens**
- **Padding Dinámico y Control de Longitud:** Relleno adaptativo por batch con `pack_padded_sequence` para que los tokens de relleno `<pad>` no corrompan los estados ocultos.

---

## 🧠 Arquitectura de la Red Neuronal (Many-to-Many)

El modelo implementa una arquitectura **Encoder-Decoder con SimpleRNN** (celdas recurrentes básicas de Elman con activación $\tanh$):

### 1. Capa de Embedding
Mapea cada token discreto a un espacio vectorial denso continuo:
$$\mathbf{e}_t = \text{Embedding}(x_t) \in \mathbb{R}^{d_{\text{emb}}}$$

### 2. Encoder SimpleRNN
Procesa la secuencia Quechua $(x_1, x_2, \dots, x_{T_x})$. Cada paso recurrente calcula:
$$h_t^{\text{enc}} = \tanh(W_{ih} \mathbf{e}_t + b_{ih} + W_{hh} h_{t-1}^{\text{enc}} + b_{hh})$$

El estado del último token real $h_{T_x}^{\text{enc}}$ constituye el **Vector de Contexto** $\mathbf{c} \in \mathbb{R}^{d_{\text{hidden}}}$.

### 3. Decoder SimpleRNN Condicionado
Inicializa su estado con el vector de contexto:
$$h_0^{\text{dec}} = \mathbf{c}$$

En cada paso $t$, la celda recurrente recibe la concatenación del embedding del token previo y el vector de contexto $\mathbf{c}$ para evitar la pérdida de memoria en secuencias largas:
$$h_t^{\text{dec}} = \tanh\left(W_{\text{in}} [\mathbf{e}(y_{t-1}); \mathbf{c}] + W_{\text{rec}} h_{t-1}^{\text{dec}} + b\right)$$

### 4. Capa de Salida (Linear Projection)
Proyecta el estado oculto del decoder a las dimensiones del vocabulario español $V_{\text{esp}}$:
$$P(y_t = w \mid y_{<t}, x) = \text{softmax}(W_{\text{out}} h_t^{\text{dec}} + b_{\text{out}})_w$$

---

## ⚙️ Entrenamiento y Mitigación de Limitaciones de SimpleRNN

Las RNNs simples son propensas a:
1. **Desvanecimiento y Explosión de Gradientes (Vanishing / Exploding Gradients):**
   - **Solución 1:** Recorte de gradientes normados (`torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)`).
   - **Solución 2:** Inyección persistente del vector de contexto en cada paso del decoder.
   - **Solución 3:** Filtrado de oraciones concisas ($\le 10$ palabras) para mantener distancias temporales manejables.
2. **Corrupción por Tokens de Padding (`<pad>`):**
   - **Solución:** Uso de `pack_padded_sequence` en PyTorch, asegurando que el estado oculto final se extraiga exactamente en el último token real de cada frase.
3. **Teacher Forcing:**
   - Durante el entrenamiento, con probabilidad $p = 0.6$, el decoder se alimenta con el token real de la referencia (ground truth), estabilizando el aprendizaje autoregresivo.

---

## 📊 Evaluación y Métricas (BLEU Score)

La evaluación se realizó sobre el conjunto de prueba independiente (`data/test.csv`, 275 oraciones) mediante **BLEU (Bilingual Evaluation Understudy)** usando `nltk.translate.bleu_score` con suavizado exponencial (`SmoothingFunction.method1`):

| Métrica | Valor Obtenido | Descripción |
| :--- | :---: | :--- |
| **Corpus BLEU-1** | **31.00%** | Precisión de unigramas (palabras individuales correctas) |
| **Corpus BLEU-2** | **19.51%** | Cohesión y precisión de bigramas (pares de palabras) |
| **Corpus BLEU-4** | **9.58%** | Métrica estándar acumulativa de 4-gramas |
| **Promedio BLEU-1 por Oración** | **40.64%** | Promedio de precisión léxica por frase |
| **Coincidencias Exactas (Exact Match)** | **25.09%** | 69 de 275 oraciones traducidas de forma idéntica a la referencia humana |

### Ejemplos de Traducciones Generadas por el Modelo:
| Quechua (Entrada) | Predicción SimpleRNN | Español (Referencia) | Estado |
| :--- | :--- | :--- | :---: |
| `allin punchaw` | **Buenos días** | Buenos días | ✅ Exacto |
| `allillanchu` | **¿cómo estás?** | ¿cómo estás? | ✅ Exacto |
| `sulpayki` | **Gracias** | Gracias | ✅ Exacto |
| `paqarinkama` | **Hasta mañana** | Hasta mañana | ✅ Exacto |
| `tupananchiskama` | **Hasta volver a encontrarnos** | Hasta volver a encontrarnos | ✅ Exacto |
| `sutiyqa juanmi` | **Mi nombre es juan** | Mi nombre es juan | ✅ Exacto |
| `ñuqa mikuni` | **Yo como** | Yo como | ✅ Exacto |
| `pay mikun` | **Él come** | Él come | ✅ Exacto |
| `ñuqa yachani` | **Yo sé** | Yo sé | ✅ Exacto |
| `taytayqa llamkan` | **Mi padre trabaja** | Mi padre trabaja | ✅ Exacto |
| `mamayqa yanun` | **Mi madre cocina** | Mi madre cocina | ✅ Exacto |
| `wasiyman rini` | **Voy a mi casa** | Voy a mi casa | ✅ Exacto |
| `misi puñun` | **El gato duerme** | El gato duerme | ✅ Exacto |
| `allqu anchatam anyan`| **El perro ladra mucho** | El perro ladra mucho | ✅ Exacto |
| `yakuta upyani` | **Bebo agua** | Bebo agua | ✅ Exacto |
| `kusisqam kani` | **Estoy feliz** | Estoy feliz | ✅ Exacto |

---

## 📁 Estructura del Proyecto

```
d:\IA\RECUPERACION\
├── data\
│   ├── raw\
│   │   ├── train.es                   # Corpus crudo de respaldo
│   │   └── train.quy
│   ├── corpus_quechua_espanol.csv     # Dataset completo consolidado (2,745 filas)
│   ├── train.csv                      # Split de entrenamiento (80%, 2,196 filas)
│   ├── val.csv                        # Split de validación (10%, 274 filas)
│   └── test.csv                       # Split de prueba (10%, 275 filas)
├── models\
│   ├── simple_rnn_que_es.pt           # Pesos del mejor modelo entrenado
│   ├── vocab_que.json                 # Vocabulario Quechua (3,356 tokens)
│   ├── vocab_es.json                  # Vocabulario Español (2,634 tokens)
│   ├── train_history.json             # Historial de épocas, pérdida y perplejidad
│   └── evaluation_results.json        # Reporte de métricas BLEU en test.csv
├── src\
│   ├── __init__.py
│   ├── dataset_builder.py             # Descarga, curaduría, limpieza y splits
│   ├── preprocessor.py                # Tokenización, clase Vocabulary y Dataset
│   ├── rnn_model.py                   # SimpleRNN Encoder, Decoder y Seq2Seq
│   ├── train.py                       # Pipeline de entrenamiento con checkpointing
│   └── evaluate.py                    # Cálculo de BLEU y traducción de oraciones
├── app.py                             # Aplicación interactiva en Streamlit
├── requirements.txt                   # Dependencias del proyecto
└── README.md                          # Documentación técnica completa
```

---

## 🚀 Instalación y Ejecución

### 1. Prerrequisitos
- Python 3.10, 3.11, 3.12 o 3.13.
- Entorno de terminal (PowerShell, Command Prompt o Bash).

### 2. Clonar / Acceder al Proyecto
```powershell
cd d:\IA\RECUPERACION
```

### 3. Instalar Dependencias
```powershell
pip install -r requirements.txt
```

### 4. Ejecutar la Aplicación Streamlit
```powershell
streamlit run app.py
```
La aplicación se abrirá automáticamente en tu navegador web en `http://localhost:8501`.

---

## 🖥️ Uso de la Aplicación Streamlit

La interfaz web incluye 5 pestañas principales:

1. **🌐 Traductor en Vivo:**
   - Escribe cualquier frase en Quechua o pulsa los botones de ejemplos rápidos.
   - Presiona **"Traducir al Español"** para ver la salida instantánea.
   - Despliega la **Inspección Técnica** para ver la tokenización, los índices y el vector de contexto.

2. **📊 Evaluación & BLEU:**
   - Consulta los indicadores de **Corpus BLEU-1**, **BLEU-2**, **BLEU-4** y **Coincidencias Exactas**.
   - Explora la tabla interactiva de oraciones de prueba con filtros por coincidencia.
   - Visualiza el histograma de distribución de BLEU scores.

3. **⚙️ Reentrenamiento Interactivo:**
   - Modifica épocas, tasa de aprendizaje, dimensiones de embedding y estado oculto, y ratio de Teacher Forcing.
   - Inicia un reentrenamiento y observa la **gráfica de pérdida en tiempo real**.

4. **📚 Explorador del Corpus:**
   - Busca palabras o frases en el dataset paralelo.
   - Analiza las estadísticas y las distribuciones de longitud de oraciones.

5. **🧠 Arquitectura Many-to-Many & Teoría:**
   - Explicación matemática detallada con diagramas, ecuaciones $\tanh$, softmax, entropía cruzada y cálculo de BLEU.

---

## 👥 Tecnologías Utilizadas
- **Python 3.13**
- **PyTorch 2.14** (Embeddings, `nn.RNN`, `pack_padded_sequence`, tensores)
- **Streamlit 1.62** (Frontend interactivo y visualizaciones)
- **NLTK 3.10** (Métricas de traducción BLEU y suavizado)
- **Pandas & NumPy** (Manipulación y estructuración del corpus)
- **Matplotlib** (Gráficos estadísticos y curvas de aprendizaje)

