"""
dataset_builder.py
Recopilación, filtrado y estructuración del corpus paralelo Quechua - Español.
Combina un corpus conversacional curado de alta calidad con frases seleccionadas
del benchmark AmericasNLP / Siminchik para entrenamiento de traducción automática.
"""

import os
import re
import random
import urllib.request
import pandas as pd

# Directorios de datos
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# 1. Corpus conversacional y cotidiano curado (Quechua Chanka / Collao - Español)
# -----------------------------------------------------------------------------
CURATED_PAIRS = [
    # Saludos y cortesía
    ("allillanchu", "¿cómo estás?"),
    ("allillanmi kani", "estoy bien"),
    ("allinllachu kachkanki", "¿estás bien?"),
    ("allinmi kachkani", "estoy bien"),
    ("allinllachu", "¿cómo estás?"),
    ("allin punchaw", "buenos días"),
    ("allin p'unchaw", "buenos días"),
    ("allin suka", "buenas tardes"),
    ("allin tuta", "buenas noches"),
    ("sulpayki", "gracias"),
    ("añay", "muchas gracias"),
    ("sulpayki taytay", "gracias señor"),
    ("sulpayki mamay", "gracias señora"),
    ("imamanta", "de nada"),
    ("tupananchiskama", "hasta volver a encontrarnos"),
    ("tupananchikkama", "hasta luego"),
    ("paqarinkama", "hasta mañana"),
    ("ripusun", "vamos"),
    ("haku ripusun", "vamos a irnos"),
    ("ama suwa", "no seas ladrón"),
    ("ama llulla", "no seas mentiroso"),
    ("ama qilla", "no seas ocioso"),

    # Identidad, origen y preguntas personales
    ("imataq sutiyki", "¿cuál es tu nombre?"),
    ("imataq sutiyki?", "¿cómo te llamas?"),
    ("sutiyqa juanmi", "mi nombre es juan"),
    ("sutiyqa mariam", "mi nombre es maría"),
    ("sutiyqa carmenmi", "mi nombre es carmen"),
    ("sutiyqa pedrom", "mi nombre es pedro"),
    ("maymantam kanki", "¿de dónde eres?"),
    ("maymantataq kanki?", "¿de dónde eres tú?"),
    ("perumantam kani", "soy de perú"),
    ("cuscomantam kani", "soy del cusco"),
    ("ayacuchomantam kani", "soy de ayacucho"),
    ("limamantam kani", "soy de lima"),
    ("pi kanki", "¿quién eres?"),
    ("ñuqa yachachiq kani", "yo soy profesor"),
    ("ñuqa yachaq kani", "yo soy estudiante"),
    ("ñuqa llamkaq kani", "yo soy trabajador"),
    ("ñuqa runa kani", "yo soy una persona"),
    ("mashka watayuqtaq kanki", "¿cuántos años tienes?"),
    ("chunka watayuqmi kani", "tengo diez años"),
    ("iskay chunka watayuqmi kani", "tengo veinte años"),

    # Pronombres y verbos básicos (presente)
    ("ñuqa rimani", "yo hablo"),
    ("qam rimanki", "tú hablas"),
    ("pay riman", "él habla"),
    ("ñuqanchik rimanchik", "nosotros hablamos"),
    ("paykuna rimanku", "ellos hablan"),
    ("ñuqa mikuni", "yo como"),
    ("qam mikunki", "tú comes"),
    ("pay mikun", "él come"),
    ("ñuqanchik mikunchik", "nosotros comemos"),
    ("paykuna mikunku", "ellos comen"),
    ("ñuqa puñuni", "yo duermo"),
    ("qam puñunki", "tú duermes"),
    ("pay puñun", "él duerme"),
    ("ñuqanchik puñunchik", "nosotros dormimos"),
    ("paykuna puñunku", "ellos duermen"),
    ("ñuqa takini", "yo canto"),
    ("qam takinki", "tú cantas"),
    ("pay takin", "él canta"),
    ("paykuna takinku", "ellos cantan"),
    ("ñuqa tusuni", "yo bailo"),
    ("qam tusunki", "tú bailas"),
    ("pay tusun", "él baila"),
    ("paykuna tusunku", "ellos bailan"),
    ("ñuqa llamkani", "yo trabajo"),
    ("qam llamkanki", "tú trabajas"),
    ("pay llamkan", "él trabaja"),
    ("ñuqanchik llamkanchik", "nosotros trabajamos"),
    ("paykuna llamkanku", "ellos trabajan"),
    ("ñuqa yachani", "yo sé"),
    ("qam yachanki", "tú sabes"),
    ("pay yachan", "él sabe"),
    ("mana yachanichu", "no lo sé"),
    ("arí yachanimi", "sí lo sé"),

    # Familia y seres queridos
    ("taytayqa llamkan", "mi padre trabaja"),
    ("mamayqa yanun", "mi madre cocina"),
    ("taytayqa allin runam", "mi padre es buena persona"),
    ("mamayqa kuyasqam", "mi madre es amada"),
    ("wawqiyqa pukllan", "mi hermano juega"),
    ("panayqa takin", "mi hermana canta"),
    ("churiymi yachaywasiman rin", "mi hijo va a la escuela"),
    ("ususiyqa tusun", "mi hija baila"),
    ("abueloyqa kawsan", "mi abuelo vive"),
    ("abuelayqa willawaran", "mi abuela me contó"),
    ("warmachaqa waqan", "el niño llora"),
    ("warmakuna pukllanku", "los niños juegan"),
    ("runakunaqa llamkanku", "las personas trabajan"),
    ("warmikunaqa takinku", "las mujeres cantan"),
    ("qharikunaqa purinku", "los hombres caminan"),

    # Comida, bebida y necesidades
    ("yakuta upyani", "bebo agua"),
    ("qam yakuta upyanki", "tú bebes agua"),
    ("pay yakuta upyan", "él bebe agua"),
    ("tantata mikuni", "como pan"),
    ("tantata rantini", "compro pan"),
    ("aychata yanuni", "cocino carne"),
    ("papata mikuni", "como papa"),
    ("sarata tarpuni", "siembro maíz"),
    ("sarata mikunku", "comen maíz"),
    ("mikuyta munani", "quiero comer"),
    ("yakuta munani", "quiero agua"),
    ("puñuyta munani", "quiero dormir"),
    ("samayta munani", "quiero descansar"),
    ("yarqawaschkankim", "tengo hambre"),
    ("yakunayawachkan", "tengo sed"),
    ("chiriyawachkan", "tengo frío"),
    ("ruphayawachkan", "tengo calor"),

    # Lugares, direcciones y movimiento
    ("wasiyman rini", "voy a mi casa"),
    ("wasiman rini", "voy a la casa"),
    ("yachaywasiman rini", "voy a la escuela"),
    ("yachaywasipi kani", "estoy en la escuela"),
    ("llaqtaman risunchik", "vamos al pueblo"),
    ("llaqtapi tiyani", "vivo en el pueblo"),
    ("urqupi kani", "estoy en el cerro"),
    ("mayupi kani", "estoy en el río"),
    ("mayuman rini", "voy hacia el río"),
    ("kaypi kachkani", "estoy aquí"),
    ("chaypi kachkan", "está allí"),
    ("maypim kachkanki", "¿dónde estás?"),
    ("maytam rinki", "¿a dónde vas?"),
    ("maymantam hamunki", "¿de dónde vienes?"),
    ("wasimantam hamuni", "vengo de la casa"),
    ("hamuy kayman", "ven aquí"),
    ("ripuy chayman", "vete para allá"),
    ("kutimusaq", "voy a volver"),
    ("paqarin kutimusaq", "mañana volveré"),

    # Animales y naturaleza
    ("allqu anchatam anyan", "el perro ladra mucho"),
    ("allqu aychata mikun", "el perro come carne"),
    ("misi puñun", "el gato duerme"),
    ("misi anchatam puñun", "el gato duerme mucho"),
    ("misi pukllan", "el gato juega"),
    ("pishqu pawan", "el ave vuela"),
    ("pisqukuna takinku", "los pájaros cantan"),
    ("kawallu phawan", "el caballo corre"),
    ("llama purin", "la llama camina"),
    ("uywakunata michini", "pastoreo los animales"),
    ("inti kanchamun", "el sol brilla"),
    ("inti llikllata qhawan", "el sol mira el manto"),
    ("para chayamun", "llega la lluvia"),
    ("para paramun", "está lloviendo"),
    ("wayra wayramun", "el viento sopla"),
    ("urquqa hatunmi", "la montaña es grande"),
    ("mayuqa purin", "el río fluye"),
    ("sachaqa qumer", "el árbol es verde"),
    ("tikaqa munaymi", "la flor es bonita"),
    ("killa kanchamun", "la luna brilla"),
    ("quyllurkuna kanchanku", "las estrellas brillan"),

    # Estados, emociones y adjetivos
    ("kusisqam kani", "estoy feliz"),
    ("kusisqam kachkanki", "estás alegre"),
    ("llakisqam kani", "estoy triste"),
    ("saykuskam kani", "estoy cansado"),
    ("onqosqam kani", "estoy enfermo"),
    ("allillanmi kachkani", "me encuentro bien"),
    ("sumaq punchaw", "día hermoso"),
    ("sumaq sipas", "joven hermosa"),
    ("allin runa", "buen hombre"),
    ("hatun wasi", "casa grande"),
    ("uchuk wasi", "casa pequeña"),
    ("chiri yaku", "agua fría"),
    ("rupha yaku", "agua caliente"),
    ("musuq pacha", "ropa nueva"),
    ("mawk'a wasi", "casa vieja"),

    # Preguntas cotidianas y respuestas
    ("imataq kay", "¿qué es esto?"),
    ("chayqa librom", "eso es un libro"),
    ("imataq chay", "¿qué es eso?"),
    ("kayqa allqum", "esto es un perro"),
    ("imata ruwachkanki", "¿qué estás haciendo?"),
    ("qillqachkani", "estoy escribiendo"),
    ("ñawinchachkani", "estoy leyendo"),
    ("llamkachkani", "estoy trabajando"),
    ("mikuchkani", "estoy comiendo"),
    ("imanasqataq waqanki", "¿por qué lloras?"),
    ("kusisqa kaspaymi takini", "porque estoy alegre canto"),
    ("haykapim hamunki", "¿cuándo vendrás?"),
    ("minchham hamusaq", "pasado mañana vendré"),
    ("hayk'aqmi tupasun", "¿cuándo nos veremos?"),
    ("paqarinmi tupasun", "mañana nos veremos"),
    ("arí munanim", "sí lo quiero"),
    ("manam munanichu", "no lo quiero"),
    ("manam atinichu", "no puedo hacerlo"),
    ("atillanimi", "sí puedo hacerlo"),

    # Tiempo y números
    ("kunan punchaw", "el día de hoy"),
    ("paqarin tuta", "mañana en la noche"),
    ("qayna punchaw", "el día de ayer"),
    ("huk punchaw", "un solo día"),
    ("iskay allqukuna", "dos perros"),
    ("kimsa misikuna", "tres gatos"),
    ("tawa runakuna", "cuatro personas"),
    ("pichqa sachakuna", "cinco árboles"),
]

def clean_text(text: str) -> str:
    """Normaliza y limpia el texto para entrenamiento."""
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    # Eliminar múltiples espacios y saltos de línea
    text = re.sub(r'\s+', ' ', text)
    # Conservar letras estándar, signos básicos y apóstrofes usados en Quechua
    text = re.sub(r"[^a-záéíóúñü\s\.,\?!¿¡'\-]", "", text)
    text = text.strip()
    return text

def download_and_filter_americasnlp(max_pairs=2200):
    """
    Descarga el corpus paralelo de AmericasNLP / JW300, lo decodifica correctamente
    y filtra oraciones cortas de alta calidad para entrenamiento con RNN simple.
    """
    url_es = "https://raw.githubusercontent.com/Helsinki-NLP/americasnlp2021-st/main/data/quechua-spanish/train.es"
    url_qu = "https://raw.githubusercontent.com/Helsinki-NLP/americasnlp2021-st/main/data/quechua-spanish/train.quy"
    
    local_es = os.path.join(RAW_DIR, "train.es")
    local_qu = os.path.join(RAW_DIR, "train.quy")
    
    if not os.path.exists(local_es):
        print(f"Descargando {url_es} ...")
        urllib.request.urlretrieve(url_es, local_es)
    if not os.path.exists(local_qu):
        print(f"Descargando {url_qu} ...")
        urllib.request.urlretrieve(url_qu, local_qu)
        
    print("Procesando y filtrando oraciones del corpus externo...")
    # Ambos archivos están codificados en UTF-8 nativo
    with open(local_es, "r", encoding="utf-8", errors="ignore") as f_es, \
         open(local_qu, "r", encoding="utf-8", errors="ignore") as f_qu:
        es_lines = f_es.readlines()
        qu_lines = f_qu.readlines()

    collected = []
    seen = set()
    
    # Patrón de exclusión de texto bibliográfico / URLs / caracteres anómalos
    exclude_pattern = re.compile(r'(www\.|http|watch tower|jw\.org|g9|g0|pág|capítulo|versículo|copyright|\d{4})', re.IGNORECASE)

    for es_raw, qu_raw in zip(es_lines, qu_lines):
        es = clean_text(es_raw)
        qu = clean_text(qu_raw)
        
        if not es or not qu:
            continue
        if exclude_pattern.search(es) or exclude_pattern.search(qu):
            continue
        if '\ufffd' in es or '\ufffd' in qu:
            continue
            
        words_es = es.split()
        words_qu = qu.split()
        
        # Filtro de longitud para RNN Simple: 3 a 8 palabras
        # Esto permite que una RNN clásica aprenda sin desvanecimiento de gradiente severo
        if 2 <= len(words_qu) <= 8 and 3 <= len(words_es) <= 9:
            # Evitar oraciones donde el quechua y español son idénticos
            if es == qu:
                continue
            pair_key = (qu, es)
            if pair_key not in seen:
                seen.add(pair_key)
                collected.append((qu, es))
                
        if len(collected) >= max_pairs:
            break
            
    print(f"Oraciones filtradas del corpus externo: {len(collected)}")
    return collected

def build_dataset():
    """Genera el dataset consolidado y crea los splits de entrenamiento, validación y prueba."""
    print("Iniciando construcción del dataset Quechua-Español...")
    
    # Pares curados (ponderados para balancear patrones conversacionales y de alta frecuencia)
    cleaned_curated = []
    seen = set()
    for q, e in CURATED_PAIRS:
        cq = clean_text(q)
        ce = clean_text(e)
        if cq and ce:
            # Replicar 5 veces para que la red recurrente aprenda las frases y sufijos conversacionales
            for _ in range(5):
                cleaned_curated.append({"quechua": cq, "espanol": ce, "fuente": "curado"})

    # Pares externos de AmericasNLP
    try:
        external_pairs = download_and_filter_americasnlp(max_pairs=1800)
    except Exception as e:
        print(f"Advertencia al descargar corpus externo: {e}. Se usarán solo los pares curados y sintéticos.")
        external_pairs = []

    cleaned_external = []
    for q, e in external_pairs:
        if (q, e) not in seen:
            seen.add((q, e))
            cleaned_external.append({"quechua": q, "espanol": e, "fuente": "americas_nlp"})

    # Combinación total
    all_pairs = cleaned_curated + cleaned_external
    df = pd.DataFrame(all_pairs)
    
    # Barajar de forma reproducible
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    total = len(df)
    print(f"Total de pares reunidos: {total}")
    
    # Guardar corpus completo
    full_path = os.path.join(DATA_DIR, "corpus_quechua_espanol.csv")
    df.to_csv(full_path, index=False, encoding="utf-8")
    print(f"Guardado corpus completo en: {full_path}")
    
    # Split: 80% Train, 10% Val, 10% Test
    n_train = int(total * 0.80)
    n_val = int(total * 0.10)
    
    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()
    
    train_path = os.path.join(DATA_DIR, "train.csv")
    val_path = os.path.join(DATA_DIR, "val.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
    
    print(f"Splits generados exitosamente:")
    print(f"  - Train: {len(train_df)} muestras ({train_path})")
    print(f"  - Val:   {len(val_df)} muestras ({val_path})")
    print(f"  - Test:  {len(test_df)} muestras ({test_path})")
    
    return full_path, train_path, val_path, test_path

if __name__ == "__main__":
    build_dataset()
