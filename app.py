import streamlit as st
import pandas as pd
import re
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Dashboard Propuestas", layout="wide")

# =========================
# 🎨 ESTILO DASHBOARD SOFT UI
# =========================
st.markdown("""
<style>
body { background: #f4f7fb; }

.title {
    text-align:center;
    font-size:42px;
    font-weight:700;
    color:#1a1a1a;
}

.subtitle {
    text-align:center;
    color:#5f6b7a;
    margin-bottom:25px;
}

.card {
    background:#f4f7fb;
    padding:20px;
    border-radius:20px;
    box-shadow: 8px 8px 16px #d1d9e6,
                -8px -8px 16px #ffffff;
    margin-bottom:20px;
}

.stButton>button {
    background: linear-gradient(135deg, #007BFF, #00A8FF);
    color:white;
    border-radius:12px;
    height:50px;
}

section[data-testid="stSidebar"] {
    background:#f4f7fb;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR
# =========================
st.sidebar.title("⚙️ Panel")

url = st.sidebar.text_input("📂 CSV URL")
analizar = st.sidebar.button("🚀 Analizar")

# =========================
# HEADER
# =========================
st.markdown('<div class="title">📊 Dashboard de Propuestas</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">IA para detectar ideas y generar propuestas</div>', unsafe_allow_html=True)

# =========================
# FUNCIONES IA
# =========================
def limpiar(texto):
    texto = texto.lower()
    texto = re.sub(r"http\S+", "", texto)
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto

def obtener_mejor_k(X, max_k=5):
    mejor_k = 2
    mejor_score = -1
    
    for k in range(2, min(max_k, X.shape[0]) + 1):
        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = modelo.fit_predict(X)
        
        if len(set(labels)) > 1:
            score = silhouette_score(X, labels)
            if score > mejor_score:
                mejor_score = score
                mejor_k = k
                
    return mejor_k

# =========================
# EJECUCIÓN
# =========================
if analizar:

    try:
        df = pd.read_csv(url)
        columna = df.columns[1]
        respuestas = df[columna].dropna().astype(str)

        if len(respuestas) < 2:
            st.warning("No hay suficientes respuestas.")
            st.stop()

        # LIMPIEZA
        respuestas_limpias = [limpiar(r) for r in respuestas]

        # STOPWORDS MEJORADAS
        stopwords = [
            "de","la","que","el","en","y","a","los","del","se",
            "las","por","un","para","con","no","una","me","mi",
            "tu","te","es","lo","como","pero","más","ya"
        ]

        # VECTORIZACIÓN
        vectorizer = TfidfVectorizer(
            stop_words=stopwords,
            ngram_range=(1,2),
            min_df=1
        )

        X = vectorizer.fit_transform(respuestas_limpias)

        # 🔥 MEJOR K AUTOMÁTICO
        k = obtener_mejor_k(X)

        modelo = KMeans(n_clusters=k, random_state=42, n_init=10)
        modelo.fit(X)

        terminos = vectorizer.get_feature_names_out()
        orden = modelo.cluster_centers_.argsort()[:, ::-1]

        # =========================
        # MÉTRICAS
        # =========================
        col1, col2, col3 = st.columns(3)
        col1.metric("Respuestas", len(respuestas))
        col2.metric("Clusters", k)
        col3.metric("Keywords", len(terminos))

        # =========================
        # TEMAS
        # =========================
        st.markdown("### 🧠 Temas detectados")

        cols = st.columns(k)
        temas = []

        for i in range(k):
            palabras = [terminos[ind] for ind in orden[i, :6]]
            palabras = [p for p in palabras if len(p) > 3]

            temas.extend(palabras)

            with cols[i]:
                st.markdown(f"""
                <div class="card">
                <h4>Grupo {i+1}</h4>
                <p>{", ".join(palabras)}</p>
                </div>
                """, unsafe_allow_html=True)

        # =========================
        # ANÁLISIS SEMÁNTICO
        # =========================
        categorias = {
            "lugar": ["cena","restaurante","cafe","picnic"],
            "detalle": ["flores","regalo","carta"],
            "ambiente": ["romantico","sorpresa","especial"],
            "extra": ["musica","mariachi","velas"]
        }

        resultados = {c: [] for c in categorias}

        for r in respuestas_limpias:
            for cat, palabras in categorias.items():
                for p in palabras:
                    if p in r:
                        resultados[cat].append(p)

        final = {
            cat: Counter(lst).most_common(1)[0][0]
            for cat, lst in resultados.items() if lst
        }

        top = [p for p, _ in Counter(temas).most_common(3)]

        # =========================
        # RESULTADO FINAL
        # =========================
        st.markdown("### 💡 Propuesta generada")

        if final:
            partes = []

            if "lugar" in final:
                partes.append(f"una {final['lugar']}")
            if "detalle" in final:
                partes.append(f"con {final['detalle']}")
            if "ambiente" in final:
                partes.append(f"en un ambiente {final['ambiente']}")
            if "extra" in final:
                partes.append(f"incluyendo {final['extra']}")

            texto = "Una propuesta ideal sería "

            if len(partes) > 1:
                texto += ", ".join(partes[:-1]) + " y " + partes[-1]
            else:
                texto += partes[0]

            if top:
                texto += f", inspirada en {', '.join(top)}"

            st.markdown(f"""
            <div class="card" style="border-left: 5px solid #007BFF;">
            <p style="font-size:18px;">{texto}</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.warning("No se pudo generar propuesta.")

    except Exception as e:
        st.error(f"Error: {e}")
