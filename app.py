import streamlit as st
import pandas as pd
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# =========================
# ⚙️ CONFIG
# =========================
st.set_page_config(page_title="Propuestas", layout="wide")

# =========================
# 🎨 ESTILO SOFT UI + ANIMACIONES
# =========================
st.markdown("""
<style>

/* Fondo general */
body {
    background-color: #e6e6e6;
}

/* Animación entrada */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(25px);}
    to {opacity: 1; transform: translateY(0);}
}

/* Tarjetas */
.card {
    background: #e6e6e6;
    border-radius: 25px;
    padding: 20px;
    margin-bottom: 25px;
    box-shadow: 10px 10px 20px #c5c5c5,
                -10px -10px 20px #ffffff;
    animation: fadeIn 0.8s ease forwards;
    transition: all 0.3s ease;
}

/* Hover */
.card:hover {
    transform: scale(1.04);
    box-shadow: 14px 14px 25px #b0b0b0,
                -14px -14px 25px #ffffff;
}

/* Botón */
.stButton>button {
    background-color: #222;
    color: white;
    border-radius: 15px;
    height: 55px;
    width: 100%;
    font-size: 18px;
    border: none;
    transition: all 0.3s ease;
}

.stButton>button:hover {
    background-color: #ff4b6e;
    transform: scale(1.05);
}

/* Input */
.stTextInput>div>div>input {
    border-radius: 15px;
    padding: 12px;
}

/* Título */
.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    animation: fadeIn 1s ease;
}

/* Subtítulo */
.subtitle {
    text-align: center;
    color: gray;
    margin-bottom: 30px;
    animation: fadeIn 1.2s ease;
}

</style>
""", unsafe_allow_html=True)

# =========================
# 💍 HEADER
# =========================
st.markdown('<div class="title">Propuestas Inteligentes</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Análisis con IA de ideas románticas</div>', unsafe_allow_html=True)

# =========================
# 📎 INPUT
# =========================
st.markdown('<div class="card">', unsafe_allow_html=True)
url = st.text_input("📎 Pega el link CSV de tu formulario")
st.markdown('</div>', unsafe_allow_html=True)

# =========================
# 🚀 BOTÓN
# =========================
if st.button("Analizar"):

    try:
        df = pd.read_csv(url)
        columna = df.columns[1]
        respuestas = df[columna].dropna().astype(str)

        if len(respuestas) == 0:
            st.warning("⚠️ No hay respuestas aún.")
            st.stop()

        # LIMPIEZA
        def limpiar(texto):
            return re.sub(r"[^\w\s]", "", texto.lower())

        respuestas_limpias = [limpiar(r) for r in respuestas]

        # STOPWORDS
        stopwords = [
            "de","la","que","el","en","y","a","los","del",
            "se","las","por","un","para","con","no","una"
        ]

        temas = []

        # =========================
        # 🧠 IA
        # =========================
        if len(respuestas_limpias) >= 2:

            vectorizer = TfidfVectorizer(
                stop_words=stopwords,
                ngram_range=(1,2)
            )

            X = vectorizer.fit_transform(respuestas_limpias)

            k = min(3, len(respuestas_limpias))
            modelo = KMeans(n_clusters=k, random_state=42)
            modelo.fit(X)

            orden = modelo.cluster_centers_.argsort()[:, ::-1]
            terminos = vectorizer.get_feature_names_out()

            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("🧠 Temas detectados")

            cols = st.columns(k)

            for i in range(k):
                palabras = [terminos[ind] for ind in orden[i, :5]]
                palabras = [p for p in palabras if len(p) > 3]

                with cols[i]:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.write(f"Grupo {i+1}")
                    st.write(", ".join(palabras))
                    st.markdown('</div>', unsafe_allow_html=True)

                temas.extend(palabras)

            st.markdown('</div>', unsafe_allow_html=True)

        # =========================
        # 📊 LÓGICA ORIGINAL
        # =========================
        categorias = {
            "lugar": ["cena", "picnic", "restaurante", "comida"],
            "detalle": ["ramo", "flores"],
            "ambiente": ["romantico", "sorpresa"],
            "extra": ["mariachi", "velas"]
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
        # 💡 RESULTADO FINAL
        # =========================
        st.markdown('<div class="card">', unsafe_allow_html=True)

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

            texto = " Una propuesta ideal sería "

            if len(partes) > 1:
                texto += ", ".join(partes[:-1]) + " y " + partes[-1]
            else:
                texto += partes[0]

            if top:
                texto += f", inspirada en {', '.join(top)}"

            st.success(texto)

        else:
            st.warning("No hay suficientes datos para generar propuesta.")

        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error: {e}")
