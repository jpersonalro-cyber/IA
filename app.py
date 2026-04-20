import streamlit as st
import pandas as pd
from collections import Counter
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

st.set_page_config(page_title="Analizador de Propuestas", layout="centered")

st.title("💍 Analizador de Propuestas de Pareja")
st.write("Analiza respuestas de un formulario usando IA básica")

url = st.text_input("📎 Pega el link CSV de tu formulario")

if st.button("Analizar"):

    try:
        df = pd.read_csv(url)
        columna = df.columns[1]
        respuestas = df[columna].dropna().astype(str)

        if len(respuestas) == 0:
            st.warning("No hay respuestas aún.")
            st.stop()

        # LIMPIEZA
        def limpiar_texto(texto):
            texto = texto.lower()
            texto = re.sub(r"[^\w\s]", "", texto)
            return texto

        respuestas_limpias = [limpiar_texto(r) for r in respuestas]

        # STOPWORDS
        stopwords_es = [
            "de","la","que","el","en","y","a","los","del","se","las",
            "por","un","para","con","no","una","su","al","lo","como"
        ]

        # IA
        temas_detectados = []

        if len(respuestas_limpias) >= 2:

            vectorizer = TfidfVectorizer(
                stop_words=stopwords_es,
                ngram_range=(1,2)
            )

            X = vectorizer.fit_transform(respuestas_limpias)

            k = min(3, len(respuestas_limpias))
            modelo = KMeans(n_clusters=k, random_state=42)
            modelo.fit(X)

            orden = modelo.cluster_centers_.argsort()[:, ::-1]
            terminos = vectorizer.get_feature_names_out()

            st.subheader("🧠 Temas detectados")

            for i in range(k):
                palabras = [terminos[ind] for ind in orden[i, :5]]
                palabras = [p for p in palabras if len(p) > 3]
                temas_detectados.append(palabras)

                st.write(f"Grupo {i+1}: {', '.join(palabras)}")

        # LÓGICA ORIGINAL
        categorias = {
            "lugar": ["cena", "picnic", "restaurante", "comida"],
            "detalle": ["ramo", "flores"],
            "ambiente": ["romantico", "romántico", "sorpresa"],
            "extra": ["mariachi", "anillo", "velas"]
        }

        resultados = {cat: [] for cat in categorias}

        for r in respuestas_limpias:
            for cat, palabras in categorias.items():
                for p in palabras:
                    if p in r:
                        resultados[cat].append(p)

        final = {}
        for cat, lista in resultados.items():
            if lista:
                final[cat] = Counter(lista).most_common(1)[0][0]

        # FRASES IA
        frases = [p for grupo in temas_detectados for p in grupo]
        top = [p for p, _ in Counter(frases).most_common(3)]

        # RESULTADO FINAL
        st.subheader("💡 Propuesta ideal")

        if len(final) == 0:
            st.warning("No hay suficientes datos.")
        else:
            partes = []

            if "lugar" in final:
                partes.append(f"una {final['lugar']}")
            if "detalle" in final:
                partes.append(f"con un {final['detalle']}")
            if "ambiente" in final:
                partes.append(f"en un ambiente {final['ambiente']}")
            if "extra" in final:
                partes.append(f"incluyendo {final['extra']}")

            texto = "Una propuesta ideal podría ser "

            if len(partes) > 1:
                texto += ", ".join(partes[:-1]) + " y " + partes[-1]
            else:
                texto += partes[0]

            if top:
                texto += f", basada en ideas como {', '.join(top)}"

            st.success(texto + ".")

    except Exception as e:
        st.error(f"Error: {e}")