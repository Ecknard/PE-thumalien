"""
dashboard/pages/2_📊_Explorer.py
Page d'exploration des posts analysés en base de données.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import streamlit as st
import plotly.express as px
import pandas as pd

from src.database.db_connector import DatabaseConnector

st.set_page_config(page_title="Explorer — Thumalien", page_icon="📊", layout="wide")
st.markdown("# 📊 Explorer les Données")
st.markdown("Parcourez et filtrez tous les posts analysés par le pipeline.")

@st.cache_resource
def get_db():
    return DatabaseConnector()

@st.cache_data(ttl=60)
def load_posts(limit=500):
    db = get_db()
    return db.get_posts_with_predictions(limit=limit)

try:
    posts = load_posts()
except Exception as e:
    st.error(f"❌ Impossible de charger les données : {e}")
    st.stop()

if not posts:
    st.info("📭 Aucune donnée analysée. Lancez d'abord le pipeline.")
    st.stop()

df = pd.DataFrame(posts)

# ============================================================
# FILTRES SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("### 🎛️ Filtres")

    # Filtre fake/réel
    filter_label = st.selectbox("Classification", ["Tous", "Fake News", "Posts Réels"])

    # Filtre langue
    languages = ["Toutes"] + sorted(df["language"].dropna().unique().tolist())
    filter_lang = st.selectbox("Langue", languages)

    # Filtre crédibilité
    cred_min, cred_max = st.slider(
        "Score de crédibilité",
        min_value=0.0, max_value=1.0, value=(0.0, 1.0), step=0.05
    )

    # Filtre émotion
    if "emotion_label" in df.columns:
        emotions = ["Toutes"] + sorted(df["emotion_label"].dropna().unique().tolist())
        filter_emotion = st.selectbox("Émotion", emotions)
    else:
        filter_emotion = "Toutes"

    # Recherche texte
    search_text = st.text_input("🔍 Recherche dans le texte", placeholder="mot-clé...")

    st.divider()
    st.markdown(f"**Total chargé :** {len(df)} posts")

# ============================================================
# APPLICATION DES FILTRES
# ============================================================
df_filtered = df.copy()

if filter_label == "Fake News":
    df_filtered = df_filtered[df_filtered["is_fake"] == True]
elif filter_label == "Posts Réels":
    df_filtered = df_filtered[df_filtered["is_fake"] == False]

if filter_lang != "Toutes":
    df_filtered = df_filtered[df_filtered["language"] == filter_lang]

if "credibility_score" in df_filtered.columns:
    df_filtered = df_filtered[
        df_filtered["credibility_score"].between(cred_min, cred_max, inclusive="both")
        | df_filtered["credibility_score"].isna()
    ]

if filter_emotion != "Toutes" and "emotion_label" in df_filtered.columns:
    df_filtered = df_filtered[df_filtered["emotion_label"] == filter_emotion]

if search_text.strip():
    mask = df_filtered["text_original"].fillna("").str.contains(search_text, case=False, na=False)
    df_filtered = df_filtered[mask]

st.markdown(f"**{len(df_filtered)} posts** correspondent aux filtres")
st.divider()

# ============================================================
# GRAPHIQUES
# ============================================================
if not df_filtered.empty:
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Score de crédibilité par langue")
        df_lang = df_filtered[df_filtered["credibility_score"].notna()]
        if not df_lang.empty and "language" in df_lang.columns:
            fig = px.box(
                df_lang, x="language", y="credibility_score",
                color="language",
                color_discrete_map={"fr": "#667eea", "en": "#764ba2"},
                labels={"language": "Langue", "credibility_score": "Score de crédibilité"},
            )
            fig.update_layout(height=280, showlegend=False, margin=dict(t=10, b=10))
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("😊 Émotions vs Classification")
        if "emotion_label" in df_filtered.columns and "is_fake" in df_filtered.columns:
            df_emo = df_filtered[df_filtered["emotion_label"].notna()].copy()
            df_emo["label"] = df_emo["is_fake"].apply(lambda x: "Fake" if x else "Réel")
            if not df_emo.empty:
                emo_counts = df_emo.groupby(["emotion_label", "label"]).size().reset_index(name="count")
                fig2 = px.bar(
                    emo_counts, x="emotion_label", y="count", color="label",
                    color_discrete_map={"Fake": "#ef4444", "Réel": "#22c55e"},
                    barmode="group",
                    labels={"emotion_label": "Émotion", "count": "Nombre", "label": "Type"},
                )
                fig2.update_layout(height=280, margin=dict(t=10, b=10))
                st.plotly_chart(fig2, use_container_width=True)

    # Timeline si créated_at disponible
    if "created_at" in df_filtered.columns:
        df_time = df_filtered[df_filtered["created_at"].notna()].copy()
        if not df_time.empty:
            try:
                df_time["date"] = pd.to_datetime(df_time["created_at"]).dt.date
                daily = df_time.groupby(["date", df_time["is_fake"].map({True: "Fake", False: "Réel", None: "Non analysé"})]).size().reset_index(name="count")
                daily.columns = ["date", "type", "count"]
                st.subheader("📅 Évolution temporelle")
                fig3 = px.area(
                    daily, x="date", y="count", color="type",
                    color_discrete_map={"Fake": "#ef4444", "Réel": "#22c55e", "Non analysé": "#9ca3af"},
                )
                fig3.update_layout(height=250, margin=dict(t=10, b=10))
                st.plotly_chart(fig3, use_container_width=True)
            except Exception:
                pass

    st.divider()

    # ============================================================
    # TABLEAU DE DONNÉES
    # ============================================================
    st.subheader("📋 Tableau des posts")

    # Colonnes à afficher
    show_cols = []
    for col in ["author", "language", "text_original", "credibility_score", "is_fake", "emotion_label", "vader_compound", "created_at"]:
        if col in df_filtered.columns:
            show_cols.append(col)

    df_show = df_filtered[show_cols].copy()

    if "credibility_score" in df_show.columns:
        df_show["credibility_score"] = df_show["credibility_score"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "—"
        )
    if "is_fake" in df_show.columns:
        df_show["is_fake"] = df_show["is_fake"].apply(
            lambda x: "🔴 Fake" if x is True else ("🟢 Réel" if x is False else "—")
        )
    if "vader_compound" in df_show.columns:
        df_show["vader_compound"] = df_show["vader_compound"].apply(
            lambda x: f"{x:+.3f}" if pd.notna(x) else "—"
        )

    rename_map = {
        "author": "Auteur", "language": "Langue", "text_original": "Texte",
        "credibility_score": "Crédibilité", "is_fake": "Classification",
        "emotion_label": "Émotion", "vader_compound": "VADER", "created_at": "Date"
    }
    df_show.rename(columns={k: v for k, v in rename_map.items() if k in df_show.columns}, inplace=True)

    st.dataframe(df_show, use_container_width=True, hide_index=True, height=400)

    # Export CSV
    csv = df_filtered.to_csv(index=False, encoding="utf-8-sig")
    st.download_button(
        "⬇️ Exporter en CSV",
        data=csv,
        file_name="thumalien_export.csv",
        mime="text/csv",
    )
