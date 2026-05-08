"""
dashboard/Home.py
Dashboard principal Thumalien — Détection de Fake News sur Bluesky
Page d'accueil avec KPIs globaux et dernières analyses.
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.database.db_connector import DatabaseConnector

# ============================================================
# CONFIG PAGE
# ============================================================
st.set_page_config(
    page_title="Thumalien — Fake News Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS CUSTOM
# ============================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #6b7280;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border-left: 4px solid #667eea;
    }
    .fake-badge {
        background: #fee2e2;
        color: #dc2626;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .real-badge {
        background: #dcfce7;
        color: #16a34a;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# CONNEXION DB
# ============================================================
@st.cache_resource
def get_db():
    return DatabaseConnector()


def safe_get_db():
    try:
        db = get_db()
        db.test_connection()
        return db, True
    except Exception as e:
        return None, str(e)


# ============================================================
# CHARGEMENT DONNÉES
# ============================================================
@st.cache_data(ttl=60)
def load_dashboard_data():
    db, ok = safe_get_db()
    if not ok or db is None:
        return None, None, None, str(ok)

    try:
        stats_posts = db.count_posts()
        stats_pred = db.get_predictions_stats()
        posts = db.get_posts_with_predictions(limit=500)
        return stats_posts, stats_pred, posts, None
    except Exception as e:
        return None, None, None, str(e)


# ============================================================
# HEADER
# ============================================================
st.markdown('<h1 class="main-title">🛡️ Thumalien</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Détection automatique de Fake News sur Bluesky '
    '— Pipeline NLP & IA multilingue (FR/EN)</p>',
    unsafe_allow_html=True
)

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/Bluesky_logo.svg/220px-Bluesky_logo.svg.png",
             width=80)
    st.markdown("### Navigation")
    st.markdown("""
    - 🏠 **Accueil** — KPIs & aperçu
    - 🔍 **Analyser** — Analyser un post
    - 📊 **Explorer** — Parcourir les données
    - 📈 **Métriques** — Performance du modèle
    - ⚡ **Énergie** — Bilan Green IT
    """)
    st.divider()
    st.markdown("**Projet Thumalien**")
    st.markdown("Mastère Big Data & IA")
    st.markdown("SUP DE VINCI — 2025/2026")

# ============================================================
# CHARGEMENT
# ============================================================
stats_posts, stats_pred, posts, error = load_dashboard_data()

if error:
    st.error(f"❌ Connexion à la base de données impossible : {error}")
    st.info("""
    **Comment résoudre :**
    1. Vérifiez que PostgreSQL est lancé
    2. Vérifiez votre fichier `.env`
    3. Lancez : `python -m src.database.db_connector`
    """)
    st.stop()

# ============================================================
# KPI CARDS
# ============================================================
st.subheader("📊 Vue d'ensemble")

col1, col2, col3, col4, col5 = st.columns(5)

total_posts = stats_posts.get("total", 0)
total_analyzed = stats_pred.get("total", 0)
fake_count = stats_pred.get("fake_count", 0)
real_count = stats_pred.get("real_count", 0)
avg_cred = stats_pred.get("avg_credibility", 0)

with col1:
    st.metric("Posts collectés", f"{total_posts:,}", help="Total posts en base de données")
with col2:
    st.metric("Posts analysés", f"{total_analyzed:,}", help="Posts avec prédiction IA")
with col3:
    fake_pct = fake_count / max(total_analyzed, 1) * 100
    st.metric("🔴 Fake News", f"{fake_count:,}", delta=f"{fake_pct:.1f}%",
              delta_color="inverse", help="Posts classifiés comme fake news")
with col4:
    st.metric("🟢 Posts réels", f"{real_count:,}", help="Posts classifiés comme fiables")
with col5:
    st.metric("Crédibilité moy.", f"{avg_cred:.1%}", help="Score moyen de crédibilité (0=fake, 1=réel)")

st.divider()

# ============================================================
# GRAPHIQUES
# ============================================================
if posts:
    df = pd.DataFrame(posts)
    df_analyzed = df[df["credibility_score"].notna()].copy()

    if not df_analyzed.empty:
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("🥧 Répartition Fake / Réel")
            fig_pie = px.pie(
                values=[fake_count, real_count],
                names=["Fake News", "Posts Réels"],
                color_discrete_map={"Fake News": "#ef4444", "Posts Réels": "#22c55e"},
                hole=0.45,
            )
            fig_pie.update_layout(
                showlegend=True, height=320,
                margin=dict(t=20, b=20, l=20, r=20),
                font=dict(size=13),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_right:
            st.subheader("📉 Distribution des scores de crédibilité")
            fig_hist = px.histogram(
                df_analyzed,
                x="credibility_score",
                nbins=20,
                color_discrete_sequence=["#667eea"],
                labels={"credibility_score": "Score de crédibilité"},
            )
            fig_hist.add_vline(x=0.5, line_dash="dash", line_color="red",
                               annotation_text="Seuil", annotation_position="top right")
            fig_hist.update_layout(height=320, margin=dict(t=30, b=20, l=20, r=20))
            st.plotly_chart(fig_hist, use_container_width=True)

        # Distribution des émotions
        if "emotion_label" in df_analyzed.columns:
            df_emotions = df_analyzed[df_analyzed["emotion_label"].notna()]
            if not df_emotions.empty:
                st.subheader("😊 Distribution des émotions")
                emotion_counts = df_emotions["emotion_label"].value_counts().reset_index()
                emotion_counts.columns = ["Émotion", "Nombre"]

                EMOTION_COLORS = {
                    "joy": "#facc15", "anger": "#ef4444", "fear": "#8b5cf6",
                    "sadness": "#3b82f6", "surprise": "#f97316",
                    "disgust": "#84cc16", "neutral": "#9ca3af",
                }
                colors = [EMOTION_COLORS.get(e, "#9ca3af") for e in emotion_counts["Émotion"]]

                fig_bar = px.bar(
                    emotion_counts, x="Émotion", y="Nombre",
                    color="Émotion",
                    color_discrete_sequence=colors,
                    labels={"Émotion": "Émotion", "Nombre": "Nombre de posts"},
                )
                fig_bar.update_layout(height=300, showlegend=False,
                                      margin=dict(t=20, b=20, l=20, r=20))
                st.plotly_chart(fig_bar, use_container_width=True)

        st.divider()

        # Derniers posts analysés
        st.subheader("🕐 Derniers posts analysés")

        display_cols = ["author", "text_original", "credibility_score", "is_fake",
                        "emotion_label", "language"]
        display_cols = [c for c in display_cols if c in df_analyzed.columns]

        df_display = df_analyzed[display_cols].head(20).copy()

        if "credibility_score" in df_display.columns:
            df_display["credibility_score"] = df_display["credibility_score"].apply(
                lambda x: f"{x:.1%}" if x is not None else "—"
            )
        if "is_fake" in df_display.columns:
            df_display["is_fake"] = df_display["is_fake"].apply(
                lambda x: "🔴 Fake" if x else "🟢 Réel"
            )

        df_display.columns = [c.replace("_", " ").title() for c in df_display.columns]
        st.dataframe(df_display, use_container_width=True, hide_index=True)

    else:
        st.info("📭 Aucun post analysé pour l'instant. Lancez le pipeline pour commencer.")

else:
    st.info("📭 Base de données vide. Lancez : `python -m src.pipeline`")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.markdown(
    "<center><small>Thumalien © 2025-2026 — SUP DE VINCI — Mastère Big Data & IA</small></center>",
    unsafe_allow_html=True
)
