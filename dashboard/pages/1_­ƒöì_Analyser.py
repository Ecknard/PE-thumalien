"""
dashboard/pages/1_🔍_Analyser.py
Page d'analyse en temps réel d'un post Bluesky.
"""
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from src.classifier.fake_news_classifier import FakeNewsClassifier
from src.emotion.emotion_analyzer import EmotionAnalyzer
from src.explainability.explainer import FakeNewsExplainer
from src.database.db_connector import DatabaseConnector

st.set_page_config(page_title="Analyser — Thumalien", page_icon="🔍", layout="wide")

st.markdown("# 🔍 Analyser un Post")
st.markdown("Entrez un texte pour obtenir une analyse complète : crédibilité, émotion et explicabilité.")

# ============================================================
# CHARGEMENT DES MODÈLES (en cache)
# ============================================================
@st.cache_resource
def load_models():
    clf = FakeNewsClassifier(prefer_bert=True)
    try:
        clf.load()
    except Exception:
        # Fallback : pas de modèle entraîné
        pass
    return clf, EmotionAnalyzer(), FakeNewsExplainer()


with st.spinner("Chargement des modèles IA..."):
    classifier, emotion_analyzer, explainer = load_models()

# ============================================================
# INPUT
# ============================================================
st.divider()

col_input, col_options = st.columns([3, 1])

with col_input:
    user_text = st.text_area(
        "📝 Texte du post Bluesky",
        height=130,
        placeholder="Collez ici le texte du post à analyser...\n\nExemple : 'URGENT !!! Le gouvernement cache la vérité sur ce scandale incroyable !!!'",
        max_chars=2000,
    )

with col_options:
    st.markdown("**Options**")
    language = st.selectbox("Langue", ["fr", "en"], help="Langue du texte")
    save_to_db = st.checkbox("Sauvegarder en DB", value=False)

    st.markdown("")
    analyze_btn = st.button("🚀 Analyser", type="primary", use_container_width=True)

# ============================================================
# EXEMPLES RAPIDES
# ============================================================
with st.expander("💡 Exemples rapides", expanded=False):
    examples = {
        "🔴 Fake (FR)": "URGENT !!! Le gouvernement cache la vérité sur ce scandale incroyable ! Partagez avant censure !!!",
        "🔴 Fake (EN)": "BREAKING: Scientists reveal they've been lying to us for decades! The media won't show this!",
        "🟢 Réel (FR)": "Rapport officiel des autorités sanitaires disponible en ligne selon le ministère de la santé.",
        "🟢 Réel (EN)": "New peer-reviewed study published in Nature explores climate adaptation strategies in coastal regions.",
    }
    cols = st.columns(4)
    for i, (label, text) in enumerate(examples.items()):
        if cols[i].button(label, use_container_width=True):
            st.session_state["example_text"] = text
            st.rerun()

if "example_text" in st.session_state:
    user_text = st.session_state.pop("example_text")

# ============================================================
# ANALYSE
# ============================================================
if analyze_btn and user_text.strip():
    st.divider()

    with st.spinner("Analyse en cours..."):
        # Classification
        pred = classifier.predict_one(user_text)

        # Émotion
        emotion = emotion_analyzer.analyze(user_text, language=language)

        # Explicabilité
        explanation = explainer.explain(user_text, pred["credibility_score"])

    # ----------------------------------------------------------
    # RÉSULTATS PRINCIPAUX
    # ----------------------------------------------------------
    is_fake = pred["is_fake"]
    credibility = pred["credibility_score"]

    if is_fake:
        st.error(f"🔴 **FAKE NEWS DÉTECTÉE** — Score de crédibilité : {credibility:.1%}")
    elif credibility < 0.65:
        st.warning(f"🟠 **CONTENU SUSPECT** — Score de crédibilité : {credibility:.1%}")
    else:
        st.success(f"🟢 **POST FIABLE** — Score de crédibilité : {credibility:.1%}")

    # Résumé de l'explication
    st.markdown(f"**Analyse :** {explanation['summary']}")

    st.divider()

    # ----------------------------------------------------------
    # MÉTRIQUES
    # ----------------------------------------------------------
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=credibility * 100,
            title={"text": "Crédibilité (%)"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#22c55e" if not is_fake else "#ef4444"},
                "steps": [
                    {"range": [0, 40], "color": "#fee2e2"},
                    {"range": [40, 65], "color": "#fef3c7"},
                    {"range": [65, 100], "color": "#dcfce7"},
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "thickness": 0.75,
                    "value": 50,
                },
            },
            number={"suffix": "%", "font": {"size": 28}},
        ))
        fig_gauge.update_layout(height=220, margin=dict(t=30, b=10, l=20, r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.metric("🤖 Confiance IA", f"{pred['confidence']:.1%}")
        st.metric("😊 Émotion", emotion["emotion_name"])
        st.metric("📊 Sentiment VADER", f"{emotion['vader_compound']:+.3f}")

    with col3:
        st.metric("⚠️ Score risque signaux", f"{explanation['signal_risk_score']:.1%}")
        n_signals = len(explanation["fake_signals"])
        n_cred = len(explanation["credibility_signals"])
        st.metric("🔴 Signaux suspects", n_signals)
        st.metric("🟢 Signaux crédibilité", n_cred)

    with col4:
        # Radar des émotions
        emotions_scores = emotion.get("all_scores", {})
        if emotions_scores:
            fig_radar = go.Figure(go.Scatterpolar(
                r=list(emotions_scores.values()),
                theta=list(emotions_scores.keys()),
                fill="toself",
                fillcolor="rgba(102, 126, 234, 0.2)",
                line=dict(color="#667eea"),
            ))
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=False, height=220,
                margin=dict(t=20, b=20, l=30, r=30),
                title="Profil émotionnel",
            )
            st.plotly_chart(fig_radar, use_container_width=True)

    # ----------------------------------------------------------
    # EXPLICABILITÉ DÉTAILLÉE
    # ----------------------------------------------------------
    st.divider()
    st.subheader("🔍 Pourquoi cette classification ?")

    col_fake, col_cred = st.columns(2)

    with col_fake:
        st.markdown("**🔴 Signaux suspects détectés**")
        if explanation["fake_signals"]:
            for signal_name, info in explanation["fake_signals"].items():
                with st.container():
                    pct = int(info["weight"] * 100)
                    st.markdown(
                        f"**{signal_name.replace('_', ' ').title()}** "
                        f"_(poids: {info['weight']:.0%})_"
                    )
                    st.markdown(f"_{info['description']}_")
                    if info.get("matches"):
                        st.code(", ".join(str(m) for m in info["matches"][:3]))
                    st.progress(min(info["weight"], 1.0))
        else:
            st.success("✅ Aucun signal suspect détecté")

    with col_cred:
        st.markdown("**🟢 Signaux de crédibilité détectés**")
        if explanation["credibility_signals"]:
            for signal_name, info in explanation["credibility_signals"].items():
                st.markdown(f"**{signal_name.replace('_', ' ').title()}**")
                st.markdown(f"_{info['description']}_")
                if info.get("matches"):
                    st.code(", ".join(str(m) for m in info["matches"][:3]))
        else:
            st.info("ℹ️ Aucun signal de crédibilité détecté")

    # ----------------------------------------------------------
    # SAUVEGARDE
    # ----------------------------------------------------------
    if save_to_db:
        try:
            db = DatabaseConnector()
            import hashlib
            fake_id = "manual_" + hashlib.md5(user_text.encode()).hexdigest()[:12]
            db.insert_posts([{
                "id": fake_id,
                "author": "manual_analysis",
                "author_name": "Analyse manuelle",
                "text": user_text,
                "language": language,
                "like_count": 0,
                "reply_count": 0,
                "repost_count": 0,
            }])
            db.upsert_prediction(fake_id, credibility, is_fake, pred["confidence"])
            db.upsert_emotion(
                fake_id, emotion["emotion_label"],
                emotion["emotion_score"], emotion["vader_compound"]
            )
            st.toast("✅ Analyse sauvegardée en base de données", icon="💾")
        except Exception as e:
            st.warning(f"Sauvegarde DB impossible : {e}")

elif analyze_btn:
    st.warning("⚠️ Veuillez saisir un texte à analyser.")
