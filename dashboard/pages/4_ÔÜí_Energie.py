"""
dashboard/pages/4_⚡_Energie.py
Bilan Green IT — suivi de la consommation énergétique du pipeline.
"""
import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parents[2]))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

st.set_page_config(page_title="Énergie — Thumalien", page_icon="⚡", layout="wide")
st.markdown("# ⚡ Bilan Green IT")
st.markdown("Suivi de la consommation énergétique du pipeline IA — engagement éco-responsable.")

from config import MODELS_DIR

ENERGY_REPORT_PATH = MODELS_DIR / "energy_report.json"

# ============================================================
# CHARGEMENT RAPPORT
# ============================================================
history = []

if ENERGY_REPORT_PATH.exists():
    with open(ENERGY_REPORT_PATH) as f:
        history = json.load(f)

# Aussi depuis la DB
try:
    from src.database.db_connector import DatabaseConnector
    db = DatabaseConnector()
    db_report = db.get_energy_report()
    db_available = True
except Exception:
    db_report = {}
    db_available = False

# ============================================================
# KPIs GLOBAUX
# ============================================================
valid = [r for r in history if r.get("emissions_kg") is not None]
total_emissions_g = sum(r["emissions_kg"] for r in valid) * 1000
total_wh = sum(r.get("energy_kwh", 0) for r in valid) * 1000
total_duration_min = sum(r.get("duration_sec", 0) for r in history) / 60
n_ops = len(history)
co2_km = total_emissions_g / 120 if total_emissions_g else 0  # 120g CO2/km voiture

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Opérations tracées", n_ops)
col2.metric("CO₂ total", f"{total_emissions_g:.4f} g CO₂eq")
col3.metric("Énergie totale", f"{total_wh:.4f} Wh")
col4.metric("Temps total", f"{total_duration_min:.1f} min")
col5.metric("Équiv. voiture", f"{co2_km:.6f} km")

st.divider()

# ============================================================
# CONTEXTE ET OBJECTIFS GREEN IT
# ============================================================
with st.expander("📚 Contexte Green IT & Objectifs du projet", expanded=True):
    st.markdown("""
    ### Pourquoi mesurer l'impact énergétique ?

    L'IA consomme des ressources énergétiques significatives — particulièrement lors de
    l'entraînement de grands modèles de langage. Dans le cadre du projet Thumalien,
    nous nous engageons à **mesurer, documenter et minimiser** cet impact.

    **Outil utilisé :** [CodeCarbon](https://codecarbon.io/) — mesure les émissions CO₂
    en temps réel en fonction du mix énergétique du pays.

    **Nos choix éco-responsables :**
    - Utilisation de **DistilBERT** (modèle distillé, 2x moins énergivore que BERT)
    - Entraînement sur CPU plutôt que GPU cloud (mix énergétique FR plus propre)
    - Batch processing pour éviter les appels API redondants
    - Mise en cache des prédictions en base de données PostgreSQL

    **Comparaison indicative :**
    | Opération | CO₂ estimé |
    |-----------|-----------|
    | 1 recherche Google | ~0.2 g CO₂ |
    | Entraînement GPT-3 | ~552 tonnes CO₂ |
    | Notre pipeline complet | < 10 g CO₂ |
    """)

if not history:
    st.info("📭 Aucune mesure énergétique disponible. Lancez le pipeline pour collecter des données.")
    st.code("python -m src.pipeline --from-file data/raw/votre_fichier.json")
    st.stop()

# ============================================================
# GRAPHIQUES
# ============================================================
df = pd.DataFrame(history)

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("🕐 Durée par opération")
    fig_dur = px.bar(
        df, x="operation", y="duration_sec",
        color="operation",
        color_discrete_sequence=px.colors.qualitative.Set2,
        labels={"operation": "Opération", "duration_sec": "Durée (s)"},
    )
    fig_dur.update_layout(height=280, showlegend=False, margin=dict(t=10, b=10))
    st.plotly_chart(fig_dur, use_container_width=True)

with col_b:
    if valid:
        st.subheader("💨 Émissions CO₂ par opération (g)")
        df_valid = pd.DataFrame(valid).copy()
        df_valid["emissions_g"] = df_valid["emissions_kg"] * 1000
        fig_co2 = px.bar(
            df_valid, x="operation", y="emissions_g",
            color="operation",
            color_discrete_sequence=px.colors.qualitative.Pastel,
            labels={"operation": "Opération", "emissions_g": "CO₂ (g)"},
        )
        fig_co2.update_layout(height=280, showlegend=False, margin=dict(t=10, b=10))
        st.plotly_chart(fig_co2, use_container_width=True)
    else:
        st.info("CodeCarbon non disponible — les durées sont tracées mais pas les émissions CO₂.")

# ============================================================
# TIMELINE
# ============================================================
if "timestamp" in df.columns:
    st.subheader("📅 Historique des opérations")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df_sorted = df.sort_values("timestamp")

    fig_timeline = px.scatter(
        df_sorted,
        x="timestamp",
        y="duration_sec",
        color="operation",
        size="duration_sec",
        hover_data=["operation", "n_samples", "model_name"],
        labels={"timestamp": "Date/Heure", "duration_sec": "Durée (s)"},
    )
    fig_timeline.update_layout(height=280, margin=dict(t=20, b=10))
    st.plotly_chart(fig_timeline, use_container_width=True)

st.divider()

# ============================================================
# TABLEAU DÉTAILLÉ
# ============================================================
st.subheader("📋 Détail des mesures")

display_cols = [c for c in ["timestamp", "operation", "duration_sec", "emissions_kg", "energy_kwh", "model_name", "n_samples"] if c in df.columns]
df_display = df[display_cols].copy()

if "emissions_kg" in df_display.columns:
    df_display["emissions_kg"] = df_display["emissions_kg"].apply(
        lambda x: f"{x*1000:.6f} g" if pd.notna(x) and x is not None else "N/A"
    )
if "energy_kwh" in df_display.columns:
    df_display["energy_kwh"] = df_display["energy_kwh"].apply(
        lambda x: f"{x*1000:.4f} Wh" if pd.notna(x) and x is not None else "N/A"
    )
if "duration_sec" in df_display.columns:
    df_display["duration_sec"] = df_display["duration_sec"].apply(lambda x: f"{x:.2f}s")

rename = {
    "timestamp": "Date", "operation": "Opération", "duration_sec": "Durée",
    "emissions_kg": "Émissions CO₂", "energy_kwh": "Énergie",
    "model_name": "Modèle", "n_samples": "Échantillons"
}
df_display.rename(columns={k: v for k, v in rename.items() if k in df_display.columns}, inplace=True)
st.dataframe(df_display, use_container_width=True, hide_index=True)

# Export
csv = df.to_csv(index=False)
st.download_button("⬇️ Exporter le rapport énergétique", data=csv,
                   file_name="thumalien_energy_report.csv", mime="text/csv")
