# 🛡️ Thumalien — Détection de Fake News sur Bluesky

> **Projet d'études — Mastère Big Data & IA — SUP DE VINCI 2025/2026**  
> Pipeline NLP & IA multilingue (FR/EN) pour détecter la désinformation sur le réseau social Bluesky.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red.svg)](https://streamlit.io)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue.svg)](https://postgresql.org)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Table des matières

- [Contexte](#-contexte)
- [Architecture](#-architecture)
- [Fonctionnalités](#-fonctionnalités)
- [Installation rapide](#-installation-rapide)
- [Utilisation](#-utilisation)
- [Structure du projet](#-structure-du-projet)
- [Pipeline détaillé](#-pipeline-détaillé)
- [Dashboard](#-dashboard)
- [Green IT](#-green-it)
- [KPIs & Métriques](#-kpis--métriques)
- [Équipe](#-équipe)

---

## 🎯 Contexte

Le client fictif **Thumalien** souhaite une solution automatisée pour :

- **Identifier** rapidement les contenus douteux ou trompeurs sur Bluesky
- **Évaluer** leur impact émotionnel (colère, peur, humour…)
- **Expliquer** pourquoi un contenu est jugé suspect (transparence IA)
- **Mesurer** la consommation énergétique du pipeline (Green IT)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE THUMALIEN                    │
├──────────┬──────────────┬───────────────┬───────────────┤
│ COLLECTE │ PRÉTRAITEMENT│ CLASSIFICATION│   ÉMOTION     │
│ Bluesky  │    NLP       │  DistilBERT   │    VADER +    │
│  ATProto │ spaCy + NLTK │  (CPU ready)  │  Heuristiques │
│  FR + EN │  Tokenize    │  Score 0→1    │  7 émotions   │
│          │  Lemmatize   │  Crédibilité  │               │
└──────────┴──────────────┴───────────────┴───────────────┘
              ↓                    ↓               ↓
        ┌─────────────────────────────────────────────┐
        │           PostgreSQL (persistance)           │
        │  posts | predictions | emotions | energy     │
        └─────────────────────────────────────────────┘
                              ↓
        ┌─────────────────────────────────────────────┐
        │         Dashboard Streamlit (4 pages)        │
        │  Accueil | Analyser | Explorer | Énergie     │
        └─────────────────────────────────────────────┘
```

---

## ✨ Fonctionnalités

| Module | Description |
|--------|-------------|
| 📡 **Collecte** | API Bluesky (ATProto), mots-clés, timeline, multi-utilisateurs |
| 🧹 **Prétraitement** | URLs, mentions, emojis, stopwords, lemmatisation (FR/EN) |
| 🤖 **Classification** | DistilBERT multilingue fine-tuné + baseline TF-IDF/LogReg |
| 📊 **Score crédibilité** | Score 0→1, seuil configurable |
| 😊 **Émotions** | VADER + lexiques : joie, colère, peur, tristesse, surprise, dégoût, neutre |
| 🔍 **Explicabilité** | Signaux linguistiques, raisons de classification, résumé textuel |
| ⚡ **Green IT** | CodeCarbon : mesure CO₂, énergie, durée par opération |
| 🗄️ **Persistance** | PostgreSQL avec schéma optimisé, index, upsert |
| 📱 **Dashboard** | Streamlit 4 pages : KPIs, analyse temps réel, exploration, énergie |
| 🐳 **Docker** | Déploiement complet en une commande |
| 🧪 **Tests** | Pytest, 30+ tests unitaires, coverage |

---

## 🚀 Installation rapide

### Option 1 : Setup automatique (recommandé)

```bash
git clone https://github.com/votre-org/thumalien.git
cd thumalien

# Copier et configurer les credentials
cp .env.example .env
# Éditer .env avec vos credentials Bluesky et PostgreSQL

# Installation complète (dépendances + modèles + baseline)
python scripts/setup.py
```

### Option 2 : Docker (le plus simple)

```bash
cp .env.example .env
# Éditer .env

docker-compose up
# Dashboard disponible sur http://localhost:8501
```

### Option 3 : Installation manuelle

```bash
pip install -r requirements.txt
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm

# Configurer .env
cp .env.example .env

# Initialiser la DB
python scripts/init_db.py

# Entraîner le modèle baseline
python scripts/train_model.py --model baseline

# Lancer le dashboard
streamlit run dashboard/Home.py
```

---

## 💻 Utilisation

### Collecte de posts Bluesky

```bash
# Collecte unique (25 posts par mot-clé, sauvegarde en DB)
python -m src.collector.bluesky_collector --mode once --limit 25 --save-db

# Collecte continue (daemon, toutes les 30 min)
python -m src.collector.bluesky_collector --mode daemon
```

### Pipeline complet

```bash
# Depuis l'API Bluesky
python -m src.pipeline --limit 25

# Depuis un fichier JSON existant (skip collecte)
python -m src.pipeline --from-file data/raw/bluesky_posts_XXXXXXXX.json
```

### Entraînement du modèle

```bash
# Baseline rapide (TF-IDF + LogisticRegression) — quelques secondes
python scripts/train_model.py --model baseline

# DistilBERT multilingue — 10-30 min sur CPU
python scripts/train_model.py --model bert

# Les deux
python scripts/train_model.py --model both
```

### Analyse d'un texte en ligne de commande

```bash
python -m src.classifier.fake_news_classifier \
  --predict "URGENT !!! Le gouvernement cache la vérité !!!"
```

### Lancer les tests

```bash
pytest                    # tous les tests
pytest -v                 # verbose
pytest --cov=src tests/   # avec coverage
```

---

## 📁 Structure du projet

```
thumalien/
├── config.py                     # Configuration centralisée
├── requirements.txt              # Dépendances Python
├── docker-compose.yml            # Orchestration Docker
├── .env.example                  # Template credentials (ne pas committer .env)
│
├── src/
│   ├── collector/
│   │   └── bluesky_collector.py  # Collecte API Bluesky (ATProto)
│   ├── preprocessing/
│   │   └── text_preprocessor.py  # Pipeline NLP (nettoyage, tokenisation, lemmatisation)
│   ├── classifier/
│   │   └── fake_news_classifier.py # DistilBERT + baseline TF-IDF
│   ├── emotion/
│   │   └── emotion_analyzer.py   # VADER + heuristiques (7 émotions)
│   ├── explainability/
│   │   └── explainer.py          # Signaux linguistiques, résumés
│   ├── monitoring/
│   │   └── energy_tracker.py     # CodeCarbon (CO₂, énergie)
│   ├── database/
│   │   └── db_connector.py       # PostgreSQL (schéma, CRUD)
│   └── pipeline.py               # Orchestrateur complet
│
├── dashboard/
│   ├── Home.py                   # Page d'accueil (KPIs globaux)
│   └── pages/
│       ├── 1_🔍_Analyser.py      # Analyse temps réel
│       ├── 2_📊_Explorer.py      # Exploration des données
│       ├── 3_📈_Métriques.py     # Performance du modèle
│       └── 4_⚡_Energie.py       # Bilan Green IT
│
├── tests/
│   └── test_pipeline.py          # 30+ tests unitaires
│
├── scripts/
│   ├── setup.py                  # Installation automatique
│   ├── init_db.py                # Initialisation PostgreSQL
│   └── train_model.py            # Entraînement modèle
│
├── docker/
│   ├── Dockerfile
│   └── init.sql
│
├── data/
│   ├── raw/                      # Posts bruts collectés (gitignored)
│   ├── processed/                # Posts prétraités (gitignored)
│   └── labeled/                  # Données labellisées (versionné)
│
├── models/                       # Modèles entraînés (gitignored sauf .json)
└── logs/                         # Logs applicatifs (gitignored)
```

---

## 🔬 Pipeline détaillé

### 1. Collecte (`src/collector/bluesky_collector.py`)
- Authentification via ATProto (App Password Bluesky)
- Recherche par mots-clés : `fake news`, `désinformation`, `complot`, `hoax`…
- Filtrage automatique FR/EN via `langdetect`
- Déduplication par URI de post
- Sauvegarde JSON + PostgreSQL

### 2. Prétraitement NLP (`src/preprocessing/text_preprocessor.py`)
- Suppression URLs, mentions, emojis
- Normalisation ponctuation et espaces
- Tokenisation et lemmatisation (spaCy `fr_core_news_sm` / `en_core_web_sm`)
- Suppression stopwords (NLTK)

### 3. Classification (`src/classifier/fake_news_classifier.py`)
- **Modèle principal** : `distilbert-base-multilingual-cased` fine-tuné
  - Léger (66M paramètres), 2x plus rapide que BERT
  - Fonctionne sur CPU sans GPU
  - Score de crédibilité entre 0 (fake) et 1 (réel)
- **Baseline** : TF-IDF (bigrams, 10k features) + LogisticRegression
  - Fallback si BERT non disponible
  - Entraînement en quelques secondes

### 4. Analyse émotionnelle (`src/emotion/emotion_analyzer.py`)
- VADER Sentiment Analysis (composante `compound`)
- Lexiques émotionnels FR/EN (joie, colère, peur, tristesse, surprise, dégoût)
- Détection de patterns fake news (urgence, censure, complot)
- 7 émotions : joy, anger, fear, sadness, surprise, disgust, neutral

### 5. Explicabilité (`src/explainability/explainer.py`)
- Détection de signaux suspects : urgence, ponctuation excessive, majuscules, complotisme, appel à partager avant censure
- Signaux de crédibilité : sources citées, langage nuancé
- Score de risque combiné (modèle + signaux)
- Résumé textuel lisible par un non-expert

---

## 📱 Dashboard

Le dashboard Streamlit comprend **4 pages** :

| Page | Description |
|------|-------------|
| 🏠 **Accueil** | KPIs globaux, graphiques répartition fake/réel, émotions, derniers posts |
| 🔍 **Analyser** | Analyse d'un texte en temps réel : jauge crédibilité, radar émotionnel, signaux |
| 📊 **Explorer** | Tableau filtrable, graphiques, export CSV |
| 📈 **Métriques** | F1, Accuracy, matrice de confusion, KPIs métier |
| ⚡ **Énergie** | Bilan CO₂, énergie par opération, historique Green IT |

```bash
streamlit run dashboard/Home.py
# Accessible sur : http://localhost:8501
```

---

## ⚡ Green IT

Le projet suit l'empreinte énergétique de chaque opération via **CodeCarbon** :

| Opération | Durée estimée | CO₂ estimé |
|-----------|---------------|-----------|
| Collecte Bluesky | ~30s | < 0.1g |
| Prétraitement NLP | ~10s | < 0.05g |
| Classification baseline | ~5s | < 0.02g |
| Fine-tuning DistilBERT | ~20 min | ~2-5g |
| Analyse émotionnelle | ~5s | < 0.02g |

**Choix éco-responsables :**
- DistilBERT (2x moins énergivore que BERT)
- Mise en cache des prédictions PostgreSQL
- Modèle CPU-first (pas de GPU cloud)

---

## 📊 KPIs & Métriques

| KPI | Objectif | Description |
|-----|----------|-------------|
| F1-Score | > 70% | Équilibre précision/rappel |
| Taux faux positifs | < 20% | Posts réels classés fake |
| Taux faux négatifs | < 15% | Fake news non détectées |
| Temps d'analyse | < 2s/post | Latence du pipeline |
| Couverture linguistique | FR + EN | Langues supportées |
| Score crédibilité | 0→1 continu | Explicable par seuillage |

---

## 👥 Équipe

| Rôle | Responsabilités |
|------|-----------------|
| **Data Engineer** | API Bluesky, base PostgreSQL, pipeline ETL |
| **Data Scientist** | Modèle NLP, classification, métriques |
| **Analyste IA** | Émotions, explicabilité, dashboard |
| **Green IT** | Monitoring énergétique, qualité code, tests |

---

## 🔧 Variables d'environnement

| Variable | Description | Exemple |
|----------|-------------|---------|
| `BLUESKY_HANDLE` | Handle Bluesky | `user.bsky.social` |
| `BLUESKY_APP_PASSWORD` | App Password Bluesky | `xxxx-xxxx-xxxx-xxxx` |
| `DB_NAME` | Nom de la base | `thumalien_db` |
| `DB_USER` | Utilisateur PostgreSQL | `thumalien_user` |
| `DB_PASSWORD` | Mot de passe PostgreSQL | — |
| `DB_HOST` | Hôte PostgreSQL | `localhost` |
| `DB_PORT` | Port PostgreSQL | `5432` |

---

## 📄 Licence

MIT License — SUP DE VINCI 2025/2026

---

*Thumalien — Mastère Big Data & IA — SUP DE VINCI*
