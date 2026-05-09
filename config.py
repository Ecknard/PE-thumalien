"""
config.py — Configuration centralisée Thumalien
Toutes les constantes, chemins et paramètres sont ici.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CHEMINS
# ============================================================
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LABELED_DIR = DATA_DIR / "labeled"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Créer les dossiers s'ils n'existent pas
for d in [RAW_DIR, PROCESSED_DIR, LABELED_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ============================================================
# BASE DE DONNÉES
# ============================================================
DB_CONFIG = {
    "dbname": os.getenv("DB_NAME", "thumalien_db"),
    "user": os.getenv("DB_USER", "thumalien_user"),
    "password": os.getenv("DB_PASSWORD", ""),
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 5432)),
}
DB_URL = (
    f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
)

# ============================================================
# BLUESKY API
# ============================================================
BLUESKY_HANDLE = os.getenv("BLUESKY_HANDLE", "")
BLUESKY_APP_PASSWORD = os.getenv("BLUESKY_APP_PASSWORD", "")

# Mots-clés de collecte (FR + EN)
COLLECTION_KEYWORDS = [
    # Français
    "fake news", "désinformation", "infox", "rumeur", "complot",
    "vérification", "fact-check", "hoax", "intox", "manipulation",
    # Anglais
    "disinformation", "misinformation", "fact check", "hoax",
    "breaking news", "rumor", "conspiracy",
]

COLLECTION_LIMIT_PER_KEYWORD = 25
COLLECTION_INTERVAL_SECONDS = 1800  # 30 min

# ============================================================
# PRÉTRAITEMENT NLP
# ============================================================
SUPPORTED_LANGUAGES = ["fr", "en"]
MIN_TOKEN_COUNT = 3

# ============================================================
# MODÈLE DE CLASSIFICATION
# ============================================================
# Modèle léger multilingue — fonctionne sans GPU
CLASSIFIER_MODEL_NAME = "distilbert-base-multilingual-cased"
CLASSIFIER_MAX_LENGTH = 128
CLASSIFIER_BATCH_SIZE = 16
CLASSIFIER_EPOCHS = 4
CLASSIFIER_LEARNING_RATE = 3e-5
CLASSIFIER_TEST_SIZE = 0.2
CLASSIFIER_OUTPUT_DIR = str(MODELS_DIR / "fake_news_classifier")

# Seuil de décision fake news (ajustable)
CREDIBILITY_THRESHOLD = 0.5

# ============================================================
# ANALYSE ÉMOTIONNELLE
# ============================================================
EMOTION_LABELS = {
    "joy": "😊 Joie",
    "anger": "😡 Colère",
    "fear": "😨 Peur",
    "sadness": "😢 Tristesse",
    "surprise": "😲 Surprise",
    "disgust": "🤢 Dégoût",
    "neutral": "😐 Neutre",
}

# ============================================================
# MONITORING ÉNERGÉTIQUE
# ============================================================
CODECARBON_PROJECT = "Thumalien-FakeNews-Detection"
CODECARBON_OUTPUT_DIR = str(LOGS_DIR)
CODECARBON_COUNTRY = "FRA"

# ============================================================
# LOGGING
# ============================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = str(LOGS_DIR / "thumalien.log")

# ============================================================
# MODÈLE LLM (Phi-3 Mini via Ollama) — sans GPU
# ============================================================
# Ollama doit être installé : https://ollama.com
# Puis : ollama pull phi3:mini  (ou python scripts/setup_ollama.py)
LLM_MODEL_NAME = os.getenv("LLM_MODEL_NAME", "phi3:mini")
LLM_OLLAMA_URL = os.getenv("LLM_OLLAMA_URL", "http://localhost:11434")
LLM_TIMEOUT_SECONDS = 60          # Timeout par requête (CPU peut être lent)
LLM_TEMPERATURE = 0.0              # Déterministe et reproductible
LLM_MAX_TOKENS = 256               # Juste assez pour JSON + raisonnement court

# Architecture hybride : DistilBERT filtre en premier,
# Phi-3 intervient uniquement sur la zone d'incertitude
HYBRID_UNCERTAINTY_LOW = 0.35     # Score < 0.35 → Phi-3 requis
HYBRID_UNCERTAINTY_HIGH = 0.65    # Score > 0.65 → Phi-3 requis
# Zone [0.35 - 0.65] → cas ambigu, Phi-3 tranche
