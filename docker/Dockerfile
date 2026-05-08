# ============================================================
# Thumalien — Dockerfile
# Python 3.11 slim, optimisé pour CPU
# ============================================================
FROM python:3.11-slim

LABEL maintainer="Thumalien Team — SUP DE VINCI 2025/2026"
LABEL description="Fake News Detection Pipeline for Bluesky"

# Dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dépendances Python (layer cacheable)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Téléchargement des modèles spaCy
RUN python -m spacy download fr_core_news_sm && \
    python -m spacy download en_core_web_sm

# Téléchargement ressources NLTK
RUN python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt')"

# Copie du code source
COPY . .

# Dossiers de données
RUN mkdir -p data/raw data/processed data/labeled models logs

# Port Streamlit
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Commande par défaut : dashboard
CMD ["streamlit", "run", "dashboard/Home.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]
