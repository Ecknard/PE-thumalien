# ============================================================
# Thumalien — Makefile
# Commandes simplifiées pour le développement
# ============================================================

.PHONY: setup install db train collect pipeline dashboard test docker-up docker-down clean

# Installation complète
setup:
	python scripts/setup.py

install:
	pip install -r requirements.txt
	python -m spacy download fr_core_news_sm
	python -m spacy download en_core_web_sm
	python -c "import nltk; [nltk.download(r, quiet=True) for r in ['stopwords', 'punkt']]"

# Base de données
db:
	python scripts/init_db.py

# Entraînement
train-baseline:
	python scripts/train_model.py --model baseline

train-bert:
	python scripts/train_model.py --model bert

train: train-baseline

# Collecte et pipeline
collect:
	python -m src.collector.bluesky_collector --mode once --limit 25 --save-db

pipeline:
	python -m src.pipeline --limit 25

# Dashboard
dashboard:
	streamlit run dashboard/Home.py

# Tests
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=src --cov-report=html

# Docker
docker-up:
	docker-compose up -d

docker-down:
	docker-compose down

docker-build:
	docker-compose build

docker-logs:
	docker-compose logs -f

# Nettoyage
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true

clean-data:
	rm -f data/raw/*.json data/processed/*.json logs/*.log

# Aide
help:
	@echo ""
	@echo "Thumalien — Commandes disponibles :"
	@echo "  make setup          Installation complète automatique"
	@echo "  make install        Installer les dépendances Python"
	@echo "  make db             Initialiser la base de données"
	@echo "  make train          Entraîner le modèle baseline"
	@echo "  make train-bert     Entraîner DistilBERT (CPU, ~20min)"
	@echo "  make collect        Collecter des posts Bluesky"
	@echo "  make pipeline       Lancer le pipeline complet"
	@echo "  make dashboard      Démarrer le dashboard Streamlit"
	@echo "  make test           Lancer les tests unitaires"
	@echo "  make docker-up      Démarrer via Docker Compose"
	@echo "  make docker-down    Arrêter les conteneurs Docker"
	@echo ""

# Ollama / LLM
setup-ollama:
	python scripts/setup_ollama.py

check-phi3:
	python -m src.classifier.llm_classifier --check

test-phi3:
	python -m src.classifier.llm_classifier --benchmark

test-hybrid:
	python -m src.classifier.hybrid_classifier
