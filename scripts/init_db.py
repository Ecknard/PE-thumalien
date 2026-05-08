#!/usr/bin/env python3
"""
scripts/init_db.py
Initialise le schéma PostgreSQL et insère des données d'exemple.
"""
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from src.database.db_connector import DatabaseConnector
from src.classifier.fake_news_classifier import create_sample_dataset
from config import LABELED_DIR


def main():
    print("🗄️  Initialisation de la base de données PostgreSQL\n")

    db = DatabaseConnector()

    # Test connexion
    if not db.test_connection():
        print("\n❌ Impossible de se connecter à PostgreSQL.")
        print("   Vérifiez votre .env et que PostgreSQL est démarré.")
        print("   Docker : docker-compose up postgres -d")
        sys.exit(1)

    # Initialiser le schéma
    if db.initialize_schema():
        print("✅ Schéma créé (tables posts, predictions, emotion_analyses, energy_tracking)")
    else:
        print("❌ Erreur lors de la création du schéma")
        sys.exit(1)

    # Stats initiales
    stats = db.count_posts()
    print(f"\n📊 État de la base :")
    print(f"   Posts total : {stats['total']}")
    print(f"   Posts FR    : {stats['fr']}")
    print(f"   Posts EN    : {stats['en']}")

    # Générer dataset d'exemple si absent
    sample_file = LABELED_DIR / "labeled_posts_sample.json"
    if not sample_file.exists():
        print("\n📝 Génération du dataset d'exemple...")
        create_sample_dataset()

    print("\n✅ Base de données prête !")
    print("   Prochaine étape : python -m src.collector.bluesky_collector --save-db")


if __name__ == "__main__":
    main()
