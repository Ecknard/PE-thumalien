#!/usr/bin/env python3
"""
scripts/setup.py
Script d'installation et de configuration automatique du projet Thumalien.
Exécutez : python scripts/setup.py
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.append(str(ROOT))


def run(cmd: str, check: bool = True) -> int:
    """Exécute une commande shell et retourne le code de retour."""
    print(f"  $ {cmd}")
    result = subprocess.run(cmd, shell=True, cwd=str(ROOT))
    if check and result.returncode != 0:
        print(f"  ❌ Erreur (code {result.returncode})")
    return result.returncode


def step(title: str):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print('='*50)


def main():
    print("\n" + "🚀 " * 20)
    print("THUMALIEN — Installation automatique")
    print("🚀 " * 20)

    # ----------------------------------------------------------
    # 1. VÉRIFICATION PYTHON
    # ----------------------------------------------------------
    step("1. Vérification Python")
    version = sys.version_info
    if version < (3, 10):
        print(f"  ❌ Python {version.major}.{version.minor} détecté. Python 3.10+ requis.")
        sys.exit(1)
    print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")

    # ----------------------------------------------------------
    # 2. FICHIER .env
    # ----------------------------------------------------------
    step("2. Configuration .env")
    env_file = ROOT / ".env"
    env_example = ROOT / ".env.example"

    if not env_file.exists():
        if env_example.exists():
            shutil.copy(env_example, env_file)
            print(f"  ✅ .env créé depuis .env.example")
            print(f"  ⚠️  IMPORTANT : Éditez le fichier .env avec vos vrais credentials !")
            print(f"     → {env_file}")
        else:
            print("  ⚠️  .env.example introuvable. Créez manuellement le fichier .env")
    else:
        print("  ✅ .env déjà présent")

    # ----------------------------------------------------------
    # 3. INSTALLATION DES DÉPENDANCES
    # ----------------------------------------------------------
    step("3. Installation des dépendances Python")
    rc = run("pip install -r requirements.txt --quiet")
    if rc == 0:
        print("  ✅ Dépendances installées")
    else:
        print("  ⚠️  Certaines dépendances ont échoué. Relancez manuellement.")

    # ----------------------------------------------------------
    # 4. MODÈLES SPACY
    # ----------------------------------------------------------
    step("4. Téléchargement des modèles spaCy")
    for model in ["fr_core_news_sm", "en_core_web_sm"]:
        rc = run(f"python -m spacy download {model} --quiet", check=False)
        status = "✅" if rc == 0 else "⚠️ "
        print(f"  {status} {model}")

    # ----------------------------------------------------------
    # 5. RESSOURCES NLTK
    # ----------------------------------------------------------
    step("5. Téléchargement ressources NLTK")
    run('python -c "import nltk; [nltk.download(r, quiet=True) for r in [\'stopwords\', \'punkt\', \'punkt_tab\']]"')
    print("  ✅ NLTK prêt")

    # ----------------------------------------------------------
    # 6. STRUCTURE DES DOSSIERS
    # ----------------------------------------------------------
    step("6. Création de la structure des dossiers")
    dirs = [
        ROOT / "data" / "raw",
        ROOT / "data" / "processed",
        ROOT / "data" / "labeled",
        ROOT / "models",
        ROOT / "logs",
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        print(f"  ✅ {d.relative_to(ROOT)}")

    # Créer les __init__.py
    for pkg in ["src", "src/collector", "src/preprocessing", "src/classifier",
                "src/emotion", "src/monitoring", "src/database", "src/explainability"]:
        init = ROOT / pkg / "__init__.py"
        init.touch()

    # ----------------------------------------------------------
    # 7. ENTRAÎNEMENT MODÈLE BASELINE
    # ----------------------------------------------------------
    step("7. Entraînement du modèle baseline (données d'exemple)")
    rc = run(
        "python -m src.classifier.fake_news_classifier --train --model baseline --sample",
        check=False
    )
    if rc == 0:
        print("  ✅ Modèle baseline entraîné et sauvegardé dans models/")
    else:
        print("  ⚠️  Entraînement échoué. Relancez : python -m src.classifier.fake_news_classifier --train --sample")

    # ----------------------------------------------------------
    # 8. TEST DE CONNEXION DB (optionnel)
    # ----------------------------------------------------------
    step("8. Test connexion PostgreSQL (optionnel)")
    rc = run("python -m src.database.db_connector", check=False)
    if rc != 0:
        print("  ⚠️  PostgreSQL non disponible. Configurez .env et relancez.")
        print("  ℹ️  Pour démarrer PostgreSQL via Docker : docker-compose up postgres -d")

    # ----------------------------------------------------------
    # 9. RÉSUMÉ
    # ----------------------------------------------------------
    print("\n" + "✅ " * 20)
    print("INSTALLATION TERMINÉE")
    print("✅ " * 20)
    print("""
Prochaines étapes :

1. Configurez votre fichier .env avec vos credentials Bluesky et PostgreSQL
   → cp .env.example .env  (puis éditez le fichier)

2. Démarrez PostgreSQL :
   → docker-compose up postgres -d
   → python -m src.database.db_connector   (initialise le schéma)

3. Collectez des données :
   → python -m src.collector.bluesky_collector --mode once --limit 25 --save-db

4. Lancez le pipeline complet :
   → python -m src.pipeline --limit 25

5. Démarrez le dashboard :
   → streamlit run dashboard/Home.py

6. Ou tout en Docker :
   → docker-compose up

7. Lancez les tests :
   → pytest
""")


if __name__ == "__main__":
    main()
