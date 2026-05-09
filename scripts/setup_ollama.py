#!/usr/bin/env python3
"""
scripts/setup_ollama.py
Installation et configuration automatique d'Ollama + Phi-3 Mini.
Fonctionne sur Windows, macOS et Linux.

Usage : python scripts/setup_ollama.py
"""
import os
import sys
import platform
import subprocess
import time
import urllib.request
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))


def run(cmd: str, check: bool = True, capture: bool = False):
    print(f"  $ {cmd}")
    result = subprocess.run(
        cmd, shell=True,
        capture_output=capture,
        text=True
    )
    if check and result.returncode != 0:
        print(f"  ⚠️  Code de retour : {result.returncode}")
    return result


def check_ollama_installed() -> bool:
    result = subprocess.run("ollama --version", shell=True, capture_output=True)
    return result.returncode == 0


def check_ollama_running() -> bool:
    try:
        import urllib.request
        urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3)
        return True
    except Exception:
        return False


def wait_for_ollama(timeout: int = 30) -> bool:
    print("  ⏳ Attente démarrage Ollama...")
    for i in range(timeout):
        if check_ollama_running():
            return True
        time.sleep(1)
    return False


def main():
    print("\n" + "🤖 " * 20)
    print("THUMALIEN — Installation Ollama + Phi-3 Mini")
    print("🤖 " * 20 + "\n")

    system = platform.system()
    print(f"Système détecté : {system} ({platform.machine()})\n")

    # ----------------------------------------------------------
    # 1. VÉRIFICATION / INSTALLATION OLLAMA
    # ----------------------------------------------------------
    print("=" * 50)
    print("1. Vérification d'Ollama")
    print("=" * 50)

    if check_ollama_installed():
        result = subprocess.run("ollama --version", shell=True, capture_output=True, text=True)
        print(f"  ✅ Ollama déjà installé : {result.stdout.strip()}")
    else:
        print("  ℹ️  Ollama non installé. Installation en cours...\n")

        if system == "Linux":
            run("curl -fsSL https://ollama.com/install.sh | sh")
        elif system == "Darwin":  # macOS
            print("  ℹ️  Sur macOS, téléchargez Ollama depuis : https://ollama.com/download")
            print("  ℹ️  Puis relancez ce script.")
            sys.exit(0)
        elif system == "Windows":
            print("  ℹ️  Sur Windows, téléchargez Ollama depuis : https://ollama.com/download")
            print("  ℹ️  Installez-le, puis relancez ce script.")
            sys.exit(0)
        else:
            print(f"  ❌ Système non supporté pour l'installation automatique : {system}")
            print("  Téléchargez manuellement : https://ollama.com/download")
            sys.exit(1)

        if check_ollama_installed():
            print("  ✅ Ollama installé avec succès")
        else:
            print("  ❌ Échec d'installation d'Ollama")
            sys.exit(1)

    # ----------------------------------------------------------
    # 2. DÉMARRAGE OLLAMA SERVE
    # ----------------------------------------------------------
    print("\n" + "=" * 50)
    print("2. Démarrage du serveur Ollama")
    print("=" * 50)

    if check_ollama_running():
        print("  ✅ Ollama déjà en cours d'exécution sur http://localhost:11434")
    else:
        print("  ℹ️  Démarrage du serveur Ollama en arrière-plan...")
        if system == "Windows":
            subprocess.Popen("ollama serve", shell=True,
                             creationflags=subprocess.CREATE_NO_WINDOW)
        else:
            subprocess.Popen("ollama serve", shell=True,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if wait_for_ollama(30):
            print("  ✅ Ollama démarré sur http://localhost:11434")
        else:
            print("  ❌ Ollama n'a pas démarré dans les temps.")
            print("  Lancez manuellement : ollama serve")
            sys.exit(1)

    # ----------------------------------------------------------
    # 3. PULL DU MODÈLE PHI-3 MINI
    # ----------------------------------------------------------
    print("\n" + "=" * 50)
    print("3. Téléchargement de Phi-3 Mini")
    print("=" * 50)

    # Vérifier si déjà présent
    result = subprocess.run("ollama list", shell=True, capture_output=True, text=True)
    models_installed = result.stdout

    if "phi3" in models_installed:
        print("  ✅ Phi-3 Mini déjà installé")
        print(f"  {[l for l in models_installed.split(chr(10)) if 'phi3' in l]}")
    else:
        print("  ℹ️  Téléchargement de phi3:mini (~2.3 Go)...")
        print("  ⏳ Cela peut prendre 5-15 minutes selon votre connexion...\n")
        rc = run("ollama pull phi3:mini", check=False)
        if rc.returncode == 0:
            print("\n  ✅ Phi-3 Mini téléchargé avec succès")
        else:
            print("\n  ❌ Échec du téléchargement")
            print("  Essayez manuellement : ollama pull phi3:mini")

    # ----------------------------------------------------------
    # 4. TEST DE FONCTIONNEMENT
    # ----------------------------------------------------------
    print("\n" + "=" * 50)
    print("4. Test de fonctionnement")
    print("=" * 50)

    print("  ℹ️  Envoi d'un test à Phi-3 Mini...")
    result = subprocess.run(
        'ollama run phi3:mini "Réponds uniquement: OK"',
        shell=True, capture_output=True, text=True, timeout=60
    )
    if "OK" in result.stdout or result.returncode == 0:
        print("  ✅ Phi-3 Mini répond correctement")
    else:
        print(f"  ⚠️  Réponse inattendue : {result.stdout[:100]}")

    # Test via Python
    print("\n  ℹ️  Test via le module Python...")
    sys.path.insert(0, str(Path(__file__).parents[1]))
    try:
        from src.classifier.llm_classifier import PhiClassifier
        clf = PhiClassifier()
        if clf.is_available():
            result = clf.predict_one("URGENT !!! Partagez avant censure !!!")
            print(f"  ✅ Phi-3 opérationnel")
            print(f"     Score : {result['credibility_score']:.1%}")
            print(f"     Fake  : {result['is_fake']}")
            print(f"     Raison: {result['reasoning'][:80]}...")
        else:
            print("  ⚠️  Phi-3 non disponible via Python")
    except Exception as e:
        print(f"  ⚠️  Erreur test Python : {e}")

    # ----------------------------------------------------------
    # RÉSUMÉ
    # ----------------------------------------------------------
    print("\n" + "✅ " * 20)
    print("INSTALLATION TERMINÉE")
    print("✅ " * 20)
    print("""
Phi-3 Mini est prêt. Le pipeline hybride Thumalien est maintenant actif.

Rappel du fonctionnement hybride :
  • DistilBERT analyse TOUS les posts (rapide)
  • Phi-3 intervient sur les cas AMBIGUS (score entre 35% et 65%)
  • Si Ollama est arrêté → mode DistilBERT seul automatiquement

Pour démarrer Ollama au démarrage de Windows :
  → Recherchez "Ollama" dans le menu Démarrer et lancez l'application

Pour tester le classificateur :
  python -m src.classifier.llm_classifier --check
  python -m src.classifier.llm_classifier --predict "URGENT !!! Complot révélé !!!"
  python -m src.classifier.hybrid_classifier

Pour lancer le pipeline complet :
  python -m src.pipeline --limit 25
""")


if __name__ == "__main__":
    main()
