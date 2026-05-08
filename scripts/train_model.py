#!/usr/bin/env python3
"""
scripts/train_model.py
Entraîne le modèle de classification fake news.
Supporte baseline (rapide) et BERT (précis, CPU-friendly).
"""
import sys
import json
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))

from src.classifier.fake_news_classifier import (
    BaselineClassifier, BERTClassifier,
    load_labeled_data, create_sample_dataset
)
from src.monitoring.energy_tracker import EnergyTracker
from config import LABELED_DIR, MODELS_DIR


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Entraînement modèle Thumalien")
    parser.add_argument("--model", choices=["baseline", "bert", "both"], default="baseline",
                        help="Modèle à entraîner (baseline=rapide, bert=précis)")
    parser.add_argument("--data", type=str, default=None,
                        help="Chemin vers le fichier JSON labellisé (sinon dataset d'exemple)")
    args = parser.parse_args()

    tracker = EnergyTracker()

    print("📂 Chargement des données...")
    if args.data:
        texts, labels = load_labeled_data(Path(args.data))
    else:
        texts, labels = load_labeled_data()

    if not texts:
        print("⚠️  Pas de données labellisées, utilisation du dataset d'exemple.")
        print("   Pour de meilleures performances, créez data/labeled/labeled_posts.json")
        texts, labels = create_sample_dataset()

    print(f"   {len(texts)} exemples : {labels.count(0)} fake / {labels.count(1)} réels\n")

    results = {}

    # Baseline
    if args.model in ("baseline", "both"):
        print("🏃 Entraînement Baseline (TF-IDF + LogisticRegression)...")
        with tracker.track("train_baseline", n_samples=len(texts), model_name="tfidf-logreg"):
            clf = BaselineClassifier()
            metrics = clf.train(texts, labels)
        results["baseline"] = metrics
        print(f"\n✅ Baseline sauvegardé : {BaselineClassifier.MODEL_PATH}")

    # BERT
    if args.model in ("bert", "both"):
        print("\n🤖 Entraînement DistilBERT multilingue (CPU, ~10-30 min)...")
        with tracker.track("train_bert", n_samples=len(texts), model_name="distilbert-multilingual"):
            clf_bert = BERTClassifier()
            metrics_bert = clf_bert.train(texts, labels)
        results["bert"] = metrics_bert

    # Rapport
    print("\n" + "="*50)
    print("📊 RÉSULTATS D'ENTRAÎNEMENT")
    print("="*50)
    for name, m in results.items():
        print(f"\n  [{name.upper()}]")
        print(f"   F1-Score : {m.get('f1', 0):.1%}")
        print(f"   Accuracy : {m.get('accuracy', 0):.1%}")
        if m.get("roc_auc"):
            print(f"   ROC-AUC  : {m['roc_auc']:.3f}")

    tracker.print_report()

    # Sauvegarder résumé
    summary_path = MODELS_DIR / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✅ Résumé sauvegardé : {summary_path}")


if __name__ == "__main__":
    main()
