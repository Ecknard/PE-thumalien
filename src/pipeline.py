"""
src/pipeline.py
Orchestrateur du pipeline complet Thumalien.
Exécute : Collecte → Prétraitement → Classification → Émotion → Sauvegarde DB
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from loguru import logger

sys.path.append(str(Path(__file__).parent))
from config import RAW_DIR, PROCESSED_DIR, LOG_FILE
from src.collector.bluesky_collector import BlueskyCollector
from src.preprocessing.text_preprocessor import preprocess_posts
from src.classifier.fake_news_classifier import FakeNewsClassifier
from src.emotion.emotion_analyzer import EmotionAnalyzer
from src.explainability.explainer import FakeNewsExplainer
from src.monitoring.energy_tracker import EnergyTracker
from src.database.db_connector import DatabaseConnector

logger.add(LOG_FILE, rotation="10 MB", level="INFO")


class ThumalienPipeline:
    """
    Pipeline complet d'analyse de fake news sur Bluesky.

    Étapes :
      1. Collecte API Bluesky
      2. Prétraitement NLP
      3. Classification fake news (DistilBERT ou baseline)
      4. Analyse émotionnelle
      5. Explicabilité
      6. Sauvegarde PostgreSQL
    """

    def __init__(self):
        self.collector = BlueskyCollector()
        self.classifier = FakeNewsClassifier(prefer_bert=True)
        self.emotion_analyzer = EmotionAnalyzer()
        self.explainer = FakeNewsExplainer()
        self.db = DatabaseConnector()
        self.energy_tracker = EnergyTracker()

        self._models_loaded = False

    def _ensure_models(self):
        """Charge les modèles une seule fois."""
        if not self._models_loaded:
            self.classifier.load()
            self._models_loaded = True

    # ----------------------------------------------------------
    # ÉTAPES INDIVIDUELLES
    # ----------------------------------------------------------

    def step_collect(self, limit_per_keyword: int = 25) -> List[Dict]:
        """Étape 1 : Collecte des posts Bluesky."""
        print("\n📡 ÉTAPE 1 — Collecte Bluesky")
        print("-" * 40)

        with self.energy_tracker.track("collect_bluesky", model_name="atproto"):
            posts = self.collector.collect_all(limit_per_keyword=limit_per_keyword)

        if posts:
            filepath = self.collector.save_to_json(posts)
            n_inserted = self.db.insert_posts(posts)
            print(f"✅ {len(posts)} posts collectés, {n_inserted} nouveaux en DB")
            print(f"   Sauvegardé : {filepath}")
        else:
            print("⚠️  Aucun post collecté")

        return posts

    def step_preprocess(self, posts: List[Dict]) -> List[Dict]:
        """Étape 2 : Prétraitement NLP."""
        print("\n🧹 ÉTAPE 2 — Prétraitement NLP")
        print("-" * 40)

        if not posts:
            print("⚠️  Aucun post à prétraiter")
            return []

        with self.energy_tracker.track("preprocessing", n_samples=len(posts)):
            processed = preprocess_posts(posts)

        print(f"✅ {len(processed)} posts prétraités")
        avg_tokens = sum(p.get("token_count", 0) for p in processed) / max(len(processed), 1)
        print(f"   Tokens moyens par post : {avg_tokens:.1f}")

        # Sauvegarder
        out_file = PROCESSED_DIR / f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

        return processed

    def step_classify(self, posts: List[Dict]) -> List[Dict]:
        """Étape 3 : Classification fake news."""
        print("\n🤖 ÉTAPE 3 — Classification Fake News")
        print("-" * 40)

        if not posts:
            return []

        self._ensure_models()

        texts = [p.get("text_clean") or p.get("text", "") for p in posts]

        with self.energy_tracker.track(
            "classification", n_samples=len(texts), model_name=self.classifier.active_model
        ):
            predictions = self.classifier.predict(texts)

        # Enrichir les posts et sauvegarder en DB
        enriched = []
        fake_count = 0
        for post, pred in zip(posts, predictions):
            post_enriched = {
                **post,
                "credibility_score": pred["credibility_score"],
                "is_fake": pred["is_fake"],
                "confidence": pred["confidence"],
            }
            enriched.append(post_enriched)

            if pred["is_fake"]:
                fake_count += 1

            # Sauvegarder en DB
            try:
                self.db.upsert_prediction(
                    post_id=post["id"],
                    credibility_score=pred["credibility_score"],
                    is_fake=pred["is_fake"],
                    confidence=pred["confidence"],
                )
            except Exception as e:
                logger.warning(f"Erreur sauvegarde prédiction: {e}")

        real_count = len(enriched) - fake_count
        print(f"✅ {len(enriched)} posts classifiés")
        print(f"   🔴 Fake news détectées : {fake_count} ({fake_count/max(len(enriched),1):.0%})")
        print(f"   🟢 Posts réels         : {real_count} ({real_count/max(len(enriched),1):.0%})")

        return enriched

    def step_analyze_emotions(self, posts: List[Dict]) -> List[Dict]:
        """Étape 4 : Analyse émotionnelle."""
        print("\n😊 ÉTAPE 4 — Analyse Émotionnelle")
        print("-" * 40)

        if not posts:
            return []

        with self.energy_tracker.track("emotion_analysis", n_samples=len(posts)):
            enriched = self.emotion_analyzer.analyze_batch(posts)

        # Sauvegarder en DB
        for post in enriched:
            try:
                self.db.upsert_emotion(
                    post_id=post["id"],
                    emotion_label=post.get("emotion_emotion_label", "neutral"),
                    emotion_score=post.get("emotion_emotion_score", 0.0),
                    vader_compound=post.get("emotion_vader_compound", 0.0),
                )
            except Exception as e:
                logger.warning(f"Erreur sauvegarde émotion: {e}")

        # Distribution
        distribution = self.emotion_analyzer.get_distribution([
            {"emotion_label": p.get("emotion_emotion_label", "neutral")} for p in enriched
        ])
        top_emotion = max(distribution, key=distribution.get)
        print(f"✅ {len(enriched)} posts analysés")
        print(f"   Émotion dominante : {top_emotion} ({distribution[top_emotion]} posts)")

        return enriched

    def step_explain(self, posts: List[Dict]) -> List[Dict]:
        """Étape 5 : Génération des explications."""
        print("\n🔍 ÉTAPE 5 — Explicabilité IA")
        print("-" * 40)

        if not posts:
            return []

        enriched = self.explainer.explain_batch(posts)
        high_risk = sum(1 for p in enriched if p.get("explanation", {}).get("is_high_risk", False))
        print(f"✅ Explications générées pour {len(enriched)} posts")
        print(f"   ⚠️  Posts à haut risque : {high_risk}")

        return enriched

    # ----------------------------------------------------------
    # PIPELINE COMPLET
    # ----------------------------------------------------------

    def run_full_pipeline(
        self,
        limit_per_keyword: int = 25,
        skip_collect: bool = False,
        input_file: Optional[Path] = None,
    ) -> List[Dict]:
        """
        Exécute le pipeline complet.

        Args:
            limit_per_keyword: posts par mot-clé (étape 1)
            skip_collect: si True, charge depuis input_file
            input_file: fichier JSON à utiliser si skip_collect=True
        """
        print("\n" + "=" * 60)
        print("🚀 PIPELINE THUMALIEN — DÉMARRAGE")
        print(f"   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)

        start = datetime.now()

        # Étape 1 : Collecte
        if skip_collect and input_file and Path(input_file).exists():
            print(f"\n📂 Chargement depuis : {input_file}")
            with open(input_file, encoding="utf-8") as f:
                posts = json.load(f)
            print(f"   {len(posts)} posts chargés")
        else:
            posts = self.step_collect(limit_per_keyword=limit_per_keyword)

        if not posts:
            print("\n❌ Aucun post disponible. Pipeline arrêté.")
            return []

        # Étapes suivantes
        posts = self.step_preprocess(posts)
        posts = self.step_classify(posts)
        posts = self.step_analyze_emotions(posts)
        posts = self.step_explain(posts)

        # Rapport final
        duration = (datetime.now() - start).total_seconds()
        print("\n" + "=" * 60)
        print("✅ PIPELINE TERMINÉ")
        print(f"   Durée totale    : {duration:.1f}s")
        print(f"   Posts traités   : {len(posts)}")
        fake = sum(1 for p in posts if p.get("is_fake", False))
        print(f"   Fake news       : {fake}/{len(posts)}")

        self.energy_tracker.print_report()

        return posts


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline Thumalien")
    parser.add_argument("--limit", type=int, default=25, help="Posts par mot-clé")
    parser.add_argument("--from-file", type=str, default=None,
                        help="Utiliser un fichier JSON existant (skip collecte)")
    parser.add_argument("--train", action="store_true",
                        help="Entraîner le modèle avant d'analyser")
    args = parser.parse_args()

    if args.train:
        from src.classifier.fake_news_classifier import FakeNewsClassifier, load_labeled_data, create_sample_dataset, BaselineClassifier
        texts, labels = load_labeled_data()
        if not texts:
            texts, labels = create_sample_dataset()
        clf = BaselineClassifier()
        clf.train(texts, labels)
        print("\n✅ Modèle baseline entraîné.")

    pipeline = ThumalienPipeline()
    pipeline.run_full_pipeline(
        limit_per_keyword=args.limit,
        skip_collect=bool(args.from_file),
        input_file=args.from_file,
    )
