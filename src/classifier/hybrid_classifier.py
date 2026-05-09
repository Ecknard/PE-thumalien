"""
src/classifier/hybrid_classifier.py
Architecture hybride : DistilBERT (rapide) + Phi-3 Mini (précis).

Logique de décision :
  ┌─────────────────────────────────────────────────────────┐
  │                  Post entrant                           │
  │                      ↓                                  │
  │           DistilBERT / Baseline (rapide)                │
  │                      ↓                                  │
  │    Score < LOW ou > HIGH ?  →  Résultat direct         │
  │    (cas clairs : très fake ou très fiable)              │
  │                      ↓ (zone d'incertitude)             │
  │           Phi-3 Mini via Ollama (précis)                │
  │           + raisonnement en langage naturel             │
  │                      ↓                                  │
  │     Score final = combinaison pondérée Bert + Phi-3     │
  └─────────────────────────────────────────────────────────┘

Avantages :
  - 95% des posts résolus par DistilBERT en <1s
  - Phi-3 n'intervient que sur les cas ambigus (les plus importants)
  - Si Ollama est absent → mode DistilBERT seul (dégradé gracieux)
  - Raisonnement Phi-3 disponible pour tous les cas ambigus
"""
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

sys.path.append(str(Path(__file__).parents[2]))
from config import (
    HYBRID_UNCERTAINTY_LOW, HYBRID_UNCERTAINTY_HIGH,
    CREDIBILITY_THRESHOLD, LOG_FILE
)

logger.add(LOG_FILE, rotation="10 MB", level="INFO")


class HybridFakeNewsClassifier:
    """
    Classificateur hybride à deux niveaux.

    Niveau 1 — DistilBERT (ou baseline TF-IDF) :
        Rapide, traite tous les posts en premier.
        Si la prédiction est claire (< LOW ou > HIGH), s'arrête là.

    Niveau 2 — Phi-3 Mini via Ollama :
        Intervient uniquement sur la zone d'incertitude [LOW, HIGH].
        Produit un raisonnement explicite + signaux détectés.
        Résultat final = moyenne pondérée (0.4 DistilBERT + 0.6 Phi-3).
    """

    def __init__(
        self,
        uncertainty_low: float = HYBRID_UNCERTAINTY_LOW,
        uncertainty_high: float = HYBRID_UNCERTAINTY_HIGH,
        phi3_weight: float = 0.6,
    ):
        self.uncertainty_low = uncertainty_low
        self.uncertainty_high = uncertainty_high
        self.phi3_weight = phi3_weight
        self.bert_weight = 1.0 - phi3_weight

        # Chargement lazy
        self._bert_clf = None
        self._phi3_clf = None
        self._phi3_available: Optional[bool] = None

    # ----------------------------------------------------------
    # CHARGEMENT LAZY DES MODÈLES
    # ----------------------------------------------------------

    @property
    def bert_clf(self):
        if self._bert_clf is None:
            from src.classifier.fake_news_classifier import FakeNewsClassifier
            self._bert_clf = FakeNewsClassifier(prefer_bert=True)
            self._bert_clf.load()
            logger.info(f"Modèle niveau 1 chargé : {self._bert_clf.active_model}")
        return self._bert_clf

    @property
    def phi3_clf(self):
        if self._phi3_clf is None:
            from src.classifier.llm_classifier import PhiClassifier
            self._phi3_clf = PhiClassifier()
        return self._phi3_clf

    @property
    def phi3_available(self) -> bool:
        if self._phi3_available is None:
            self._phi3_available = self.phi3_clf.is_available()
            if self._phi3_available:
                logger.info("Phi-3 Mini disponible — mode hybride activé")
            else:
                logger.warning(
                    "Phi-3 non disponible — mode DistilBERT seul. "
                    "Pour activer Phi-3 : ollama pull phi3:mini && ollama serve"
                )
        return self._phi3_available

    # ----------------------------------------------------------
    # LOGIQUE HYBRIDE
    # ----------------------------------------------------------

    def _is_uncertain(self, score: float) -> bool:
        """
        Retourne True si le score DistilBERT est dans la zone d'incertitude.
        Ces cas sont envoyés à Phi-3 pour un second avis.
        """
        return self.uncertainty_low <= score <= self.uncertainty_high

    def _combine_scores(
        self,
        bert_score: float,
        phi3_score: float,
        phi3_confidence: float,
    ) -> float:
        """
        Combinaison pondérée des deux scores.
        Si Phi-3 a une haute confiance, son poids augmente.
        """
        # Ajustement dynamique : si Phi-3 très confiant → poids augmenté
        dynamic_phi3_weight = self.phi3_weight + (phi3_confidence - 0.5) * 0.2
        dynamic_phi3_weight = max(0.4, min(0.8, dynamic_phi3_weight))
        dynamic_bert_weight = 1.0 - dynamic_phi3_weight

        combined = bert_score * dynamic_bert_weight + phi3_score * dynamic_phi3_weight
        return round(combined, 4)

    def predict_one(self, text: str) -> Dict:
        """
        Prédit la crédibilité d'un texte avec l'architecture hybride.
        """
        start = time.time()

        # --- Niveau 1 : DistilBERT ---
        bert_result = self.bert_clf.predict_one(text)
        bert_score = bert_result["credibility_score"]

        result = {
            **bert_result,
            "bert_score": bert_score,
            "bert_confidence": bert_result["confidence"],
            "phi3_used": False,
            "phi3_score": None,
            "phi3_reasoning": None,
            "phi3_signals": [],
            "decision_path": "distilbert_only",
        }

        # --- Niveau 2 : Phi-3 si zone d'incertitude + disponible ---
        if self._is_uncertain(bert_score) and self.phi3_available:
            logger.debug(
                f"Score DistilBERT ambigu ({bert_score:.2f}) "
                f"→ escalade vers Phi-3"
            )
            phi3_result = self.phi3_clf.predict_one(text)
            phi3_score = phi3_result["credibility_score"]
            phi3_conf = phi3_result["confidence"]

            # Si Phi-3 a échoué (confidence=0), on garde DistilBERT
            if phi3_conf > 0:
                combined = self._combine_scores(bert_score, phi3_score, phi3_conf)
                result.update({
                    "credibility_score": combined,
                    "is_fake": combined < CREDIBILITY_THRESHOLD,
                    "confidence": round((bert_result["confidence"] + phi3_conf) / 2, 4),
                    "phi3_used": True,
                    "phi3_score": phi3_score,
                    "phi3_reasoning": phi3_result.get("reasoning"),
                    "phi3_signals": phi3_result.get("signals", []),
                    "decision_path": "hybrid_bert_phi3",
                })
            else:
                result["decision_path"] = "distilbert_only_phi3_failed"

        result["total_latency_sec"] = round(time.time() - start, 2)
        return result

    def predict(self, texts: List[str]) -> List[Dict]:
        """Analyse une liste de textes."""
        results = []
        uncertain_count = 0

        for i, text in enumerate(texts):
            result = self.predict_one(text)
            results.append(result)
            if result["phi3_used"]:
                uncertain_count += 1

        phi3_rate = uncertain_count / max(len(texts), 1) * 100
        logger.info(
            f"Hybrid pipeline: {len(texts)} posts analysés | "
            f"DistilBERT seul: {len(texts)-uncertain_count} | "
            f"Phi-3 escaladé: {uncertain_count} ({phi3_rate:.0f}%)"
        )
        return results

    def get_stats(self, results: List[Dict]) -> Dict:
        """Statistiques d'utilisation du pipeline hybride."""
        if not results:
            return {}
        phi3_used = sum(1 for r in results if r.get("phi3_used", False))
        avg_bert = sum(r.get("bert_score", 0) for r in results) / len(results)
        avg_combined = sum(r.get("credibility_score", 0) for r in results) / len(results)
        avg_latency = sum(r.get("total_latency_sec", 0) for r in results) / len(results)

        return {
            "total_posts": len(results),
            "distilbert_only": len(results) - phi3_used,
            "phi3_escalated": phi3_used,
            "phi3_rate_pct": round(phi3_used / len(results) * 100, 1),
            "avg_bert_score": round(avg_bert, 3),
            "avg_combined_score": round(avg_combined, 3),
            "avg_latency_sec": round(avg_latency, 2),
            "fake_count": sum(1 for r in results if r.get("is_fake", False)),
            "real_count": sum(1 for r in results if not r.get("is_fake", False)),
        }

    @property
    def active_models(self) -> str:
        bert_name = self.bert_clf.active_model if self._bert_clf else "non chargé"
        phi3_status = "actif" if self.phi3_available else "inactif"
        return f"DistilBERT ({bert_name}) + Phi-3 Mini ({phi3_status})"


# ============================================================
# CLI TEST
# ============================================================

if __name__ == "__main__":
    print("🧪 Test de l'architecture hybride Thumalien\n")

    clf = HybridFakeNewsClassifier()
    print(f"Modèles : {clf.active_models}\n")

    test_cases = [
        # Cas clairs → DistilBERT devrait suffire
        "URGENT !!! COMPLOT révélé les élites vous mentent !!! Partagez avant censure !!!",
        "Rapport officiel du ministère de la santé publié selon les autorités sanitaires.",
        # Cas ambigus → Phi-3 devrait intervenir
        "Bien sûr que tout va bien dans ce pays... cherchez la vérité vous-mêmes.",
        "Intéressant... on dirait que les 'experts' ont encore changé d'avis.",
        # EN
        "Breaking: Scientists reveal they've been lying! Share before deleted!!!",
        "New peer-reviewed study in The Lancet shows promising results for gene therapy.",
    ]

    results = clf.predict(test_cases)

    print("─" * 70)
    for text, r in zip(test_cases, results):
        label = "🔴 FAKE" if r["is_fake"] else "🟢 RÉEL"
        phi3_tag = " [+Phi-3]" if r["phi3_used"] else ""
        print(f"{label} [{r['credibility_score']:.0%}]{phi3_tag}")
        print(f"  Texte  : {text[:65]}...")
        if r.get("phi3_reasoning"):
            print(f"  Phi-3  : {r['phi3_reasoning']}")
        print()

    stats = clf.get_stats(results)
    print("─" * 70)
    print(f"📊 Statistiques pipeline :")
    print(f"   DistilBERT seul  : {stats['distilbert_only']}/{stats['total_posts']} posts")
    print(f"   Phi-3 escaladé   : {stats['phi3_escalated']}/{stats['total_posts']} posts ({stats['phi3_rate_pct']}%)")
    print(f"   Latence moyenne  : {stats['avg_latency_sec']}s/post")
