"""
src/explainability/explainer.py
Explicabilité IA : pourquoi un post est classé fake news.
Approche : poids TF-IDF + analyse des mots-clés suspects.
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from loguru import logger

sys.path.append(str(Path(__file__).parents[2]))
from config import LOG_FILE

logger.add(LOG_FILE, rotation="10 MB", level="INFO")

# ============================================================
# SIGNAUX LINGUISTIQUES DES FAKE NEWS
# ============================================================

FAKE_NEWS_SIGNALS = {
    "urgence": {
        "patterns": [r"\bURGENT\b", r"\bALERTE\b", r"\bBREAKING\b", r"\bIMMÉDIAT\b"],
        "description": "Langage d'urgence artificielle pour créer la panique",
        "weight": 0.8,
    },
    "ponctuation_excessive": {
        "patterns": [r"!!!+", r"\?\?\?+", r"!!!?\s*!!!"],
        "description": "Ponctuation excessive typique des contenus sensationnalistes",
        "weight": 0.6,
    },
    "majuscules_excessives": {
        "patterns": [r"\b[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖÙÚÛÜ]{4,}\b"],
        "description": "Majuscules excessives pour attirer l'attention",
        "weight": 0.5,
    },
    "complot": {
        "patterns": [
            r"\bcomplot\b", r"\bconspiracy\b", r"\bcaché\b", r"\bhidden\b",
            r"\bélit[es]\b", r"\belite[s]?\b", r"\bNWO\b", r"\bdeep.?state\b",
        ],
        "description": "Vocabulaire conspirationniste",
        "weight": 0.75,
    },
    "censure_anticipée": {
        "patterns": [
            r"partage[zr]? avant", r"share before", r"avant (que|d'être) censur",
            r"before (it[' ]?s )?(deleted|removed|banned)",
        ],
        "description": "Appel à partager avant une supposée censure",
        "weight": 0.9,
    },
    "source_vague": {
        "patterns": [
            r"des sources? (fiables?|sûres?|confidentielles?)",
            r"selon (des experts?|des scientifiques?)",
            r"(les médias?|mainstream) (cache[nt]?|tais[ent]?)",
            r"media (won't|don't|doesn't) (show|tell|report)",
        ],
        "description": "Sources vagues ou accusations envers les médias",
        "weight": 0.7,
    },
    "chiffres_non_sourcés": {
        "patterns": [r"\d+\s*%\s*(des|of)\s*(gens|people|population|experts?)"],
        "description": "Statistiques sans source citée",
        "weight": 0.4,
    },
    "appel_émotionnel": {
        "patterns": [
            r"DANGER\s*(!|de mort)?", r"ATTENTION\s*!",
            r"risque mortel", r"deadly risk", r"votre (vie|santé) en danger",
        ],
        "description": "Appels émotionnels à la peur",
        "weight": 0.65,
    },
}

# Signaux de crédibilité (renforcent l'authenticité)
CREDIBILITY_SIGNALS = {
    "source_citée": {
        "patterns": [
            r"selon (le|la|les|l[\''])\s+\w+", r"d[\'']après \w+",
            r"source\s*:", r"via @\w+", r"publié dans",
        ],
        "description": "Source explicitement citée",
        "weight": -0.5,  # Négatif = réduit le score fake
    },
    "nuance": {
        "patterns": [
            r"\bselon\b", r"\bapparemment\b", r"\bprobablement\b",
            r"\bseems\b", r"\bapparently\b", r"\baccording to\b",
        ],
        "description": "Langage nuancé et prudent",
        "weight": -0.3,
    },
}


class FakeNewsExplainer:
    """
    Générateur d'explications pour les prédictions fake news.
    Analyse les signaux linguistiques du texte.
    """

    def explain(self, text: str, credibility_score: float) -> Dict:
        """
        Génère une explication complète pour un texte analysé.

        Args:
            text: texte original du post
            credibility_score: score de crédibilité (0=fake, 1=réel)

        Returns:
            dict avec signaux détectés, score global et résumé
        """
        fake_signals_found = self._detect_signals(text, FAKE_NEWS_SIGNALS)
        cred_signals_found = self._detect_signals(text, CREDIBILITY_SIGNALS)

        # Score de risque basé sur les signaux linguistiques
        signal_risk_score = sum(s["weight"] for s in fake_signals_found.values())
        signal_cred_score = sum(abs(s["weight"]) for s in cred_signals_found.values())
        signal_net_score = min(max(signal_risk_score - signal_cred_score, 0), 1.0)

        # Combiner score modèle + signaux linguistiques
        combined_risk = (1 - credibility_score) * 0.7 + signal_net_score * 0.3

        return {
            "credibility_score": round(credibility_score, 4),
            "signal_risk_score": round(signal_net_score, 4),
            "combined_risk_score": round(combined_risk, 4),
            "is_high_risk": combined_risk > 0.6,
            "fake_signals": fake_signals_found,
            "credibility_signals": cred_signals_found,
            "summary": self._generate_summary(
                text, credibility_score, fake_signals_found, cred_signals_found
            ),
            "top_reasons": self._top_reasons(fake_signals_found),
        }

    def _detect_signals(self, text: str, signal_dict: Dict) -> Dict[str, Dict]:
        """Détecte les signaux présents dans le texte."""
        found = {}
        for signal_name, config in signal_dict.items():
            matches = []
            for pattern in config["patterns"]:
                hits = re.findall(pattern, text, re.IGNORECASE)
                matches.extend(hits)

            if matches:
                found[signal_name] = {
                    "description": config["description"],
                    "weight": config["weight"],
                    "matches": list(set(matches))[:3],  # Max 3 exemples
                    "count": len(matches),
                }
        return found

    def _top_reasons(self, signals: Dict) -> List[str]:
        """Retourne les 3 principales raisons de suspicion."""
        sorted_signals = sorted(signals.items(), key=lambda x: x[1]["weight"], reverse=True)
        return [v["description"] for _, v in sorted_signals[:3]]

    def _generate_summary(
        self,
        text: str,
        credibility_score: float,
        fake_signals: Dict,
        cred_signals: Dict,
    ) -> str:
        """Génère un résumé textuel de l'analyse."""
        n_fake = len(fake_signals)
        n_cred = len(cred_signals)

        if credibility_score >= 0.75 and n_fake == 0:
            return "✅ Ce contenu présente les caractéristiques d'un post fiable : langage nuancé, sources potentiellement citées, pas de signaux alarmistes."

        elif credibility_score >= 0.6:
            extras = ""
            if n_fake > 0:
                extras = f" Attention : {n_fake} signal(s) suspect(s) détecté(s)."
            return f"🟡 Ce contenu semble globalement fiable mais mérite vérification.{extras}"

        elif credibility_score >= 0.4:
            signal_list = ", ".join(list(fake_signals.keys())[:3]) if fake_signals else "aucun"
            return (
                f"🟠 Ce contenu présente un risque modéré de désinformation. "
                f"Signaux suspects : {signal_list}. "
                f"Vérifiez les sources avant de partager."
            )

        else:
            signal_list = ", ".join(list(fake_signals.keys())[:3]) if fake_signals else "style sensationnaliste"
            return (
                f"🔴 Ce contenu présente de forts marqueurs de fake news : {signal_list}. "
                f"Score de crédibilité très bas ({credibility_score:.0%}). "
                f"Ne partagez pas sans vérification par une source officielle."
            )

    def explain_batch(self, posts_with_predictions: List[Dict]) -> List[Dict]:
        """Génère des explications pour une liste de posts avec prédictions."""
        results = []
        for post in posts_with_predictions:
            text = post.get("text_original", post.get("text", ""))
            score = post.get("credibility_score", 0.5)
            explanation = self.explain(text, score)
            results.append({**post, "explanation": explanation})
        return results


# Instance globale
explainer = FakeNewsExplainer()


if __name__ == "__main__":
    test_cases = [
        ("URGENT !!! Le gouvernement cache la vérité sur ce scandale INCROYABLE ! Partagez avant censure !!!", 0.12),
        ("Rapport officiel des autorités sanitaires disponible en ligne selon le ministère de la santé", 0.87),
        ("Breaking news : cette révélation CHOC que les médias mainstream vous cachent !", 0.23),
    ]

    for text, score in test_cases:
        print("\n" + "=" * 60)
        print(f"📝 Texte: {text[:70]}...")
        result = explainer.explain(text, score)
        print(f"\n{result['summary']}")
        print(f"\n🎯 Score crédibilité : {result['credibility_score']:.1%}")
        print(f"⚠️  Score risque signaux : {result['signal_risk_score']:.1%}")
        if result["top_reasons"]:
            print(f"\n🔍 Raisons principales :")
            for reason in result["top_reasons"]:
                print(f"   • {reason}")
