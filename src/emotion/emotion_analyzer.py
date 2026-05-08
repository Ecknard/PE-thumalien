"""
src/emotion/emotion_analyzer.py
Analyse émotionnelle des posts Bluesky.

Approche hybride :
  - VADER Sentiment (rapide, multilingue approximatif)
  - Heuristiques lexicales pour les 7 émotions de base
  - Compatible FR et EN
"""
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

sys.path.append(str(Path(__file__).parents[2]))
from config import EMOTION_LABELS, LOG_FILE

logger.add(LOG_FILE, rotation="10 MB", level="INFO")

# ============================================================
# LEXIQUES ÉMOTIONNELS (FR + EN)
# ============================================================

EMOTION_LEXICON = {
    "joy": {
        "fr": ["heureux", "joie", "content", "super", "génial", "formidable", "excellent",
               "bravo", "magnifique", "fantastique", "incroyable", "merci", "amour", "fête"],
        "en": ["happy", "joy", "great", "awesome", "wonderful", "fantastic", "excellent",
               "brilliant", "amazing", "love", "celebrate", "congrats", "beautiful"],
    },
    "anger": {
        "fr": ["colère", "furieux", "énervé", "scandale", "honte", "inacceptable",
               "inadmissible", "révoltant", "outrageux", "injuste", "mensonge", "tromperie"],
        "en": ["angry", "furious", "outraged", "scandal", "shame", "unacceptable",
               "disgrace", "lie", "fraud", "corrupt", "ridiculous", "pathetic"],
    },
    "fear": {
        "fr": ["peur", "danger", "alerte", "urgent", "menace", "risque", "catastrophe",
               "attention", "avertissement", "inquiet", "anxieux", "effrayant"],
        "en": ["fear", "danger", "alert", "urgent", "threat", "risk", "catastrophe",
               "warning", "scared", "anxious", "terrifying", "beware"],
    },
    "sadness": {
        "fr": ["triste", "deuil", "perte", "malheur", "désolé", "tragédie", "regret",
               "souffrance", "douleur", "mourir", "décès", "pleurer"],
        "en": ["sad", "grief", "loss", "unfortunate", "sorry", "tragedy", "regret",
               "suffering", "pain", "death", "crying", "heartbreak"],
    },
    "surprise": {
        "fr": ["incroyable", "choc", "révélation", "stupéfiant", "surprenant", "inattendu",
               "extraordinaire", "hallucinant", "wow", "jamais vu"],
        "en": ["incredible", "shocking", "revelation", "stunning", "surprising", "unexpected",
               "extraordinary", "unbelievable", "wow", "omg", "insane"],
    },
    "disgust": {
        "fr": ["dégoût", "répugnant", "honteux", "abject", "ignoble", "nauséabond",
               "corruption", "sale", "indigne", "horrible"],
        "en": ["disgusting", "repulsive", "shameful", "vile", "disgusted", "filthy",
               "corrupt", "nasty", "gross", "horrible"],
    },
}

# Indicateurs d'intensité émotionnelle liés aux fake news
FAKE_NEWS_EMOTION_PATTERNS = [
    (r"URGENT|ALERTE|BREAKING", "fear", 0.3),
    (r"!!!+", "anger", 0.2),
    (r"complot|conspiracy|cover.?up", "fear", 0.25),
    (r"mensonge|lie|fake|hoax", "anger", 0.2),
    (r"CHOC|RÉVÉLATION|SHOCKING", "surprise", 0.25),
    (r"scandale|scandal", "anger", 0.2),
]


class EmotionAnalyzer:
    """
    Analyseur d'émotions hybride VADER + heuristiques lexicales.
    """

    def __init__(self):
        self._vader = SentimentIntensityAnalyzer()

    def _vader_sentiment(self, text: str) -> Dict:
        """Analyse VADER : retourne compound, pos, neg, neu."""
        scores = self._vader.polarity_scores(text)
        return scores

    def _lexicon_scores(self, text: str, language: str = "fr") -> Dict[str, float]:
        """
        Calcule un score pour chaque émotion via matching lexical.
        Score = (nombre de mots matchés) / (longueur du texte normalisée)
        """
        text_lower = text.lower()
        words = re.findall(r"\b\w+\b", text_lower)
        n_words = max(len(words), 1)

        scores = {emotion: 0.0 for emotion in EMOTION_LEXICON}

        for emotion, lexicon in EMOTION_LEXICON.items():
            lang_words = lexicon.get(language, []) + lexicon.get("en", [])
            hits = sum(1 for w in words if w in lang_words)
            scores[emotion] = min(hits / n_words * 5, 1.0)  # Normalisé 0-1

        # Boost via patterns fake news
        for pattern, emotion, boost in FAKE_NEWS_EMOTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                scores[emotion] = min(scores[emotion] + boost, 1.0)

        return scores

    def _map_vader_to_emotion(self, compound: float, lexicon_scores: Dict[str, float]) -> str:
        """
        Détermine l'émotion dominante en combinant VADER et lexique.
        """
        # VADER compound : -1 (très négatif) → +1 (très positif)
        if compound >= 0.5:
            vader_emotion = "joy"
        elif compound <= -0.5:
            # Distinguer colère vs tristesse vs peur
            candidates = {k: lexicon_scores[k] for k in ["anger", "sadness", "fear", "disgust"]}
            vader_emotion = max(candidates, key=candidates.get)
            if max(candidates.values()) < 0.05:
                vader_emotion = "anger"  # Par défaut négatif fort
        elif -0.1 <= compound <= 0.1:
            vader_emotion = "neutral"
        else:
            # Zone intermédiaire : laisser le lexique décider
            non_neutral = {k: v for k, v in lexicon_scores.items() if k != "neutral"}
            if non_neutral and max(non_neutral.values()) > 0.05:
                vader_emotion = max(non_neutral, key=non_neutral.get)
            else:
                vader_emotion = "neutral"

        # Si le lexique est plus confiant que VADER, l'utiliser
        top_lexicon = max(lexicon_scores, key=lexicon_scores.get)
        if lexicon_scores[top_lexicon] > 0.3 and top_lexicon != vader_emotion:
            return top_lexicon

        return vader_emotion

    def analyze(self, text: str, language: str = "fr") -> Dict:
        """
        Analyse complète d'un texte.

        Returns:
            {
                emotion_label: str,   # émotion dominante (clé)
                emotion_name: str,    # nom affichable avec emoji
                emotion_score: float, # confiance 0-1
                vader_compound: float,
                all_scores: dict      # scores pour toutes les émotions
            }
        """
        if not text or not isinstance(text, str):
            return self._neutral_result()

        vader = self._vader_sentiment(text)
        lexicon = self._lexicon_scores(text, language=language)

        emotion_label = self._map_vader_to_emotion(vader["compound"], lexicon)

        # Score de confiance = max(lexicon_score, abs(compound))
        if emotion_label == "neutral":
            confidence = 1.0 - abs(vader["compound"])
        else:
            confidence = max(lexicon.get(emotion_label, 0.0), abs(vader["compound"]))

        return {
            "emotion_label": emotion_label,
            "emotion_name": EMOTION_LABELS.get(emotion_label, emotion_label),
            "emotion_score": round(min(confidence, 1.0), 4),
            "vader_compound": round(vader["compound"], 4),
            "vader_positive": round(vader["pos"], 4),
            "vader_negative": round(vader["neg"], 4),
            "vader_neutral": round(vader["neu"], 4),
            "all_scores": {k: round(v, 4) for k, v in lexicon.items()},
        }

    def _neutral_result(self) -> Dict:
        return {
            "emotion_label": "neutral",
            "emotion_name": EMOTION_LABELS["neutral"],
            "emotion_score": 1.0,
            "vader_compound": 0.0,
            "vader_positive": 0.0,
            "vader_negative": 0.0,
            "vader_neutral": 1.0,
            "all_scores": {k: 0.0 for k in EMOTION_LEXICON},
        }

    def analyze_batch(self, posts: List[Dict]) -> List[Dict]:
        """
        Analyse émotionnelle d'une liste de posts.
        Ajoute les résultats à chaque post et retourne la liste enrichie.
        """
        enriched = []
        for post in posts:
            text = post.get("text_clean") or post.get("text", "")
            lang = post.get("language", "fr")
            analysis = self.analyze(text, language=lang)
            enriched.append({**post, **{f"emotion_{k}": v for k, v in analysis.items()}})

        logger.info(f"Analyse émotionnelle de {len(enriched)} posts terminée.")
        return enriched

    def get_distribution(self, analyses: List[Dict]) -> Dict[str, int]:
        """Retourne la distribution des émotions dans une liste d'analyses."""
        distribution = {label: 0 for label in EMOTION_LEXICON}
        distribution["neutral"] = 0

        for a in analyses:
            label = a.get("emotion_label", "neutral")
            distribution[label] = distribution.get(label, 0) + 1

        return distribution


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    analyzer = EmotionAnalyzer()

    examples = [
        ("URGENT !!! Le gouvernement cache la vérité sur ce scandale !!!", "fr"),
        ("Intéressant article scientifique sur les avancées en médecine", "fr"),
        ("They are hiding the truth! Share before it gets deleted!", "en"),
        ("Beautiful day for a walk in the park, feeling grateful", "en"),
        ("Rapport officiel des autorités sanitaires disponible en ligne", "fr"),
    ]

    print("=" * 60)
    print("ANALYSE ÉMOTIONNELLE — THUMALIEN")
    print("=" * 60)

    for text, lang in examples:
        result = analyzer.analyze(text, language=lang)
        print(f"\n📝 [{lang.upper()}] {text[:60]}...")
        print(f"   {result['emotion_name']} (score: {result['emotion_score']:.2f})")
        print(f"   VADER compound: {result['vader_compound']:+.3f}")
