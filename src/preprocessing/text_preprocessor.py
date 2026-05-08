"""
src/preprocessing/text_preprocessor.py
Pipeline NLP complet : nettoyage → tokenisation → lemmatisation.
Supporte le français et l'anglais.
"""
import re
import sys
from pathlib import Path
from typing import List, Optional

import emoji
import spacy
import nltk
from loguru import logger

# Téléchargement automatique des ressources NLTK
for resource in ["stopwords", "punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords

sys.path.append(str(Path(__file__).parents[2]))
from config import LOG_FILE

logger.add(LOG_FILE, rotation="10 MB", level="INFO")

# Chargement lazy des modèles spaCy
_nlp_cache = {}


def _get_nlp(language: str):
    if language not in _nlp_cache:
        model = "fr_core_news_sm" if language == "fr" else "en_core_web_sm"
        try:
            _nlp_cache[language] = spacy.load(model)
        except OSError:
            logger.error(f"Modèle spaCy '{model}' non installé. Exécutez: python -m spacy download {model}")
            raise
    return _nlp_cache[language]


_STOP_WORDS = {
    "fr": set(stopwords.words("french")),
    "en": set(stopwords.words("english")),
}


class TextPreprocessor:
    """
    Pipeline complet de prétraitement NLP.
    Chaque étape est indépendante et configurable.
    """

    def __init__(self, language: str = "fr"):
        """
        Args:
            language: 'fr' ou 'en'
        """
        if language not in ("fr", "en"):
            raise ValueError(f"Langue non supportée: {language}. Utilisez 'fr' ou 'en'.")
        self.language = language
        self._nlp = None  # Lazy loading

    @property
    def nlp(self):
        if self._nlp is None:
            self._nlp = _get_nlp(self.language)
        return self._nlp

    @property
    def stop_words(self):
        return _STOP_WORDS.get(self.language, set())

    # ----------------------------------------------------------
    # ÉTAPES DE NETTOYAGE
    # ----------------------------------------------------------

    @staticmethod
    def remove_urls(text: str) -> str:
        """Supprime les URLs."""
        text = re.sub(r"https?://\S+", "", text)
        text = re.sub(r"www\.\S+", "", text)
        return text

    @staticmethod
    def remove_mentions(text: str) -> str:
        """Supprime les @mentions."""
        return re.sub(r"@\w+", "", text)

    @staticmethod
    def clean_hashtags(text: str, keep_text: bool = True) -> str:
        """#FakeNews → 'FakeNews' (keep_text=True) ou '' (keep_text=False)."""
        if keep_text:
            return re.sub(r"#(\w+)", r"\1", text)
        return re.sub(r"#\w+", "", text)

    @staticmethod
    def remove_emojis(text: str) -> str:
        """Supprime les emojis."""
        return emoji.replace_emoji(text, replace="")

    @staticmethod
    def normalize_whitespace(text: str) -> str:
        """Normalise les espaces et sauts de ligne."""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def normalize_punctuation(text: str) -> str:
        """Réduit la ponctuation répétée (!!!! → !)."""
        text = re.sub(r"!{2,}", "!", text)
        text = re.sub(r"\?{2,}", "?", text)
        text = re.sub(r"\.{3,}", "...", text)
        return text

    @staticmethod
    def to_lowercase(text: str) -> str:
        return text.lower()

    def tokenize(self, text: str) -> List[str]:
        """Tokenise le texte en mots (sans ponctuation)."""
        return re.findall(r"\b\w+\b", text)

    def remove_stopwords(self, tokens: List[str]) -> List[str]:
        """Supprime les mots vides."""
        return [t for t in tokens if t not in self.stop_words and len(t) > 2]

    def lemmatize(self, tokens: List[str]) -> List[str]:
        """Lemmatise les tokens via spaCy."""
        doc = self.nlp(" ".join(tokens))
        return [token.lemma_ for token in doc if token.lemma_.strip()]

    # ----------------------------------------------------------
    # PIPELINE COMPLET
    # ----------------------------------------------------------

    def preprocess(
        self,
        text: str,
        remove_urls: bool = True,
        remove_mentions: bool = True,
        remove_hashtags_symbol: bool = True,
        remove_emojis_flag: bool = True,
        clean_punct: bool = True,
        lowercase: bool = True,
        remove_stops: bool = True,
        lemmatize_flag: bool = True,
    ) -> List[str]:
        """
        Pipeline complet de prétraitement.
        Retourne une liste de tokens nettoyés.
        """
        if not text or not isinstance(text, str):
            return []

        if remove_urls:
            text = self.remove_urls(text)
        if remove_mentions:
            text = self.remove_mentions(text)
        if remove_hashtags_symbol:
            text = self.clean_hashtags(text, keep_text=True)
        if remove_emojis_flag:
            text = self.remove_emojis(text)
        if clean_punct:
            text = self.normalize_punctuation(text)

        text = self.normalize_whitespace(text)

        if lowercase:
            text = self.to_lowercase(text)

        tokens = self.tokenize(text)

        if remove_stops:
            tokens = self.remove_stopwords(tokens)
        if lemmatize_flag:
            tokens = self.lemmatize(tokens)

        # Filtrer les tokens vides et les chiffres seuls
        tokens = [t for t in tokens if t.strip() and not t.isdigit()]

        return tokens

    def preprocess_to_text(self, text: str, **kwargs) -> str:
        """Même chose mais retourne une chaîne de caractères."""
        return " ".join(self.preprocess(text, **kwargs))


# ----------------------------------------------------------
# FONCTION DE BATCH
# ----------------------------------------------------------

def preprocess_posts(posts: List[dict]) -> List[dict]:
    """
    Prétraite une liste de posts et ajoute les champs 'text_clean' et 'tokens'.
    Détecte automatiquement la langue de chaque post.
    """
    preprocessors = {
        "fr": TextPreprocessor(language="fr"),
        "en": TextPreprocessor(language="en"),
    }
    fallback = TextPreprocessor(language="en")

    enriched = []
    for post in posts:
        lang = post.get("language", "en")
        preprocessor = preprocessors.get(lang, fallback)

        original_text = post.get("text", post.get("text_original", ""))
        tokens = preprocessor.preprocess(original_text)
        text_clean = " ".join(tokens)

        enriched.append({
            **post,
            "text_original": original_text,
            "text_clean": text_clean,
            "tokens": tokens,
            "token_count": len(tokens),
            "processed_at": __import__("datetime").datetime.now().isoformat(),
        })

    logger.info(f"Prétraitement de {len(enriched)} posts terminé.")
    return enriched


# ----------------------------------------------------------
# POINT D'ENTRÉE CLI
# ----------------------------------------------------------

if __name__ == "__main__":
    import json
    from config import RAW_DIR, PROCESSED_DIR

    # Trouver le fichier JSON le plus récent
    json_files = sorted(RAW_DIR.glob("*.json"), reverse=True)
    if not json_files:
        print("❌ Aucun fichier JSON dans data/raw/")
        print("   Lancez d'abord: python -m src.collector.bluesky_collector")
        sys.exit(1)

    input_file = json_files[0]
    print(f"📂 Traitement de: {input_file}")

    with open(input_file, encoding="utf-8") as f:
        posts = json.load(f)

    processed = preprocess_posts(posts)

    output_file = PROCESSED_DIR / (input_file.stem + "_processed.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    print(f"✅ {len(processed)} posts traités → {output_file}")

    # Exemple
    if processed:
        p = processed[0]
        print(f"\n📝 Exemple:")
        print(f"  Original : {p['text_original'][:80]}")
        print(f"  Clean    : {p['text_clean'][:80]}")
        print(f"  Tokens   : {p['tokens'][:8]}")
