"""
tests/test_pipeline.py
Tests unitaires du pipeline Thumalien.
"""
import sys
import json
import pytest
from pathlib import Path

sys.path.append(str(Path(__file__).parents[1]))


# ============================================================
# TEST PREPROCESSING
# ============================================================
class TestTextPreprocessor:
    def setup_method(self):
        from src.preprocessing.text_preprocessor import TextPreprocessor
        self.preprocessor_fr = TextPreprocessor(language="fr")
        self.preprocessor_en = TextPreprocessor(language="en")

    def test_remove_urls(self):
        text = "Voir https://example.com pour plus d'infos"
        result = self.preprocessor_fr.remove_urls(text)
        assert "https://" not in result

    def test_remove_mentions(self):
        text = "Bonjour @user123 comment ça va ?"
        result = self.preprocessor_fr.remove_mentions(text)
        assert "@user123" not in result

    def test_clean_hashtags(self):
        text = "#FakeNews propagée sur les réseaux"
        result = self.preprocessor_fr.clean_hashtags(text, keep_text=True)
        assert "#" not in result
        assert "FakeNews" in result

    def test_remove_emojis(self):
        text = "Bonjour 😊 comment ça va 🚀 ?"
        result = self.preprocessor_fr.remove_emojis(text)
        assert "😊" not in result
        assert "🚀" not in result

    def test_normalize_whitespace(self):
        text = "trop    d'espaces    ici"
        result = self.preprocessor_fr.normalize_whitespace(text)
        assert "  " not in result

    def test_normalize_punctuation(self):
        text = "URGENT !!!! Incroyable ????"
        result = self.preprocessor_fr.normalize_punctuation(text)
        assert "!!!!" not in result
        assert "????" not in result

    def test_tokenize(self):
        text = "le gouvernement cache la vérité"
        tokens = self.preprocessor_fr.tokenize(text)
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_preprocess_returns_list(self):
        text = "URGENT !!! Le gouvernement cache la vérité sur ce #scandale !!!"
        tokens = self.preprocessor_fr.preprocess(text)
        assert isinstance(tokens, list)

    def test_preprocess_to_text_returns_string(self):
        text = "Article intéressant sur la désinformation"
        result = self.preprocessor_fr.preprocess_to_text(text)
        assert isinstance(result, str)

    def test_preprocess_en(self):
        text = "BREAKING NEWS: They are hiding the truth! Share before censored @user"
        tokens = self.preprocessor_en.preprocess(text)
        assert isinstance(tokens, list)

    def test_empty_text(self):
        tokens = self.preprocessor_fr.preprocess("")
        assert tokens == []

    def test_none_text(self):
        tokens = self.preprocessor_fr.preprocess(None)
        assert tokens == []


# ============================================================
# TEST BATCH PREPROCESSING
# ============================================================
class TestPreprocessBatch:
    def test_preprocess_posts(self):
        from src.preprocessing.text_preprocessor import preprocess_posts
        posts = [
            {"id": "1", "text": "URGENT !!! Scandale incroyable !!!", "language": "fr"},
            {"id": "2", "text": "Interesting article about climate change", "language": "en"},
        ]
        result = preprocess_posts(posts)
        assert len(result) == 2
        assert "text_clean" in result[0]
        assert "tokens" in result[0]
        assert "token_count" in result[0]
        assert isinstance(result[0]["tokens"], list)

    def test_preprocess_preserves_fields(self):
        from src.preprocessing.text_preprocessor import preprocess_posts
        posts = [{"id": "test_1", "text": "Test texte", "language": "fr", "author": "user1"}]
        result = preprocess_posts(posts)
        assert result[0]["id"] == "test_1"
        assert result[0]["author"] == "user1"


# ============================================================
# TEST EMOTION ANALYZER
# ============================================================
class TestEmotionAnalyzer:
    def setup_method(self):
        from src.emotion.emotion_analyzer import EmotionAnalyzer
        self.analyzer = EmotionAnalyzer()

    def test_analyze_returns_dict(self):
        result = self.analyzer.analyze("Je suis très heureux aujourd'hui !", "fr")
        assert isinstance(result, dict)
        assert "emotion_label" in result
        assert "emotion_score" in result
        assert "vader_compound" in result

    def test_emotion_score_range(self):
        result = self.analyzer.analyze("URGENT DANGER ALERTE !!!", "fr")
        assert 0.0 <= result["emotion_score"] <= 1.0

    def test_vader_compound_range(self):
        result = self.analyzer.analyze("super article génial", "fr")
        assert -1.0 <= result["vader_compound"] <= 1.0

    def test_empty_text(self):
        result = self.analyzer.analyze("", "fr")
        assert result["emotion_label"] == "neutral"

    def test_fake_news_fear_signal(self):
        text = "URGENT ALERTE DANGER ce complot va vous tuer !!!"
        result = self.analyzer.analyze(text, "fr")
        assert result["emotion_label"] in ["fear", "anger", "surprise"]

    def test_positive_text(self):
        text = "Wonderful happy day, feeling great and joyful!"
        result = self.analyzer.analyze(text, "en")
        assert result["vader_compound"] >= 0

    def test_analyze_batch(self):
        posts = [
            {"id": "1", "text_clean": "urgence danger alerte", "language": "fr"},
            {"id": "2", "text_clean": "happy wonderful day", "language": "en"},
        ]
        enriched = self.analyzer.analyze_batch(posts)
        assert len(enriched) == 2
        assert "emotion_emotion_label" in enriched[0]

    def test_all_scores_present(self):
        result = self.analyzer.analyze("Texte de test", "fr")
        assert "all_scores" in result
        assert isinstance(result["all_scores"], dict)


# ============================================================
# TEST EXPLAINER
# ============================================================
class TestFakeNewsExplainer:
    def setup_method(self):
        from src.explainability.explainer import FakeNewsExplainer
        self.explainer = FakeNewsExplainer()

    def test_explain_returns_dict(self):
        result = self.explainer.explain("Texte normal", 0.8)
        assert isinstance(result, dict)
        assert "credibility_score" in result
        assert "summary" in result
        assert "fake_signals" in result
        assert "top_reasons" in result

    def test_high_credibility_no_fake(self):
        text = "Rapport officiel selon le ministère de la santé"
        result = self.explainer.explain(text, 0.9)
        assert result["credibility_score"] == 0.9
        assert not result["is_high_risk"]

    def test_low_credibility_is_risky(self):
        text = "URGENT !!! COMPLOT révélé ! Partagez avant censure !!!"
        result = self.explainer.explain(text, 0.1)
        assert result["is_high_risk"]
        assert len(result["fake_signals"]) > 0

    def test_censorship_appeal_detected(self):
        text = "Partagez avant que ce soit censuré !"
        result = self.explainer.explain(text, 0.3)
        assert "censure_anticipée" in result["fake_signals"]

    def test_urgency_detected(self):
        text = "URGENT ALERTE BREAKING NEWS !"
        result = self.explainer.explain(text, 0.2)
        assert "urgence" in result["fake_signals"]

    def test_top_reasons_list(self):
        text = "URGENT COMPLOT les élites vous cachent la vérité !!!"
        result = self.explainer.explain(text, 0.1)
        assert isinstance(result["top_reasons"], list)
        assert len(result["top_reasons"]) <= 3

    def test_explain_batch(self):
        posts = [
            {"id": "1", "text_original": "URGENT danger complot !!!", "credibility_score": 0.1},
            {"id": "2", "text_original": "Article scientifique publié dans Nature", "credibility_score": 0.9},
        ]
        results = self.explainer.explain_batch(posts)
        assert len(results) == 2
        assert "explanation" in results[0]


# ============================================================
# TEST CLASSIFIER (baseline uniquement, sans modèle entraîné)
# ============================================================
class TestBaselineClassifier:
    def test_train_and_predict(self):
        from src.classifier.fake_news_classifier import BaselineClassifier, create_sample_dataset
        texts, labels = create_sample_dataset()
        clf = BaselineClassifier()
        metrics = clf.train(texts, labels)

        assert "accuracy" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

        result = clf.predict_one("URGENT complot révélé les élites vous mentent !!!")
        assert "credibility_score" in result
        assert "is_fake" in result
        assert 0.0 <= result["credibility_score"] <= 1.0

    def test_predict_batch(self):
        from src.classifier.fake_news_classifier import BaselineClassifier, create_sample_dataset
        texts, labels = create_sample_dataset()
        clf = BaselineClassifier()
        clf.train(texts, labels)

        test_texts = [
            "URGENT complot révélé les élites vous mentent !!!",
            "Article scientifique publié dans Nature sur le climat",
        ]
        results = clf.predict(test_texts)
        assert len(results) == 2
        for r in results:
            assert "credibility_score" in r
            assert "is_fake" in r
            assert "confidence" in r

    def test_create_sample_dataset(self):
        from src.classifier.fake_news_classifier import create_sample_dataset
        texts, labels = create_sample_dataset()
        assert len(texts) == len(labels)
        assert len(texts) > 0
        assert 0 in labels
        assert 1 in labels

    def test_load_labeled_data_missing_file(self):
        from src.classifier.fake_news_classifier import load_labeled_data
        texts, labels = load_labeled_data(Path("/nonexistent/path.json"))
        assert texts == []
        assert labels == []


# ============================================================
# TEST DB CONNECTOR (mock — sans vraie DB)
# ============================================================
class TestDatabaseConnectorMock:
    def test_import(self):
        from src.database.db_connector import DatabaseConnector
        assert DatabaseConnector is not None

    def test_schema_sql_not_empty(self):
        from src.database.db_connector import SCHEMA_SQL
        assert "CREATE TABLE IF NOT EXISTS posts" in SCHEMA_SQL
        assert "CREATE TABLE IF NOT EXISTS predictions" in SCHEMA_SQL
        assert "CREATE TABLE IF NOT EXISTS emotion_analyses" in SCHEMA_SQL
        assert "CREATE TABLE IF NOT EXISTS energy_tracking" in SCHEMA_SQL


# ============================================================
# TEST CONFIG
# ============================================================
class TestConfig:
    def test_paths_exist_or_created(self):
        from config import RAW_DIR, PROCESSED_DIR, LABELED_DIR, MODELS_DIR, LOGS_DIR
        for d in [RAW_DIR, PROCESSED_DIR, LABELED_DIR, MODELS_DIR, LOGS_DIR]:
            assert d.exists(), f"Dossier manquant : {d}"

    def test_supported_languages(self):
        from config import SUPPORTED_LANGUAGES
        assert "fr" in SUPPORTED_LANGUAGES
        assert "en" in SUPPORTED_LANGUAGES

    def test_emotion_labels(self):
        from config import EMOTION_LABELS
        assert "joy" in EMOTION_LABELS
        assert "anger" in EMOTION_LABELS
        assert "fear" in EMOTION_LABELS
        assert "neutral" in EMOTION_LABELS

    def test_credibility_threshold(self):
        from config import CREDIBILITY_THRESHOLD
        assert 0.0 < CREDIBILITY_THRESHOLD < 1.0


# ============================================================
# POINT D'ENTRÉE
# ============================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
