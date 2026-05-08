"""
src/classifier/fake_news_classifier.py
Classification fake news avec deux niveaux :
  1. Baseline rapide : TF-IDF + LogisticRegression (toujours dispo)
  2. Modèle principal : DistilBERT multilingue fine-tuné (CPU-friendly)

Le modèle principal est sélectionné automatiquement selon les données disponibles.
"""
import json
import os
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils import shuffle

sys.path.append(str(Path(__file__).parents[2]))
from config import (
    CLASSIFIER_MODEL_NAME, CLASSIFIER_MAX_LENGTH, CLASSIFIER_BATCH_SIZE,
    CLASSIFIER_EPOCHS, CLASSIFIER_LEARNING_RATE, CLASSIFIER_TEST_SIZE,
    CLASSIFIER_OUTPUT_DIR, CREDIBILITY_THRESHOLD, LABELED_DIR, MODELS_DIR, LOG_FILE
)

logger.add(LOG_FILE, rotation="10 MB", level="INFO")


# ============================================================
# BASELINE : TF-IDF + LOGISTIC REGRESSION
# ============================================================

class BaselineClassifier:
    """
    Classificateur baseline léger.
    Fonctionne sans GPU et s'entraîne en quelques secondes.
    Utilisé pour tester le pipeline et comme fallback.
    """

    MODEL_PATH = MODELS_DIR / "baseline_classifier.pkl"

    def __init__(self):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=10_000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,
            )),
            ("clf", LogisticRegression(
                max_iter=1000,
                C=1.0,
                class_weight="balanced",
                random_state=42,
            )),
        ])
        self.is_fitted = False
        self.metrics: Dict = {}

    def train(self, texts: List[str], labels: List[int]) -> Dict:
        """Entraîne le pipeline TF-IDF + LogReg."""
        logger.info(f"Entraînement baseline sur {len(texts)} exemples...")

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels,
            test_size=CLASSIFIER_TEST_SIZE,
            random_state=42,
            stratify=labels,
        )

        self.pipeline.fit(X_train, y_train)
        self.is_fitted = True

        preds = self.pipeline.predict(X_test)
        probs = self.pipeline.predict_proba(X_test)[:, 1]

        self.metrics = {
            "accuracy": round(accuracy_score(y_test, preds), 4),
            "precision": round(precision_score(y_test, preds, zero_division=0), 4),
            "recall": round(recall_score(y_test, preds, zero_division=0), 4),
            "f1": round(f1_score(y_test, preds, zero_division=0), 4),
            "roc_auc": round(roc_auc_score(y_test, probs), 4) if len(set(y_test)) > 1 else 0.0,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "model": "TF-IDF + LogisticRegression",
            "trained_at": datetime.now().isoformat(),
        }

        print(f"\n📊 Baseline — Résultats:")
        print(f"   Accuracy : {self.metrics['accuracy']:.1%}")
        print(f"   F1-Score : {self.metrics['f1']:.1%}")
        print(f"   ROC-AUC  : {self.metrics['roc_auc']:.3f}")
        print(f"\n{classification_report(y_test, preds, target_names=['Fake', 'Réel'])}")

        self.save()
        return self.metrics

    def predict(self, texts: List[str]) -> List[Dict]:
        """Prédit la crédibilité d'une liste de textes."""
        if not self.is_fitted:
            self.load()

        probs = self.pipeline.predict_proba(texts)
        results = []
        for i, text in enumerate(texts):
            credibility = float(probs[i][1])  # Prob d'être RÉEL
            results.append({
                "text": text[:100],
                "credibility_score": round(credibility, 4),
                "is_fake": credibility < CREDIBILITY_THRESHOLD,
                "confidence": round(max(probs[i]), 4),
                "model": "baseline",
            })
        return results

    def predict_one(self, text: str) -> Dict:
        return self.predict([text])[0]

    def save(self):
        MODELS_DIR.mkdir(exist_ok=True)
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "metrics": self.metrics}, f)
        logger.info(f"Baseline sauvegardé: {self.MODEL_PATH}")

    def load(self):
        if not self.MODEL_PATH.exists():
            raise FileNotFoundError(f"Modèle non trouvé: {self.MODEL_PATH}. Entraînez d'abord avec train().")
        with open(self.MODEL_PATH, "rb") as f:
            data = pickle.load(f)
        self.pipeline = data["pipeline"]
        self.metrics = data.get("metrics", {})
        self.is_fitted = True
        logger.info("Baseline chargé.")


# ============================================================
# MODÈLE PRINCIPAL : DistilBERT Multilingue
# ============================================================

class BERTClassifier:
    """
    Classificateur basé sur DistilBERT multilingue.
    Fine-tunable sur CPU (modèle distil = 2x plus rapide que BERT).
    Charge le modèle pré-entraîné ou fine-tuné si disponible.
    """

    MODEL_SAVE_PATH = Path(CLASSIFIER_OUTPUT_DIR) / "best_model"
    METRICS_PATH = Path(CLASSIFIER_OUTPUT_DIR) / "metrics.json"

    def __init__(self, model_name: str = CLASSIFIER_MODEL_NAME):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.is_fitted = False
        self.metrics: Dict = {}

    def _lazy_import(self):
        """Import PyTorch/Transformers seulement si nécessaire (évite erreurs si absent)."""
        try:
            import torch
            from transformers import (
                AutoTokenizer, AutoModelForSequenceClassification,
                TrainingArguments, Trainer, EarlyStoppingCallback
            )
            return torch, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
        except ImportError as e:
            raise ImportError(
                "PyTorch/Transformers non installés. "
                "Exécutez: pip install torch transformers\n"
                f"Erreur: {e}"
            )

    def load_pretrained(self) -> bool:
        """Charge le modèle fine-tuné si disponible, sinon le modèle de base."""
        torch, AutoTokenizer, AutoModelForSequenceClassification, *_ = self._lazy_import()

        source = self.MODEL_SAVE_PATH if self.MODEL_SAVE_PATH.exists() else self.model_name

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(source))
            self.model = AutoModelForSequenceClassification.from_pretrained(
                str(source), num_labels=2
            )
            self.model.eval()
            self.is_fitted = True

            if self.METRICS_PATH.exists():
                with open(self.METRICS_PATH) as f:
                    self.metrics = json.load(f)

            logger.info(f"Modèle BERT chargé depuis: {source}")
            return True
        except Exception as e:
            logger.error(f"Impossible de charger le modèle BERT: {e}")
            return False

    def train(self, texts: List[str], labels: List[int]) -> Dict:
        """
        Fine-tune DistilBERT sur les données labellisées.
        Optimisé CPU : batch_size réduit, fp16 désactivé.
        """
        torch, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback = self._lazy_import()

        logger.info(f"Fine-tuning {self.model_name} sur {len(texts)} exemples...")
        print(f"\n🤖 Fine-tuning {self.model_name}")
        print(f"   Device : {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print(f"   Cela peut prendre 10-30 min sur CPU selon le dataset...\n")

        # Dataset PyTorch
        class _DS(torch.utils.data.Dataset):
            def __init__(self, enc, labs):
                self.enc = enc
                self.labs = labs
            def __len__(self): return len(self.labs)
            def __getitem__(self, i):
                return {k: v[i] for k, v in self.enc.items()} | {"labels": torch.tensor(self.labs[i])}

        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=CLASSIFIER_TEST_SIZE,
            random_state=42, stratify=labels
        )

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )

        def _encode(txts):
            return tokenizer(
                txts, padding=True, truncation=True,
                max_length=CLASSIFIER_MAX_LENGTH, return_tensors="pt"
            )

        train_ds = _DS(_encode(X_train), y_train)
        test_ds = _DS(_encode(X_test), y_test)

        def _metrics(pred):
            labs = pred.label_ids
            preds = pred.predictions.argmax(-1)
            return {
                "accuracy": accuracy_score(labs, preds),
                "f1": f1_score(labs, preds, zero_division=0),
                "precision": precision_score(labs, preds, zero_division=0),
                "recall": recall_score(labs, preds, zero_division=0),
            }

        os.makedirs(CLASSIFIER_OUTPUT_DIR, exist_ok=True)
        args = TrainingArguments(
            output_dir=CLASSIFIER_OUTPUT_DIR,
            num_train_epochs=CLASSIFIER_EPOCHS,
            per_device_train_batch_size=CLASSIFIER_BATCH_SIZE,
            per_device_eval_batch_size=CLASSIFIER_BATCH_SIZE,
            learning_rate=CLASSIFIER_LEARNING_RATE,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            save_total_limit=2,
            warmup_steps=50,
            fp16=False,  # Désactivé pour CPU
            logging_steps=20,
            report_to="none",
        )

        trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=test_ds,
            compute_metrics=_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )

        result = trainer.train()
        eval_res = trainer.evaluate()

        # Sauvegarder
        self.MODEL_SAVE_PATH.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(self.MODEL_SAVE_PATH))
        tokenizer.save_pretrained(str(self.MODEL_SAVE_PATH))

        self.model = model
        self.tokenizer = tokenizer
        self.is_fitted = True

        self.metrics = {
            "model": self.model_name,
            "accuracy": round(eval_res.get("eval_accuracy", 0), 4),
            "f1": round(eval_res.get("eval_f1", 0), 4),
            "precision": round(eval_res.get("eval_precision", 0), 4),
            "recall": round(eval_res.get("eval_recall", 0), 4),
            "train_loss": round(result.training_loss, 4),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "trained_at": datetime.now().isoformat(),
        }

        with open(self.METRICS_PATH, "w") as f:
            json.dump(self.metrics, f, indent=2)

        print(f"\n✅ Fine-tuning terminé:")
        print(f"   F1-Score : {self.metrics['f1']:.1%}")
        print(f"   Accuracy : {self.metrics['accuracy']:.1%}")

        return self.metrics

    def predict(self, texts: List[str]) -> List[Dict]:
        """Prédit la crédibilité d'une liste de textes."""
        import torch
        if not self.is_fitted:
            self.load_pretrained()

        results = []
        for text in texts:
            inputs = self.tokenizer(
                text, return_tensors="pt",
                truncation=True, max_length=CLASSIFIER_MAX_LENGTH, padding=True
            )
            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=1)[0].numpy()

            credibility = float(probs[1])
            results.append({
                "text": text[:100],
                "credibility_score": round(credibility, 4),
                "is_fake": credibility < CREDIBILITY_THRESHOLD,
                "confidence": round(float(max(probs)), 4),
                "prob_fake": round(float(probs[0]), 4),
                "prob_real": round(float(probs[1]), 4),
                "model": "distilbert-multilingual",
            })
        return results

    def predict_one(self, text: str) -> Dict:
        return self.predict([text])[0]


# ============================================================
# INTERFACE UNIFIÉE
# ============================================================

class FakeNewsClassifier:
    """
    Interface unifiée : charge le meilleur modèle disponible.
    Ordre de priorité : BERT fine-tuné > Baseline TF-IDF
    """

    def __init__(self, prefer_bert: bool = True):
        self._bert = BERTClassifier()
        self._baseline = BaselineClassifier()
        self._active = None
        self._prefer_bert = prefer_bert

    def load(self):
        """Charge le meilleur modèle disponible."""
        if self._prefer_bert and self._bert.load_pretrained():
            self._active = self._bert
            print("✅ Modèle BERT multilingue chargé")
        elif BaselineClassifier.MODEL_PATH.exists():
            self._baseline.load()
            self._active = self._baseline
            print("✅ Modèle baseline TF-IDF chargé")
        else:
            raise RuntimeError(
                "Aucun modèle disponible. Entraînez d'abord:\n"
                "  python -m src.classifier.fake_news_classifier --train"
            )

    def predict(self, texts: List[str]) -> List[Dict]:
        if self._active is None:
            self.load()
        return self._active.predict(texts)

    def predict_one(self, text: str) -> Dict:
        return self.predict([text])[0]

    @property
    def active_model(self) -> str:
        if self._active is None:
            return "non chargé"
        return getattr(self._active, "model_name", "baseline")

    @property
    def metrics(self) -> Dict:
        if self._active is None:
            return {}
        return self._active.metrics


# ============================================================
# DONNÉES D'ENTRAÎNEMENT & LABELISATION
# ============================================================

def load_labeled_data(path: Optional[Path] = None) -> Tuple[List[str], List[int]]:
    """
    Charge les données labellisées depuis un fichier JSON.
    Format attendu: [{"text": "...", "label": 0|1}, ...]
      0 = Fake News  |  1 = Post Réel
    """
    path = path or (LABELED_DIR / "labeled_posts.json")

    if not path.exists():
        logger.warning(f"Fichier de labels non trouvé: {path}")
        return [], []

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["text"] for item in data if "text" in item and "label" in item]
    labels = [int(item["label"]) for item in data if "text" in item and "label" in item]

    logger.info(f"{len(texts)} exemples chargés — Fake: {labels.count(0)}, Réel: {labels.count(1)}")
    return texts, labels


def create_sample_dataset() -> Tuple[List[str], List[int]]:
    """
    Génère un dataset d'exemples annotés pour tester le pipeline.
    À enrichir avec de vraies données collectées sur Bluesky.
    """
    fake_examples = [
        "URGENT !!! Le gouvernement cache la vérité sur ce scandale incroyable !",
        "ATTENTION DANGER ! Ne mangez surtout pas ce produit, risque mortel !!!",
        "RÉVÉLATION CHOC : Un complot mondial découvert par un scientifique courageux",
        "BREAKING NEWS: La vérité que les médias ne veulent pas vous dire !",
        "ALERTE IMMÉDIATE : Partagez vite avant que ce message soit censuré !",
        "INCROYABLE mais vrai ! Personne n'en parle dans les journaux mainstream",
        "COMPLOT révélé : Les élites manipulent la population depuis 50 ans",
        "SCANDALE étouffé par les médias mainstream, la preuve en vidéo exclusive",
        "URGENT à partager ! Votre santé est en danger, big pharma vous ment",
        "EXPLOSION imminente de la bulle financière, sauvez votre argent maintenant",
        "They are hiding the truth about what really happened that night",
        "SHOCKING: Government officials caught lying about this major incident",
        "The media won't show you this! Share before it gets deleted!!!",
        "BREAKING: Scientists reveal they have been lying to us for decades",
        "URGENT: This banned treatment cures everything, doctors don't want you to know",
        "Massive cover-up exposed! The elite don't want you to see this video",
        "WARN EVERYONE: This food is secretly poisoning millions right now",
        "They deleted this 3 times already! Share immediately before it disappears",
    ]

    real_examples = [
        "Intéressant article sur les avancées récentes en intelligence artificielle",
        "Le nouveau rapport du GIEC met en lumière l'urgence de la crise climatique",
        "Débat constructif sur les politiques publiques à la radio ce matin",
        "Étude scientifique publiée dans Nature sur la biodiversité marine",
        "Conférence passionnante sur l'histoire de l'architecture gothique",
        "Les résultats de l'enquête montrent une tendance intéressante",
        "Discussion nuancée sur les enjeux économiques actuels",
        "Rapport officiel des autorités sanitaires disponible en ligne",
        "Analyse approfondie des données statistiques de l'INSEE",
        "Présentation académique sur les innovations technologiques",
        "The new study published in Nature explores marine ecosystem dynamics",
        "Official report released by health authorities on vaccination coverage",
        "Researchers present findings at annual data science conference today",
        "Analysis of recent economic data shows steady growth in the sector",
        "Scientists publish peer-reviewed research on climate adaptation strategies",
        "University researchers explore new approaches to renewable energy storage",
        "Government releases official statistics on employment for the quarter",
        "Medical professionals discuss new treatment protocols in academic journal",
    ]

    texts = fake_examples + real_examples
    labels = [0] * len(fake_examples) + [1] * len(real_examples)
    texts, labels = shuffle(texts, labels, random_state=42)

    # Sauvegarder
    LABELED_DIR.mkdir(exist_ok=True)
    sample_path = LABELED_DIR / "labeled_posts_sample.json"
    with open(sample_path, "w", encoding="utf-8") as f:
        json.dump([{"text": t, "label": l} for t, l in zip(texts, labels)], f,
                  ensure_ascii=False, indent=2)

    print(f"✅ Dataset d'exemple créé: {sample_path} ({len(texts)} exemples)")
    return texts, labels


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classificateur Fake News Thumalien")
    parser.add_argument("--train", action="store_true", help="Entraîner le modèle")
    parser.add_argument("--model", choices=["baseline", "bert"], default="baseline",
                        help="Modèle à utiliser (baseline=rapide, bert=précis)")
    parser.add_argument("--sample", action="store_true",
                        help="Utiliser le dataset d'exemple (si pas de données labellisées)")
    parser.add_argument("--predict", type=str, default=None,
                        help="Texte à classifier")
    args = parser.parse_args()

    if args.train:
        texts, labels = load_labeled_data()
        if not texts or args.sample:
            print("⚠️  Pas de données labellisées, utilisation du dataset d'exemple.")
            texts, labels = create_sample_dataset()

        if args.model == "bert":
            clf = BERTClassifier()
            clf.train(texts, labels)
        else:
            clf = BaselineClassifier()
            clf.train(texts, labels)

    if args.predict:
        clf = FakeNewsClassifier(prefer_bert=(args.model == "bert"))
        result = clf.predict_one(args.predict)
        label = "🔴 FAKE NEWS" if result["is_fake"] else "🟢 POST RÉEL"
        print(f"\n{label}")
        print(f"  Score de crédibilité : {result['credibility_score']:.1%}")
        print(f"  Confiance            : {result['confidence']:.1%}")
