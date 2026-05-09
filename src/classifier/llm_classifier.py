"""
src/classifier/llm_classifier.py
Classificateur Fake News basé sur Phi-3 Mini via Ollama.
Fonctionne entièrement sur CPU — aucun GPU requis.

Fonctionnement :
  - Phi-3 Mini (3.8B paramètres, quantisé Q4) tourne via Ollama (serveur local)
  - Prompt few-shot en FR/EN avec format de réponse JSON strict
  - Capable de détecter l'ironie, le sarcasme et la désinformation "soft"
    que DistilBERT rate (pas de marqueurs évidents)
  - Retourne un score de crédibilité + un raisonnement en langage naturel

Prérequis :
  - Ollama installé : https://ollama.com/download
  - Modèle téléchargé : ollama pull phi3:mini
  - Serveur lancé : ollama serve  (ou démarrage automatique)
"""
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests
from loguru import logger

sys.path.append(str(Path(__file__).parents[2]))
from config import (
    LLM_MODEL_NAME, LLM_OLLAMA_URL,
    LLM_TIMEOUT_SECONDS, LLM_TEMPERATURE, LLM_MAX_TOKENS,
    CREDIBILITY_THRESHOLD, LOG_FILE
)

logger.add(LOG_FILE, rotation="10 MB", level="INFO")


# ============================================================
# PROMPT ENGINEERING
# ============================================================

SYSTEM_PROMPT = """Tu es un expert en détection de désinformation et fact-checking, spécialisé sur les réseaux sociaux.
Ton rôle est d'analyser des posts Bluesky et d'évaluer leur crédibilité.

RÈGLES ABSOLUES :
1. Tu réponds UNIQUEMENT en JSON valide, rien d'autre.
2. Le JSON doit contenir exactement ces champs :
   - "credibility_score": nombre entre 0.0 (fake news certain) et 1.0 (fiable)
   - "is_fake": true si score < 0.5, false sinon
   - "confidence": ta confiance dans la décision (0.0 à 1.0)
   - "reasoning": explication courte (max 2 phrases) en français
   - "signals": liste des signaux suspects détectés (liste vide si aucun)

CRITÈRES D'ÉVALUATION — Posts suspects (score faible) :
- Langage d'urgence artificielle (URGENT, ALERTE, BREAKING NEWS)
- Appel à partager avant censure supposée
- Vocabulaire complotiste (élites, NWO, deep state, ils vous cachent)
- Majuscules et ponctuation excessives (!!!, ???)
- Sources vagues ou absentes ("des experts disent", "on m'a dit que")
- Statistiques sans source
- Ironie ou sarcasme camouflant une désinformation
- Contenu émotionnellement manipulatoire sans fond factuel

CRITÈRES D'ÉVALUATION — Posts fiables (score élevé) :
- Sources explicitement citées (étude, ministère, journal de référence)
- Langage nuancé (selon, apparemment, des recherches suggèrent)
- Contexte factuel vérifiable
- Pas de marqueurs alarmistes"""

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": 'Analyse ce post Bluesky : "URGENT !!! Le gouvernement cache la vérité sur ce scandale incroyable ! Partagez avant que ce soit censuré !!!"'
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "credibility_score": 0.05,
            "is_fake": True,
            "confidence": 0.95,
            "reasoning": "Cumul de signaux classiques de désinformation : urgence artificielle, appel à partager avant censure, source absente, ponctuation excessive. Aucun fait vérifiable.",
            "signals": ["urgence_artificielle", "appel_censure", "source_absente", "ponctuation_excessive"]
        }, ensure_ascii=False)
    },
    {
        "role": "user",
        "content": 'Analyse ce post Bluesky : "Bien sûr que les vaccins sont totalement sûrs... c\'est ce qu\'ils veulent vous faire croire. Cherchez la vérité vous-mêmes."'
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "credibility_score": 0.12,
            "is_fake": True,
            "confidence": 0.88,
            "reasoning": "Désinformation soft par ironie et insinuation : le message ne dit rien de factuel mais sous-entend un complot sans preuve. Vocabulaire 'ils' vague et appel à 'chercher la vérité' typique des théories complotistes.",
            "signals": ["ironie_sarcasme", "insinuation_complot", "appel_vague", "source_absente"]
        }, ensure_ascii=False)
    },
    {
        "role": "user",
        "content": 'Analyse ce post Bluesky : "Selon une étude publiée dans The Lancet (2024), les nouvelles thérapies géniques montrent une efficacité de 87% sur les essais cliniques de phase 3."'
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "credibility_score": 0.91,
            "is_fake": False,
            "confidence": 0.87,
            "reasoning": "Source académique précise citée (The Lancet, 2024), chiffres spécifiques et méthodologie mentionnée (phase 3). Langage factuel sans marqueurs alarmistes.",
            "signals": []
        }, ensure_ascii=False)
    },
    {
        "role": "user",
        "content": 'Analyse ce post Bluesky : "Breaking: Scientists reveal they\'ve been lying to us for decades! The media won\'t show this! Share before deleted!!!"'
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "credibility_score": 0.04,
            "is_fake": True,
            "confidence": 0.97,
            "reasoning": "Tous les marqueurs classiques de fake news en anglais : Breaking news non sourcé, accusation générale de mensonge, appel à partager avant suppression, aucun fait vérifiable.",
            "signals": ["urgence_artificielle", "appel_censure", "accusation_media", "source_absente", "ponctuation_excessive"]
        }, ensure_ascii=False)
    },
]


# ============================================================
# CLASSE PRINCIPALE
# ============================================================

class PhiClassifier:
    """
    Classificateur Fake News basé sur Phi-3 Mini via Ollama.

    Avantages vs DistilBERT :
      + Détecte l'ironie, le sarcasme, la désinformation "soft"
      + Produit un raisonnement en langage naturel
      + Zero-shot : pas de fine-tuning requis sur des données labellisées
      + Multilingue natif (FR, EN, et bien d'autres)

    Inconvénients vs DistilBERT :
      - 5 à 20x plus lent en inférence CPU
      - Nécessite Ollama installé (dépendance externe)
      - RAM : ~4.5 Go pour Phi-3 Mini Q4
    """

    def __init__(
        self,
        model_name: str = LLM_MODEL_NAME,
        ollama_url: str = LLM_OLLAMA_URL,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self._available: Optional[bool] = None

    # ----------------------------------------------------------
    # VÉRIFICATION OLLAMA
    # ----------------------------------------------------------

    def is_available(self) -> bool:
        """Vérifie qu'Ollama tourne et que le modèle est disponible."""
        if self._available is not None:
            return self._available
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if resp.status_code != 200:
                self._available = False
                return False
            models = [m["name"] for m in resp.json().get("models", [])]
            # Vérifier si le modèle (ou une variante) est présent
            base = self.model_name.split(":")[0]
            self._available = any(base in m for m in models)
            if not self._available:
                logger.warning(
                    f"Ollama disponible mais modèle '{self.model_name}' absent. "
                    f"Modèles installés : {models}. "
                    f"Lancez : ollama pull {self.model_name}"
                )
            return self._available
        except requests.exceptions.ConnectionError:
            self._available = False
            logger.warning(
                f"Ollama non joignable sur {self.ollama_url}. "
                "Lancez : ollama serve"
            )
            return False

    # ----------------------------------------------------------
    # CONSTRUCTION DU PROMPT
    # ----------------------------------------------------------

    def _build_messages(self, text: str) -> List[Dict]:
        """
        Construit les messages pour l'API Ollama /api/chat.
        Utilise le few-shot pour guider le format de réponse.
        """
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(FEW_SHOT_EXAMPLES)
        messages.append({
            "role": "user",
            "content": f'Analyse ce post Bluesky : "{text[:500]}"'
        })
        return messages

    # ----------------------------------------------------------
    # APPEL API OLLAMA
    # ----------------------------------------------------------

    def _call_ollama(self, messages: List[Dict]) -> Optional[str]:
        """Appelle l'API Ollama /api/chat et retourne le contenu brut."""
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": LLM_TEMPERATURE,
                "num_predict": LLM_MAX_TOKENS,
            }
        }
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/chat",
                json=payload,
                timeout=LLM_TIMEOUT_SECONDS,
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]
        except requests.exceptions.Timeout:
            logger.error(f"Timeout Ollama après {LLM_TIMEOUT_SECONDS}s")
            return None
        except Exception as e:
            logger.error(f"Erreur appel Ollama: {e}")
            return None

    # ----------------------------------------------------------
    # PARSING JSON
    # ----------------------------------------------------------

    def _parse_response(self, raw: str, text: str) -> Dict:
        """
        Parse la réponse JSON de Phi-3.
        Robuste : extrait le JSON même si le modèle ajoute du texte autour.
        """
        if not raw:
            return self._fallback_result(text, reason="réponse vide Ollama")

        # Nettoyage : extraire le bloc JSON si entouré de texte
        raw = raw.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        # Trouver le premier { et le dernier }
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return self._fallback_result(text, reason="JSON non trouvé dans la réponse")

        try:
            data = json.loads(raw[start:end])
            # Validation et normalisation
            credibility = float(data.get("credibility_score", 0.5))
            credibility = max(0.0, min(1.0, credibility))
            confidence = float(data.get("confidence", 0.8))
            confidence = max(0.0, min(1.0, confidence))

            return {
                "credibility_score": round(credibility, 4),
                "is_fake": credibility < CREDIBILITY_THRESHOLD,
                "confidence": round(confidence, 4),
                "reasoning": data.get("reasoning", "Analyse effectuée par Phi-3."),
                "signals": data.get("signals", []),
                "model": f"phi3-ollama ({self.model_name})",
                "raw_response": raw[start:end],
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Erreur parsing JSON Phi-3: {e} | Réponse: {raw[:200]}")
            return self._fallback_result(text, reason=f"JSON invalide: {e}")

    def _fallback_result(self, text: str, reason: str = "") -> Dict:
        """Résultat par défaut si Phi-3 échoue."""
        logger.warning(f"Fallback Phi-3 ({reason}) — texte: {text[:50]}")
        return {
            "credibility_score": 0.5,
            "is_fake": False,
            "confidence": 0.0,
            "reasoning": f"Analyse LLM indisponible ({reason}). Score neutre appliqué.",
            "signals": [],
            "model": "phi3-fallback",
            "error": reason,
        }

    # ----------------------------------------------------------
    # INTERFACE PUBLIQUE
    # ----------------------------------------------------------

    def predict_one(self, text: str) -> Dict:
        """Analyse un seul texte."""
        if not self.is_available():
            return self._fallback_result(text, reason="Ollama non disponible")

        start = time.time()
        messages = self._build_messages(text)
        raw = self._call_ollama(messages)
        result = self._parse_response(raw, text)
        result["latency_sec"] = round(time.time() - start, 2)

        logger.debug(
            f"Phi-3 | score={result['credibility_score']:.2f} | "
            f"fake={result['is_fake']} | latency={result['latency_sec']}s"
        )
        return result

    def predict(self, texts: List[str]) -> List[Dict]:
        """Analyse une liste de textes séquentiellement."""
        results = []
        for i, text in enumerate(texts):
            logger.debug(f"Phi-3 analyse {i+1}/{len(texts)}")
            results.append(self.predict_one(text))
        return results

    def get_model_info(self) -> Dict:
        """Retourne les informations sur le modèle Ollama."""
        if not self.is_available():
            return {"available": False, "model": self.model_name}
        try:
            resp = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            models = resp.json().get("models", [])
            base = self.model_name.split(":")[0]
            info = next((m for m in models if base in m["name"]), {})
            return {
                "available": True,
                "model": self.model_name,
                "size_gb": round(info.get("size", 0) / 1e9, 2),
                "ollama_url": self.ollama_url,
            }
        except Exception:
            return {"available": True, "model": self.model_name}


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Classificateur Phi-3 Thumalien")
    parser.add_argument("--check", action="store_true", help="Vérifier qu'Ollama est dispo")
    parser.add_argument("--predict", type=str, help="Texte à analyser")
    parser.add_argument("--benchmark", action="store_true", help="Test sur les exemples few-shot")
    args = parser.parse_args()

    clf = PhiClassifier()

    if args.check:
        available = clf.is_available()
        info = clf.get_model_info()
        print(f"{'✅' if available else '❌'} Ollama: {'disponible' if available else 'non disponible'}")
        if available:
            print(f"   Modèle : {info.get('model')}")
            print(f"   Taille : {info.get('size_gb', '?')} Go")

    if args.predict:
        print(f"\n🔍 Analyse : {args.predict[:80]}")
        result = clf.predict_one(args.predict)
        label = "🔴 FAKE NEWS" if result["is_fake"] else "🟢 POST FIABLE"
        print(f"\n{label}")
        print(f"  Crédibilité : {result['credibility_score']:.1%}")
        print(f"  Confiance   : {result['confidence']:.1%}")
        print(f"  Raisonnement: {result['reasoning']}")
        if result.get("signals"):
            print(f"  Signaux     : {', '.join(result['signals'])}")
        print(f"  Latence     : {result.get('latency_sec', '?')}s")

    if args.benchmark:
        test_cases = [
            ("URGENT !!! Le gouvernement cache la vérité !! Partagez avant censure !!!", True),
            ("Bien sûr que tout va bien... cherchez la vérité vous-mêmes.", True),
            ("Selon une étude publiée dans Nature, le réchauffement climatique s'accélère.", False),
            ("Breaking: Scientists have been lying for decades! Share before deleted!!!", True),
        ]
        print("\n🧪 BENCHMARK Phi-3 Mini\n" + "="*50)
        correct = 0
        for text, expected_fake in test_cases:
            result = clf.predict_one(text)
            ok = result["is_fake"] == expected_fake
            correct += int(ok)
            icon = "✅" if ok else "❌"
            print(f"{icon} [{result['credibility_score']:.0%}] {text[:60]}...")
            print(f"   → {result['reasoning']}")
        print(f"\nPrécision benchmark : {correct}/{len(test_cases)} ({correct/len(test_cases):.0%})")
