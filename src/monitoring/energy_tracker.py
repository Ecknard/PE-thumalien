"""
src/monitoring/energy_tracker.py
Suivi de la consommation énergétique (Green IT) via CodeCarbon.
Wrapper simple pour mesurer chaque opération du pipeline.
"""
import json
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Dict, Optional

from loguru import logger

sys.path.append(str(Path(__file__).parents[2]))
from config import CODECARBON_PROJECT, CODECARBON_OUTPUT_DIR, CODECARBON_COUNTRY, LOG_FILE, MODELS_DIR

logger.add(LOG_FILE, rotation="10 MB", level="INFO")

ENERGY_REPORT_PATH = Path(MODELS_DIR) / "energy_report.json"


def _try_import_codecarbon():
    try:
        from codecarbon import EmissionsTracker
        return EmissionsTracker
    except ImportError:
        return None


class EnergyTracker:
    """
    Tracker énergétique basé sur CodeCarbon.
    Fallback gracieux si CodeCarbon n'est pas installé.
    """

    def __init__(self, project_name: str = CODECARBON_PROJECT):
        self.project_name = project_name
        self._EmissionsTracker = _try_import_codecarbon()
        self._available = self._EmissionsTracker is not None
        self._history: list = []

        if not self._available:
            logger.warning("CodeCarbon non disponible. Tracking énergétique désactivé.")
            print("⚠️  CodeCarbon non installé. Pip: pip install codecarbon")

    @contextmanager
    def track(self, operation_name: str, n_samples: int = 0, model_name: str = ""):
        """
        Context manager pour mesurer l'empreinte énergétique d'une opération.

        Usage:
            with tracker.track("fine_tuning_bert", n_samples=500, model_name="distilbert"):
                model.train(...)
        """
        start_time = time.time()

        if self._available:
            tracker = self._EmissionsTracker(
                project_name=self.project_name,
                output_dir=CODECARBON_OUTPUT_DIR,
                log_level="error",
                save_to_file=True,
            )
            tracker.start()
            try:
                yield
            finally:
                emissions = tracker.stop()
                duration = time.time() - start_time

                # CodeCarbon retourne les émissions en kg CO2
                energy_kwh = getattr(tracker, "_total_energy", None)
                if hasattr(energy_kwh, "kWh"):
                    energy_kwh = float(energy_kwh.kWh)
                else:
                    energy_kwh = float(emissions or 0) * 2.5  # Estimation

                record = {
                    "operation": operation_name,
                    "emissions_kg": round(float(emissions or 0), 8),
                    "energy_kwh": round(energy_kwh, 8),
                    "duration_sec": round(duration, 2),
                    "model_name": model_name,
                    "n_samples": n_samples,
                    "timestamp": datetime.now().isoformat(),
                }
                self._history.append(record)
                self._log_and_save(record)

                print(f"\n⚡ Monitoring énergétique [{operation_name}]:")
                print(f"   Émissions CO₂ : {record['emissions_kg']*1000:.4f} g CO₂eq")
                print(f"   Énergie       : {record['energy_kwh']*1000:.4f} Wh")
                print(f"   Durée         : {record['duration_sec']:.1f}s")
        else:
            # Fallback sans CodeCarbon
            yield
            duration = time.time() - start_time
            record = {
                "operation": operation_name,
                "emissions_kg": None,
                "energy_kwh": None,
                "duration_sec": round(duration, 2),
                "model_name": model_name,
                "n_samples": n_samples,
                "timestamp": datetime.now().isoformat(),
                "note": "CodeCarbon non disponible — durée seulement",
            }
            self._history.append(record)
            self._log_and_save(record)
            print(f"\n⏱️  Opération [{operation_name}] terminée en {duration:.1f}s (no energy tracking)")

    def _log_and_save(self, record: Dict):
        """Persiste l'historique dans un fichier JSON + DB si disponible."""
        logger.info(f"Energy: {record}")

        # Sauvegarde JSON locale
        ENERGY_REPORT_PATH.parent.mkdir(exist_ok=True)
        history = self._load_history_from_file()
        history.append(record)
        with open(ENERGY_REPORT_PATH, "w") as f:
            json.dump(history, f, indent=2)

        # Tentative d'insertion en DB
        try:
            from src.database.db_connector import db
            if record.get("emissions_kg") is not None:
                db.log_energy(
                    operation=record["operation"],
                    emissions_kg=record["emissions_kg"],
                    energy_kwh=record["energy_kwh"],
                    duration_sec=record["duration_sec"],
                    model_name=record["model_name"],
                    n_samples=record["n_samples"],
                )
        except Exception:
            pass  # DB optionnelle

    def _load_history_from_file(self) -> list:
        if ENERGY_REPORT_PATH.exists():
            with open(ENERGY_REPORT_PATH) as f:
                return json.load(f)
        return []

    def get_report(self) -> Dict:
        """Rapport énergétique cumulé depuis le fichier JSON."""
        history = self._load_history_from_file()

        valid = [r for r in history if r.get("emissions_kg") is not None]
        total_emissions = sum(r["emissions_kg"] for r in valid)
        total_kwh = sum(r["energy_kwh"] for r in valid)
        total_duration = sum(r["duration_sec"] for r in history)

        return {
            "n_operations": len(history),
            "n_measured": len(valid),
            "total_emissions_kg": round(total_emissions, 8),
            "total_emissions_g": round(total_emissions * 1000, 5),
            "total_kwh": round(total_kwh, 8),
            "total_wh": round(total_kwh * 1000, 5),
            "total_hours": round(total_duration / 3600, 3),
            "co2_eq_km_voiture": round(total_emissions / 0.12, 4),  # 120g CO2/km
            "operations": [r["operation"] for r in history],
            "codecarbon_available": self._available,
        }

    def print_report(self):
        """Affiche un rapport formaté dans la console."""
        r = self.get_report()
        print("\n" + "=" * 50)
        print("⚡ RAPPORT ÉNERGÉTIQUE — THUMALIEN")
        print("=" * 50)
        print(f"  Opérations mesurées : {r['n_measured']}/{r['n_operations']}")
        print(f"  CO₂ total          : {r['total_emissions_g']:.4f} g CO₂eq")
        print(f"  Énergie totale     : {r['total_wh']:.4f} Wh")
        print(f"  Temps total        : {r['total_hours']:.2f}h")
        print(f"  Équivalent voiture : {r['co2_eq_km_voiture']:.6f} km")
        print("=" * 50)

    def tracked_function(self, operation_name: str, model_name: str = ""):
        """Décorateur pour tracker automatiquement une fonction."""
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.track(operation_name, model_name=model_name):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


# Instance globale réutilisable
tracker = EnergyTracker()


if __name__ == "__main__":
    print("🔋 Test du tracker énergétique...")

    with tracker.track("test_operation", n_samples=100, model_name="test"):
        # Simule une opération
        time.sleep(2)
        _ = [i**2 for i in range(100_000)]

    tracker.print_report()
