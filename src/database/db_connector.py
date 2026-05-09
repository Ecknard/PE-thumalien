"""
src/database/db_connector.py
Gestionnaire de connexion PostgreSQL avec context manager,
initialisation du schéma et méthodes CRUD.
"""
import sys
import os
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from typing import Optional, List, Dict, Any

import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from loguru import logger

sys.path.append(str(Path(__file__).parents[2]))
from config import DB_CONFIG, LOG_FILE

logger.add(LOG_FILE, rotation="10 MB", level="INFO")

# ============================================================
# SCHÉMA SQL
# ============================================================

# Migration : supprime les anciennes tables si elles ont un schéma incompatible
# (ex : id INTEGER au lieu de TEXT depuis l'ancienne version du projet)
MIGRATE_SQL = """
DO $$
BEGIN
    -- Vérifier si la colonne id de posts est bien TEXT
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'posts' AND column_name = 'id'
        AND data_type != 'text'
    ) THEN
        DROP TABLE IF EXISTS energy_tracking CASCADE;
        DROP TABLE IF EXISTS emotion_analyses CASCADE;
        DROP TABLE IF EXISTS predictions CASCADE;
        DROP TABLE IF EXISTS posts CASCADE;
        RAISE NOTICE 'Anciennes tables supprimées (schéma incompatible — migration vers TEXT id)';
    END IF;

    -- Vérifier si la colonne author existe (ancienne table sans elle)
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'posts')
    AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'posts' AND column_name = 'author'
    ) THEN
        DROP TABLE IF EXISTS energy_tracking CASCADE;
        DROP TABLE IF EXISTS emotion_analyses CASCADE;
        DROP TABLE IF EXISTS predictions CASCADE;
        DROP TABLE IF EXISTS posts CASCADE;
        RAISE NOTICE 'Anciennes tables supprimées (colonnes manquantes — migration)';
    END IF;
END$$;
"""

SCHEMA_SQL = """
-- Table principale des posts collectés
CREATE TABLE IF NOT EXISTS posts (
    id              TEXT PRIMARY KEY,
    author          TEXT NOT NULL,
    author_name     TEXT,
    text_original   TEXT NOT NULL,
    text_clean      TEXT,
    language        TEXT,
    like_count      INTEGER DEFAULT 0,
    reply_count     INTEGER DEFAULT 0,
    repost_count    INTEGER DEFAULT 0,
    created_at      TEXT,
    extracted_at    TIMESTAMPTZ DEFAULT NOW(),
    processed_at    TIMESTAMPTZ,
    is_analyzed     BOOLEAN DEFAULT FALSE
);

-- Table des prédictions fake news
CREATE TABLE IF NOT EXISTS predictions (
    id                  SERIAL PRIMARY KEY,
    post_id             TEXT REFERENCES posts(id) ON DELETE CASCADE,
    credibility_score   FLOAT NOT NULL,
    is_fake             BOOLEAN NOT NULL,
    confidence          FLOAT NOT NULL,
    model_version       TEXT DEFAULT 'v1',
    predicted_at        TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(post_id)
);

-- Table des analyses émotionnelles
CREATE TABLE IF NOT EXISTS emotion_analyses (
    id              SERIAL PRIMARY KEY,
    post_id         TEXT REFERENCES posts(id) ON DELETE CASCADE,
    emotion_label   TEXT NOT NULL,
    emotion_score   FLOAT NOT NULL,
    vader_compound  FLOAT,
    analyzed_at     TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(post_id)
);

-- Table du monitoring énergétique
CREATE TABLE IF NOT EXISTS energy_tracking (
    id              SERIAL PRIMARY KEY,
    operation       TEXT NOT NULL,
    emissions_kg    FLOAT,
    energy_kwh      FLOAT,
    duration_sec    FLOAT,
    model_name      TEXT,
    n_samples       INTEGER,
    tracked_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Index pour les performances
CREATE INDEX IF NOT EXISTS idx_posts_language ON posts(language);
CREATE INDEX IF NOT EXISTS idx_posts_is_analyzed ON posts(is_analyzed);
CREATE INDEX IF NOT EXISTS idx_predictions_is_fake ON predictions(is_fake);
CREATE INDEX IF NOT EXISTS idx_predictions_credibility ON predictions(credibility_score);
"""


class DatabaseConnector:
    """Gestionnaire de connexion PostgreSQL robuste."""

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or DB_CONFIG

    def get_connection(self) -> psycopg2.extensions.connection:
        try:
            conn = psycopg2.connect(**self.config)
            return conn
        except psycopg2.OperationalError as e:
            logger.error(f"Connexion DB impossible: {e}")
            raise

    @contextmanager
    def get_cursor(self, commit: bool = True, dict_cursor: bool = False):
        """Context manager sécurisé pour les curseurs."""
        conn = self.get_connection()
        cursor_factory = RealDictCursor if dict_cursor else None
        cursor = conn.cursor(cursor_factory=cursor_factory) if cursor_factory else conn.cursor()
        try:
            yield cursor
            if commit:
                conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Erreur SQL: {e}")
            raise
        finally:
            cursor.close()
            conn.close()

    def initialize_schema(self) -> bool:
        """Crée toutes les tables si elles n'existent pas.
        Exécute d'abord la migration pour supprimer les tables
        incompatibles issues d'une ancienne version du projet."""
        try:
            # Étape 1 : migration (drop si schéma incompatible)
            with self.get_cursor() as cur:
                cur.execute(MIGRATE_SQL)
            # Étape 2 : création des tables
            with self.get_cursor() as cur:
                cur.execute(SCHEMA_SQL)
            logger.info("Schéma initialisé avec succès.")
            return True
        except Exception as e:
            logger.error(f"Erreur init schéma: {e}")
            return False

    def test_connection(self) -> bool:
        """Teste la connexion et affiche la version PostgreSQL."""
        try:
            with self.get_cursor() as cur:
                cur.execute("SELECT version();")
                version = cur.fetchone()[0]
            logger.info(f"Connexion OK — {version}")
            print(f"✅ Connexion PostgreSQL OK\n   {version}")
            return True
        except Exception as e:
            logger.error(f"Echec connexion: {e}")
            print(f"❌ Connexion échouée: {e}")
            return False

    # ----------------------------------------------------------
    # POSTS
    # ----------------------------------------------------------

    def insert_posts(self, posts: List[Dict]) -> int:
        """
        Insère une liste de posts (ignore les doublons via ON CONFLICT).
        Retourne le nombre de posts réellement insérés.
        """
        if not posts:
            return 0

        query = """
            INSERT INTO posts
                (id, author, author_name, text_original, text_clean,
                 language, like_count, reply_count, repost_count, created_at, extracted_at)
            VALUES %s
            ON CONFLICT (id) DO NOTHING
        """
        values = [
            (
                p.get("id"), p.get("author"), p.get("author_name", ""),
                p.get("text", p.get("text_original", "")),
                p.get("text_clean"),
                p.get("language"), p.get("like_count", 0),
                p.get("reply_count", 0), p.get("repost_count", 0),
                p.get("created_at"),
                datetime.fromisoformat(p["extracted_at"]) if p.get("extracted_at") else datetime.now(),
            )
            for p in posts
        ]

        with self.get_cursor() as cur:
            execute_values(cur, query, values)
            inserted = cur.rowcount

        logger.info(f"{inserted} posts insérés en DB.")
        return inserted

    def get_posts(
        self,
        language: Optional[str] = None,
        only_unanalyzed: bool = False,
        limit: int = 100,
    ) -> List[Dict]:
        """Récupère des posts avec filtres optionnels."""
        conditions = []
        params = []

        if language:
            conditions.append("language = %s")
            params.append(language)
        if only_unanalyzed:
            conditions.append("is_analyzed = FALSE")

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        query = f"""
            SELECT * FROM posts
            {where}
            ORDER BY extracted_at DESC
            LIMIT %s
        """
        params.append(limit)

        with self.get_cursor(dict_cursor=True) as cur:
            cur.execute(query, params)
            return [dict(row) for row in cur.fetchall()]

    def count_posts(self) -> Dict[str, int]:
        """Retourne les statistiques de la table posts."""
        with self.get_cursor() as cur:
            cur.execute("SELECT COUNT(*) FROM posts")
            total = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM posts WHERE language='fr'")
            fr = cur.fetchone()[0]
            cur.execute("SELECT COUNT(*) FROM posts WHERE language='en'")
            en = cur.fetchone()[0]
        return {"total": total, "fr": fr, "en": en}

    # ----------------------------------------------------------
    # PRÉDICTIONS
    # ----------------------------------------------------------

    def upsert_prediction(self, post_id: str, credibility_score: float,
                          is_fake: bool, confidence: float,
                          model_version: str = "v1") -> None:
        """Insère ou met à jour une prédiction."""
        query = """
            INSERT INTO predictions (post_id, credibility_score, is_fake, confidence, model_version)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (post_id) DO UPDATE SET
                credibility_score = EXCLUDED.credibility_score,
                is_fake = EXCLUDED.is_fake,
                confidence = EXCLUDED.confidence,
                model_version = EXCLUDED.model_version,
                predicted_at = NOW()
        """
        with self.get_cursor() as cur:
            cur.execute(query, (post_id, credibility_score, is_fake, confidence, model_version))

        # Marquer le post comme analysé
        with self.get_cursor() as cur:
            cur.execute(
                "UPDATE posts SET is_analyzed=TRUE, processed_at=NOW() WHERE id=%s",
                (post_id,)
            )

    def get_predictions_stats(self) -> Dict:
        """Statistiques globales des prédictions."""
        with self.get_cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN is_fake THEN 1 ELSE 0 END) as fake_count,
                    AVG(credibility_score) as avg_credibility,
                    AVG(confidence) as avg_confidence
                FROM predictions
            """)
            row = cur.fetchone()
        return {
            "total": row[0] or 0,
            "fake_count": row[1] or 0,
            "real_count": (row[0] or 0) - (row[1] or 0),
            "avg_credibility": round(row[2] or 0, 3),
            "avg_confidence": round(row[3] or 0, 3),
        }

    def get_posts_with_predictions(self, limit: int = 200) -> List[Dict]:
        """Retourne les posts enrichis de leurs prédictions et émotions."""
        query = """
            SELECT
                p.id, p.author, p.author_name, p.text_original, p.text_clean,
                p.language, p.like_count, p.reply_count, p.repost_count,
                p.created_at, p.extracted_at,
                pr.credibility_score, pr.is_fake, pr.confidence, pr.model_version,
                ea.emotion_label, ea.emotion_score, ea.vader_compound
            FROM posts p
            LEFT JOIN predictions pr ON p.id = pr.post_id
            LEFT JOIN emotion_analyses ea ON p.id = ea.post_id
            WHERE p.is_analyzed = TRUE
            ORDER BY p.extracted_at DESC
            LIMIT %s
        """
        with self.get_cursor(dict_cursor=True) as cur:
            cur.execute(query, (limit,))
            return [dict(row) for row in cur.fetchall()]

    # ----------------------------------------------------------
    # ÉMOTIONS
    # ----------------------------------------------------------

    def upsert_emotion(self, post_id: str, emotion_label: str,
                       emotion_score: float, vader_compound: float) -> None:
        """Insère ou met à jour une analyse émotionnelle."""
        query = """
            INSERT INTO emotion_analyses (post_id, emotion_label, emotion_score, vader_compound)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (post_id) DO UPDATE SET
                emotion_label = EXCLUDED.emotion_label,
                emotion_score = EXCLUDED.emotion_score,
                vader_compound = EXCLUDED.vader_compound,
                analyzed_at = NOW()
        """
        with self.get_cursor() as cur:
            cur.execute(query, (post_id, emotion_label, emotion_score, vader_compound))

    def get_emotion_distribution(self) -> List[Dict]:
        """Distribution des émotions sur tous les posts analysés."""
        with self.get_cursor(dict_cursor=True) as cur:
            cur.execute("""
                SELECT emotion_label, COUNT(*) as count, AVG(emotion_score) as avg_score
                FROM emotion_analyses
                GROUP BY emotion_label
                ORDER BY count DESC
            """)
            return [dict(row) for row in cur.fetchall()]

    # ----------------------------------------------------------
    # MONITORING ÉNERGÉTIQUE
    # ----------------------------------------------------------

    def log_energy(self, operation: str, emissions_kg: float,
                   energy_kwh: float, duration_sec: float,
                   model_name: str = "", n_samples: int = 0) -> None:
        """Enregistre une mesure énergétique."""
        with self.get_cursor() as cur:
            cur.execute(
                """
                INSERT INTO energy_tracking
                    (operation, emissions_kg, energy_kwh, duration_sec, model_name, n_samples)
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (operation, emissions_kg, energy_kwh, duration_sec, model_name, n_samples)
            )

    def get_energy_report(self) -> Dict:
        """Rapport énergétique cumulé."""
        with self.get_cursor() as cur:
            cur.execute("""
                SELECT
                    SUM(emissions_kg) as total_emissions_kg,
                    SUM(energy_kwh) as total_kwh,
                    SUM(duration_sec) as total_seconds,
                    COUNT(*) as n_operations
                FROM energy_tracking
            """)
            row = cur.fetchone()
        return {
            "total_emissions_kg": round(row[0] or 0, 6),
            "total_kwh": round(row[1] or 0, 6),
            "total_hours": round((row[2] or 0) / 3600, 2),
            "n_operations": row[3] or 0,
            "co2_equivalent_km": round((row[0] or 0) * 5.5, 3),  # ~180g CO2/km
        }


# Instance globale
db = DatabaseConnector()

if __name__ == "__main__":
    print("🔍 Test de connexion et initialisation du schéma...")
    if db.test_connection():
        db.initialize_schema()
        print("✅ Schéma créé.")
        stats = db.count_posts()
        print(f"📊 Posts en DB: {stats}")
