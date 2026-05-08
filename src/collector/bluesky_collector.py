"""
src/collector/bluesky_collector.py
Collecte de posts Bluesky via l'API ATProto.
Supporte la recherche par mots-clés, timeline et utilisateurs.
"""
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from atproto import Client
from langdetect import detect, DetectorFactory
from loguru import logger

sys.path.append(str(Path(__file__).parents[2]))
from config import (
    BLUESKY_HANDLE, BLUESKY_APP_PASSWORD,
    COLLECTION_KEYWORDS, COLLECTION_LIMIT_PER_KEYWORD,
    COLLECTION_INTERVAL_SECONDS, SUPPORTED_LANGUAGES,
    RAW_DIR, LOG_FILE
)

DetectorFactory.seed = 0
logger.add(LOG_FILE, rotation="10 MB", level="INFO")


def _detect_language(text: str) -> str:
    """Détecte la langue d'un texte. Retourne 'unknown' si erreur."""
    try:
        return detect(text)
    except Exception:
        return "unknown"


def _parse_post(post, method_tag: str = "unknown") -> Dict:
    """Transforme un objet post ATProto en dictionnaire structuré."""
    return {
        "id": post.uri,
        "author": post.author.handle,
        "author_name": getattr(post.author, "display_name", "") or "",
        "text": post.record.text,
        "created_at": post.record.created_at,
        "like_count": post.like_count or 0,
        "reply_count": post.reply_count or 0,
        "repost_count": post.repost_count or 0,
        "language": _detect_language(post.record.text),
        "collection_method": method_tag,
        "extracted_at": datetime.now().isoformat(),
    }


class BlueskyCollector:
    """
    Collecteur de posts Bluesky.
    Gère l'authentification, la collecte et la sauvegarde.
    """

    def __init__(self, handle: str = BLUESKY_HANDLE, password: str = BLUESKY_APP_PASSWORD):
        self.client = Client()
        self._handle = handle
        self._password = password
        self._authenticated = False

    def authenticate(self) -> bool:
        """Se connecte à l'API Bluesky."""
        try:
            self.client.login(self._handle, self._password)
            self._authenticated = True
            logger.info(f"Authentification Bluesky OK pour {self._handle}")
            return True
        except Exception as e:
            logger.error(f"Authentification Bluesky échouée: {e}")
            return False

    def _ensure_auth(self):
        if not self._authenticated:
            if not self.authenticate():
                raise RuntimeError("Impossible de s'authentifier à Bluesky.")

    # ----------------------------------------------------------
    # MÉTHODES DE COLLECTE
    # ----------------------------------------------------------

    def collect_by_keywords(
        self,
        keywords: Optional[List[str]] = None,
        limit_per_keyword: int = COLLECTION_LIMIT_PER_KEYWORD,
        languages: Optional[List[str]] = None,
    ) -> List[Dict]:
        """
        Collecte des posts via recherche par mots-clés.
        Filtre automatiquement par langue (FR/EN par défaut).
        """
        self._ensure_auth()
        keywords = keywords or COLLECTION_KEYWORDS
        languages = languages or SUPPORTED_LANGUAGES

        all_posts: List[Dict] = []
        seen_ids: set = set()

        for keyword in keywords:
            try:
                results = self.client.app.bsky.feed.search_posts(
                    {"q": keyword, "limit": limit_per_keyword}
                )
                new = 0
                for post in results.posts:
                    if post.uri in seen_ids:
                        continue
                    parsed = _parse_post(post, method_tag=f"keyword:{keyword}")
                    if parsed["language"] in languages:
                        seen_ids.add(post.uri)
                        all_posts.append(parsed)
                        new += 1

                logger.debug(f"Keyword '{keyword}': {new} posts FR/EN trouvés")
                time.sleep(1)

            except Exception as e:
                logger.warning(f"Erreur keyword '{keyword}': {e}")
                continue

        logger.info(f"collect_by_keywords: {len(all_posts)} posts collectés")
        return all_posts

    def collect_timeline(self, limit: int = 100, languages: Optional[List[str]] = None) -> List[Dict]:
        """Collecte les posts de la timeline personnelle."""
        self._ensure_auth()
        languages = languages or SUPPORTED_LANGUAGES

        posts: List[Dict] = []
        try:
            timeline = self.client.get_timeline(limit=limit)
            for feed_view in timeline.feed:
                parsed = _parse_post(feed_view.post, method_tag="timeline")
                if parsed["language"] in languages:
                    posts.append(parsed)
        except Exception as e:
            logger.error(f"Erreur collecte timeline: {e}")

        logger.info(f"collect_timeline: {len(posts)} posts collectés")
        return posts

    def collect_from_users(
        self,
        usernames: List[str],
        limit_per_user: int = 50,
        languages: Optional[List[str]] = None,
    ) -> List[Dict]:
        """Collecte les posts d'une liste d'utilisateurs spécifiques."""
        self._ensure_auth()
        languages = languages or SUPPORTED_LANGUAGES

        all_posts: List[Dict] = []
        seen_ids: set = set()

        for username in usernames:
            try:
                feed = self.client.get_author_feed(actor=username, limit=limit_per_user)
                for feed_view in feed.feed:
                    post = feed_view.post
                    if post.uri in seen_ids:
                        continue
                    parsed = _parse_post(post, method_tag=f"user:{username}")
                    if parsed["language"] in languages:
                        seen_ids.add(post.uri)
                        all_posts.append(parsed)
                time.sleep(1)
            except Exception as e:
                logger.warning(f"Erreur utilisateur '{username}': {e}")
                continue

        logger.info(f"collect_from_users: {len(all_posts)} posts collectés")
        return all_posts

    def collect_all(self, limit_per_keyword: int = COLLECTION_LIMIT_PER_KEYWORD) -> List[Dict]:
        """
        Collecte combinée : mots-clés + timeline.
        Déduplique automatiquement.
        """
        all_posts: List[Dict] = []
        seen_ids: set = set()

        kw_posts = self.collect_by_keywords(limit_per_keyword=limit_per_keyword)
        tl_posts = self.collect_timeline(limit=100)

        for p in kw_posts + tl_posts:
            if p["id"] not in seen_ids:
                seen_ids.add(p["id"])
                all_posts.append(p)

        logger.info(f"collect_all: {len(all_posts)} posts uniques collectés")
        return all_posts

    # ----------------------------------------------------------
    # SAUVEGARDE
    # ----------------------------------------------------------

    @staticmethod
    def save_to_json(posts: List[Dict], filename: Optional[str] = None) -> Path:
        """Sauvegarde les posts en JSON dans data/raw/."""
        if not filename:
            filename = f"bluesky_posts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = RAW_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(posts, f, ensure_ascii=False, indent=2)
        logger.info(f"Posts sauvegardés: {filepath}")
        return filepath

    # ----------------------------------------------------------
    # MODE AUTOMATIQUE (daemon)
    # ----------------------------------------------------------

    def run_daemon(self, interval: int = COLLECTION_INTERVAL_SECONDS):
        """
        Collecte automatique toutes les `interval` secondes.
        S'intègre avec la DB si disponible.
        Lancez avec : python -m src.collector.bluesky_collector
        """
        from src.database.db_connector import db

        logger.info(f"Démarrage du daemon de collecte (intervalle: {interval}s)")
        print(f"🚀 Daemon Thumalien démarré — collecte toutes les {interval//60} min")
        print("   Ctrl+C pour arrêter\n")

        iteration = 1
        while True:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Itération {iteration}...")
            try:
                posts = self.collect_all()
                if posts:
                    filepath = self.save_to_json(posts)
                    inserted = db.insert_posts(posts)
                    print(f"  ✅ {len(posts)} posts collectés, {inserted} nouveaux en DB")
                else:
                    print("  ⚠️  Aucun post FR/EN trouvé")
            except Exception as e:
                logger.error(f"Erreur daemon itération {iteration}: {e}")
                print(f"  ❌ Erreur: {e}")

            iteration += 1
            print(f"  💤 Prochaine collecte dans {interval//60} min...")
            time.sleep(interval)


# ----------------------------------------------------------
# POINT D'ENTRÉE CLI
# ----------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Collecteur Bluesky Thumalien")
    parser.add_argument("--mode", choices=["once", "daemon"], default="once",
                        help="'once' pour une collecte unique, 'daemon' pour collecte continue")
    parser.add_argument("--limit", type=int, default=25,
                        help="Nombre de posts par mot-clé")
    parser.add_argument("--save-db", action="store_true",
                        help="Sauvegarder en base de données PostgreSQL")
    args = parser.parse_args()

    collector = BlueskyCollector()

    if args.mode == "daemon":
        collector.run_daemon()
    else:
        print("🔍 Collecte unique en cours...")
        posts = collector.collect_all(limit_per_keyword=args.limit)
        filepath = collector.save_to_json(posts)
        print(f"✅ {len(posts)} posts sauvegardés → {filepath}")

        if args.save_db:
            from src.database.db_connector import db
            inserted = db.insert_posts(posts)
            print(f"✅ {inserted} posts insérés en DB")
