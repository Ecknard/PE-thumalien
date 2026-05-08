-- docker/init.sql
-- Exécuté automatiquement au premier démarrage du conteneur PostgreSQL
-- Crée les extensions utiles

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- Pour la recherche full-text rapide

-- Index GIN pour la recherche dans les textes
-- (sera créé après la création des tables par db_connector.py)
