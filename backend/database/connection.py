import psycopg2
from psycopg2.extras import RealDictCursor
from config import config


def get_connection():
    return psycopg2.connect(config.dsn)


def init_db():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS documents (
                    id          BIGSERIAL PRIMARY KEY,
                    text        TEXT        NOT NULL,
                    source      TEXT        NOT NULL,
                    question    TEXT,
                    answer      TEXT,
                    metadata    JSONB       NOT NULL DEFAULT '{{}}',
                    embedding   VECTOR({config.VECTOR_DIM}),
                    ts_vector   TSVECTOR GENERATED ALWAYS AS (
                                    to_tsvector('english', coalesce(text, ''))
                                ) STORED
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_embedding_idx
                ON documents USING hnsw (embedding vector_cosine_ops);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS documents_ts_idx
                ON documents USING gin (ts_vector);
                """
            )
            conn.commit()
