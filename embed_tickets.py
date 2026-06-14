#!/usr/bin/env python
"""
Backfill vector embeddings for tickets in local Postgres.

Requires:
- pgvector extension installed
- EMBEDDING_MODEL set in .env (default: all-MiniLM-L6-v2)
- DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS set in .env
"""

import os
from typing import List

from dotenv import load_dotenv

load_dotenv()


def _vector_to_pgvector(values: List[float]) -> str:
    return "[" + ",".join(f"{v:.6f}" for v in values) + "]"


def main() -> None:
    from sentence_transformers import SentenceTransformer
    import psycopg

    db_host = os.getenv("DB_HOST", "127.0.0.1")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_NAME", "ticketsdb")
    db_user = os.getenv("DB_USER", "postgres")
    db_pass = os.getenv("DB_PASS", "")
    model_name = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    if not db_pass:
        raise SystemExit("DB_PASS is required in .env")

    print(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)

    conn = psycopg.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_pass,
    )

    with conn:
        with conn.cursor() as cur:
            cur.execute("SELECT ticket_id, title, description FROM tickets WHERE embedding IS NULL")
            rows = cur.fetchall()

            print(f"Found {len(rows)} ticket(s) without embeddings")
            for ticket_id, title, description in rows:
                text = f"{title}\n{description or ''}".strip()
                embedding = model.encode(text, normalize_embeddings=True)
                vector = _vector_to_pgvector(embedding.tolist())
                cur.execute(
                    "UPDATE tickets SET embedding = %s::vector WHERE ticket_id = %s",
                    (vector, ticket_id),
                )
                print(f"Updated ticket {ticket_id}")

    conn.close()
    print("Done")


if __name__ == "__main__":
    main()
