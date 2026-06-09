"""
Build the full index from the command line (outside Docker).

Usage:
    python scripts/build_index.py [--datasets pubmedqa medqa radqa]

Requires the backend package to be importable and a running PostgreSQL instance.
Set environment variables as per .env.example before running.
"""

import argparse
import sys
import os

# Allow running from project root or scripts/ dir
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from database.connection import init_db
from indexing.indexer import run_indexing


def main():
    parser = argparse.ArgumentParser(description="Build Fuzzy RAG index")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["pubmedqa", "medqa", "radqa"],
        default=["pubmedqa", "medqa", "radqa"],
        help="Datasets to index (default: all three)",
    )
    args = parser.parse_args()

    print(f"Initialising database schema...")
    init_db()

    print(f"Indexing datasets: {args.datasets}")
    run_indexing(datasets=args.datasets)

    print("Indexing complete.")


if __name__ == "__main__":
    main()
