"""
seed_api_key.py — One-time script to create an API key and seed it into MySQL.

Usage:
    uv run python scripts/seed_api_key.py --name "dev-key" --rpm 60

The raw key is printed ONCE to stdout and never stored.
Only the SHA-256 hash is written to MySQL.

Run from the project root with the .env file present.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import secrets
import sys
from pathlib import Path

# Add project root to path so 'app' is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


def generate_api_key() -> str:
    """Generate a cryptographically secure 32-byte hex API key (64 chars)."""
    return secrets.token_hex(32)


def hash_key(raw_key: str) -> str:
    """Return SHA-256 hex digest of raw_key."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


async def seed_key(name: str, rpm: int) -> None:
    """Insert a new API key into MySQL and print the raw key."""
    from app.db.mysql import init_mysql, get_db_session
    from app.db.models import ApiKey

    await init_mysql()

    raw_key = generate_api_key()
    key_hash = hash_key(raw_key)

    async with get_db_session() as session:
        existing = await session.get(ApiKey, None)  # just to test session
        new_key = ApiKey(
            name=name,
            key_hash=key_hash,
            rate_limit_rpm=rpm,
            is_active=True,
        )
        session.add(new_key)

    print("\n" + "=" * 60)
    print("API KEY CREATED — save this, it will NOT be shown again")
    print("=" * 60)
    print(f"Name       : {name}")
    print(f"Raw Key    : {raw_key}")
    print(f"Key Hash   : {key_hash}")
    print(f"Rate Limit : {rpm} RPM")
    print("=" * 60)
    print("\nUse in requests:")
    print(f'  curl -H "X-API-Key: {raw_key}" http://localhost:8000/api/v1/health\n')


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed an API key into InsightDocket MySQL")
    parser.add_argument("--name", required=True, help="Human-readable label for the key")
    parser.add_argument("--rpm", type=int, default=60, help="Rate limit in requests per minute")
    args = parser.parse_args()

    asyncio.run(seed_key(name=args.name, rpm=args.rpm))


if __name__ == "__main__":
    main()
