from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(r"C:\MausamSaathiSIH26\MAUSAMSAATHI_DEMO")
ENV_CANDIDATES = [ROOT / ".env", ROOT / "backend" / ".env"]
REQUEST_TIMEOUT = 20

def _dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in ENV_CANDIDATES:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"').strip("'")
    return values

def env(name: str, default: str = "") -> str:
    value = os.getenv(name, "").strip()
    if value:
        return value
    return _dotenv().get(name, default).strip()

def extract_bearer_token(authorization: str | None) -> str:
    if not authorization:
        raise ValueError("Missing Authorization header.")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise ValueError("Authorization header must be Bearer <access_token>.")
    return token.strip()

def verify_supabase_access_token(access_token: str) -> dict[str, Any]:
    base = env("SUPABASE_URL").rstrip("/")
    api_key = env("SUPABASE_PUBLISHABLE_KEY") or env("SUPABASE_SECRET_KEY")
    if not base or not api_key:
        raise RuntimeError("Supabase URL/key is not configured.")

    req = urllib.request.Request(
        f"{base}/auth/v1/user",
        headers={
            "apikey": api_key,
            "Authorization": f"Bearer {access_token}",
            "Accept": "application/json",
            "User-Agent": "MAUSAMSAATHI-Auth/1.0",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            if response.status != 200:
                raise RuntimeError(f"Supabase Auth returned HTTP {response.status}")
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        if exc.code in {401, 403}:
            raise ValueError("Invalid or expired Supabase access token.")
        raise RuntimeError(f"Supabase Auth verification failed HTTP {exc.code}: {detail}")
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach Supabase Auth: {exc}")

    if not isinstance(payload, dict) or not payload.get("id"):
        raise ValueError("Supabase Auth returned no valid user id.")
    return payload
