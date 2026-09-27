from __future__ import annotations

import json
import os
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(r"C:\MausamSaathiSIH26\MAUSAMSAATHI_DEMO")
DB_PATH = ROOT / "data_test" / "processed" / "persistence" / "mausamsaathi_local.sqlite3"
ENV_CANDIDATES = [ROOT / ".env", ROOT / "backend" / ".env"]
BUCKET = "mausamsaathi-scientific"
REQUEST_TIMEOUT = 30


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _read_dotenv() -> dict[str, str]:
    values: dict[str, str] = {}
    for path in ENV_CANDIDATES:
        if not path.exists():
            continue
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and value:
                values[key] = value
    return values


def env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    if value:
        return value.strip()
    return _read_dotenv().get(name, default).strip()


def cloud_configured() -> bool:
    return bool(env("SUPABASE_URL") and env("SUPABASE_SECRET_KEY"))


def _conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def initialize() -> None:
    # Importing the existing schema is intentionally side-effect-light.
    from persistence import SCHEMA_SQL

    with _conn() as conn:
        conn.executescript(SCHEMA_SQL)
        stamp = now_iso()
        conn.execute(
            """
            INSERT INTO app_meta(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            ("application_persistence", "step17", stamp),
        )


def _cloud_request(
    path: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str = "application/json",
) -> tuple[int, bytes]:
    base = env("SUPABASE_URL").rstrip("/")
    key = env("SUPABASE_SECRET_KEY")

    if not base or not key:
        raise RuntimeError("Cloud persistence is not configured.")

    url = f"{base}{path}"
    headers = {
        "apikey": key,
        "User-Agent": "MAUSAMSAATHI-Persistence/1.0",
    }
    if key.startswith("sb_secret_"):
        pass
    elif key.count(".") == 2:
        headers["Authorization"] = f"Bearer {key}"
    else:
        raise RuntimeError("Unsupported Supabase key format.")

    if content_type:
        headers["Content-Type"] = content_type

    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read()


def cloud_upsert(table: str, rows: list[dict[str, Any]], conflict: str) -> None:
    if not rows:
        return
    params = urllib.parse.urlencode({"on_conflict": conflict})
    body = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    status, payload = _cloud_request(
        f"/rest/v1/{table}?{params}",
        method="POST",
        body=body,
    )
    if status not in {200, 201, 204}:
        raise RuntimeError(
            f"Cloud upsert {table} failed HTTP {status}: "
            f"{payload.decode('utf-8', errors='replace')[:800]}"
        )


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


def create_user(
    *,
    name: str | None = None,
    phone_e164: str | None = None,
    preferred_language: str = "hi",
    external_auth_id: str | None = None,
) -> dict[str, Any]:
    # When Supabase Auth is available, use the Auth UUID as the canonical user id.
    user_id = str(external_auth_id or uuid.uuid4())
    stamp = now_iso()
    initialize()

    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO users(
                user_id, external_auth_id, name, phone_e164,
                preferred_language, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(user_id) DO UPDATE SET
                external_auth_id=excluded.external_auth_id,
                name=COALESCE(excluded.name, users.name),
                phone_e164=COALESCE(excluded.phone_e164, users.phone_e164),
                preferred_language=excluded.preferred_language,
                updated_at=excluded.updated_at
            """,
            (
                user_id,
                external_auth_id,
                name,
                phone_e164,
                preferred_language,
                stamp,
                stamp,
            ),
        )
        row = conn.execute(
            "SELECT * FROM users WHERE user_id=?",
            (user_id,),
        ).fetchone()

    return _row_dict(row) or {}


def register_device(
    *,
    user_id: str,
    platform: str | None = None,
    app_version: str | None = None,
    push_token: str | None = None,
    device_id: str | None = None,
) -> dict[str, Any]:
    initialize()
    device_id = str(device_id or uuid.uuid4())
    stamp = now_iso()

    with _conn() as conn:
        exists = conn.execute(
            "SELECT 1 FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        if not exists:
            raise ValueError("User does not exist locally.")

        conn.execute(
            """
            INSERT INTO devices(
                device_id, user_id, platform, app_version,
                push_token, last_seen_at, is_active, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)
            ON CONFLICT(device_id) DO UPDATE SET
                user_id=excluded.user_id,
                platform=excluded.platform,
                app_version=excluded.app_version,
                push_token=excluded.push_token,
                last_seen_at=excluded.last_seen_at,
                is_active=1,
                updated_at=excluded.updated_at
            """,
            (
                device_id,
                user_id,
                platform,
                app_version,
                push_token,
                stamp,
                stamp,
                stamp,
            ),
        )
        row = conn.execute(
            "SELECT * FROM devices WHERE device_id=?",
            (device_id,),
        ).fetchone()

    return _row_dict(row) or {}


def set_user_location(
    *,
    user_id: str,
    local_body_code: str,
    is_primary: bool = True,
) -> dict[str, Any]:
    initialize()
    stamp = now_iso()

    with _conn() as conn:
        if not conn.execute(
            "SELECT 1 FROM users WHERE user_id=?", (user_id,)
        ).fetchone():
            raise ValueError("User does not exist locally.")

        if not conn.execute(
            "SELECT 1 FROM locations WHERE local_body_code=?",
            (str(local_body_code),),
        ).fetchone():
            raise ValueError(f"Unknown local_body_code: {local_body_code}")

        if is_primary:
            conn.execute(
                "UPDATE user_locations SET is_primary=0, updated_at=? WHERE user_id=?",
                (stamp, user_id),
            )

        conn.execute(
            """
            INSERT INTO user_locations(
                user_id, local_body_code, is_primary, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(user_id, local_body_code) DO UPDATE SET
                is_primary=excluded.is_primary,
                updated_at=excluded.updated_at
            """,
            (user_id, str(local_body_code), int(is_primary), stamp, stamp),
        )
        row = conn.execute(
            """
            SELECT ul.*, l.state, l.district, l.block, l.panchayat
            FROM user_locations ul
            JOIN locations l ON l.local_body_code=ul.local_body_code
            WHERE ul.user_id=? AND ul.local_body_code=?
            """,
            (user_id, str(local_body_code)),
        ).fetchone()

    return _row_dict(row) or {}


def upsert_crop_profile(
    *,
    user_id: str,
    local_body_code: str,
    crop_name: str,
    growth_stage: str | None = None,
    area_hectare: float | None = None,
    notes: str | None = None,
    crop_profile_id: str | None = None,
) -> dict[str, Any]:
    initialize()
    crop_profile_id = str(crop_profile_id or uuid.uuid4())
    stamp = now_iso()

    with _conn() as conn:
        if not conn.execute(
            "SELECT 1 FROM users WHERE user_id=?", (user_id,)
        ).fetchone():
            raise ValueError("User does not exist locally.")
        if not conn.execute(
            "SELECT 1 FROM locations WHERE local_body_code=?",
            (str(local_body_code),),
        ).fetchone():
            raise ValueError(f"Unknown local_body_code: {local_body_code}")

        conn.execute(
            """
            INSERT INTO crop_profiles(
                crop_profile_id, user_id, local_body_code, crop_name,
                growth_stage, area_hectare, notes, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(crop_profile_id) DO UPDATE SET
                local_body_code=excluded.local_body_code,
                crop_name=excluded.crop_name,
                growth_stage=excluded.growth_stage,
                area_hectare=excluded.area_hectare,
                notes=excluded.notes,
                updated_at=excluded.updated_at
            """,
            (
                crop_profile_id,
                user_id,
                str(local_body_code),
                crop_name,
                growth_stage,
                area_hectare,
                notes,
                stamp,
                stamp,
            ),
        )
        row = conn.execute(
            "SELECT * FROM crop_profiles WHERE crop_profile_id=?",
            (crop_profile_id,),
        ).fetchone()

    return _row_dict(row) or {}


def queue_notification(
    *,
    channel: str,
    message: str,
    user_id: str | None = None,
    local_body_code: str | None = None,
    advisory_id: str | None = None,
    language: str = "hi",
    scheduled_at: str | None = None,
) -> dict[str, Any]:
    initialize()
    notification_id = str(uuid.uuid4())
    stamp = now_iso()

    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO notifications(
                notification_id, user_id, local_body_code, advisory_id,
                channel, language, message, status, scheduled_at,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?, ?)
            """,
            (
                notification_id,
                user_id,
                local_body_code,
                advisory_id,
                channel,
                language,
                message,
                scheduled_at,
                stamp,
                stamp,
            ),
        )
        row = conn.execute(
            "SELECT * FROM notifications WHERE notification_id=?",
            (notification_id,),
        ).fetchone()

        conn.execute(
            """
            INSERT INTO sync_queue(
                sync_id, entity_type, entity_id, operation,
                payload_json, status, attempts, created_at, updated_at
            )
            VALUES (?, 'notification', ?, 'upsert', ?, 'pending', 0, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                notification_id,
                json.dumps(dict(row), ensure_ascii=False),
                stamp,
                stamp,
            ),
        )

    return _row_dict(row) or {}


def create_sos(
    *,
    user_id: str | None = None,
    device_id: str | None = None,
    local_body_code: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    initialize()
    sos_id = str(uuid.uuid4())
    stamp = now_iso()

    with _conn() as conn:
        conn.execute(
            """
            INSERT INTO sos_events(
                sos_event_id, user_id, device_id, local_body_code,
                latitude, longitude, status, triggered_at, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, 'created', ?, ?)
            """,
            (
                sos_id,
                user_id,
                device_id,
                local_body_code,
                latitude,
                longitude,
                stamp,
                json.dumps(payload or {}, ensure_ascii=False),
            ),
        )
        row = conn.execute(
            "SELECT * FROM sos_events WHERE sos_event_id=?",
            (sos_id,),
        ).fetchone()

        conn.execute(
            """
            INSERT INTO sync_queue(
                sync_id, entity_type, entity_id, operation,
                payload_json, status, attempts, created_at, updated_at
            )
            VALUES (?, 'sos_event', ?, 'upsert', ?, 'pending', 0, ?, ?)
            """,
            (
                str(uuid.uuid4()),
                sos_id,
                json.dumps(dict(row), ensure_ascii=False),
                stamp,
                stamp,
            ),
        )

    return _row_dict(row) or {}


def user_context(user_id: str) -> dict[str, Any]:
    initialize()
    with _conn() as conn:
        user = conn.execute(
            "SELECT * FROM users WHERE user_id=?", (user_id,)
        ).fetchone()
        if user is None:
            raise ValueError("User not found.")

        devices = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM devices WHERE user_id=? ORDER BY created_at DESC",
                (user_id,),
            ).fetchall()
        ]
        locations = [
            dict(row)
            for row in conn.execute(
                """
                SELECT ul.*, l.state, l.district, l.block, l.panchayat
                FROM user_locations ul
                JOIN locations l ON l.local_body_code=ul.local_body_code
                WHERE ul.user_id=?
                ORDER BY ul.is_primary DESC, ul.updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        ]
        crops = [
            dict(row)
            for row in conn.execute(
                "SELECT * FROM crop_profiles WHERE user_id=? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        ]

    return {
        "user": dict(user),
        "devices": devices,
        "locations": locations,
        "crop_profiles": crops,
    }


def sync_user_to_cloud(user_id: str) -> dict[str, Any]:
    initialize()
    ctx = user_context(user_id)
    user = ctx["user"]

    # Cloud profiles is backed by auth.users. Therefore a guest/local user is
    # never pushed to cloud until external_auth_id is linked.
    external_auth_id = user.get("external_auth_id")
    if not external_auth_id or user_id != external_auth_id:
        return {
            "status": "deferred",
            "reason": "User is local-only. Link the Supabase Auth UUID before cloud sync.",
            "user_id": user_id,
        }

    cloud_profile = {
        "user_id": user_id,
        "name": user.get("name"),
        "phone_e164": user.get("phone_e164"),
        "preferred_language": user.get("preferred_language") or "hi",
    }
    cloud_upsert("profiles", [cloud_profile], "user_id")

    if ctx["devices"]:
        cloud_upsert("devices", ctx["devices"], "device_id")
    if ctx["locations"]:
        cloud_upsert(
            "user_locations",
            [
                {
                    "user_id": row["user_id"],
                    "local_body_code": row["local_body_code"],
                    "is_primary": bool(row["is_primary"]),
                }
                for row in ctx["locations"]
            ],
            "user_id,local_body_code",
        )
    if ctx["crop_profiles"]:
        cloud_upsert(
            "crop_profiles",
            [
                {
                    "crop_profile_id": row["crop_profile_id"],
                    "user_id": row["user_id"],
                    "local_body_code": row["local_body_code"],
                    "crop_name": row["crop_name"],
                    "growth_stage": row["growth_stage"],
                    "area_hectare": row["area_hectare"],
                    "notes": row["notes"],
                }
                for row in ctx["crop_profiles"]
            ],
            "crop_profile_id",
        )

    return {
        "status": "PASS",
        "user_id": user_id,
        "synced": {
            "profile": True,
            "devices": len(ctx["devices"]),
            "locations": len(ctx["locations"]),
            "crop_profiles": len(ctx["crop_profiles"]),
        },
    }


def status() -> dict[str, Any]:
    initialize()
    with _conn() as conn:
        counts = {}
        for table in (
            "users",
            "devices",
            "user_locations",
            "crop_profiles",
            "advisories",
            "notifications",
            "notification_deliveries",
            "sos_events",
            "sync_queue",
            "system_refresh_runs",
            "data_assets",
            "data_asset_versions",
        ):
            counts[table] = int(
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )

    return {
        "status": "PASS",
        "database": str(DB_PATH),
        "cloud_configured": cloud_configured(),
        "cloud_bucket": BUCKET,
        "counts": counts,
    }
