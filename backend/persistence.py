from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(r"C:\MausamSaathiSIH26\MAUSAMSAATHI_DEMO")
PERSISTENCE_DIR = PROJECT_ROOT / "data_test" / "persistence"
SQLITE_PATH = PERSISTENCE_DIR / "mausamsaathi_local.sqlite3"
MANIFEST_PATH = PERSISTENCE_DIR / "data_registry_manifest.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


SCHEMA_SQL = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS app_meta (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS locations (
    local_body_code TEXT PRIMARY KEY,
    state TEXT NOT NULL,
    district TEXT NOT NULL,
    block TEXT NOT NULL,
    panchayat TEXT NOT NULL,
    source_subdistrict TEXT,
    source TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_locations_district_block
    ON locations(district, block);

CREATE TABLE IF NOT EXISTS users (
    user_id TEXT PRIMARY KEY,
    external_auth_id TEXT UNIQUE,
    name TEXT,
    phone_e164 TEXT,
    preferred_language TEXT NOT NULL DEFAULT 'hi',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS devices (
    device_id TEXT PRIMARY KEY,
    user_id TEXT,
    platform TEXT,
    app_version TEXT,
    push_token TEXT,
    last_seen_at TEXT,
    is_active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_devices_user
    ON devices(user_id);

CREATE TABLE IF NOT EXISTS user_locations (
    user_id TEXT NOT NULL,
    local_body_code TEXT NOT NULL,
    is_primary INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (user_id, local_body_code),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (local_body_code) REFERENCES locations(local_body_code) ON DELETE RESTRICT
);

CREATE TABLE IF NOT EXISTS crop_profiles (
    crop_profile_id TEXT PRIMARY KEY,
    user_id TEXT NOT NULL,
    local_body_code TEXT NOT NULL,
    crop_name TEXT NOT NULL,
    growth_stage TEXT,
    area_hectare REAL,
    notes TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (local_body_code) REFERENCES locations(local_body_code) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_crop_profiles_user
    ON crop_profiles(user_id, local_body_code);

CREATE TABLE IF NOT EXISTS advisories (
    advisory_id TEXT PRIMARY KEY,
    local_body_code TEXT NOT NULL,
    crop_name TEXT,
    issue_date TEXT NOT NULL,
    horizon_days INTEGER NOT NULL,
    language TEXT NOT NULL DEFAULT 'hi',
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    actions_json TEXT NOT NULL,
    source TEXT NOT NULL,
    model_version TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (local_body_code) REFERENCES locations(local_body_code) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS idx_advisories_location_date
    ON advisories(local_body_code, issue_date DESC);

CREATE TABLE IF NOT EXISTS notifications (
    notification_id TEXT PRIMARY KEY,
    user_id TEXT,
    local_body_code TEXT,
    advisory_id TEXT,
    channel TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'hi',
    message TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    scheduled_at TEXT,
    sent_at TEXT,
    provider_message_id TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
    FOREIGN KEY (local_body_code) REFERENCES locations(local_body_code) ON DELETE SET NULL,
    FOREIGN KEY (advisory_id) REFERENCES advisories(advisory_id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_notifications_user_status
    ON notifications(user_id, status, created_at DESC);

CREATE TABLE IF NOT EXISTS notification_deliveries (
    delivery_id TEXT PRIMARY KEY,
    notification_id TEXT NOT NULL,
    attempt_no INTEGER NOT NULL DEFAULT 1,
    provider TEXT,
    provider_message_id TEXT,
    status TEXT NOT NULL,
    error_message TEXT,
    attempted_at TEXT NOT NULL,
    FOREIGN KEY (notification_id) REFERENCES notifications(notification_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_delivery_notification
    ON notification_deliveries(notification_id, attempted_at DESC);

CREATE TABLE IF NOT EXISTS sos_events (
    sos_event_id TEXT PRIMARY KEY,
    user_id TEXT,
    device_id TEXT,
    local_body_code TEXT,
    latitude REAL,
    longitude REAL,
    status TEXT NOT NULL DEFAULT 'created',
    triggered_at TEXT NOT NULL,
    resolved_at TEXT,
    payload_json TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE SET NULL,
    FOREIGN KEY (device_id) REFERENCES devices(device_id) ON DELETE SET NULL,
    FOREIGN KEY (local_body_code) REFERENCES locations(local_body_code) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_sos_status_time
    ON sos_events(status, triggered_at DESC);

CREATE TABLE IF NOT EXISTS sync_queue (
    sync_id TEXT PRIMARY KEY,
    entity_type TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    operation TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_sync_queue_pending
    ON sync_queue(status, next_attempt_at, created_at);

CREATE TABLE IF NOT EXISTS system_refresh_runs (
    refresh_run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    issue_date TEXT,
    status TEXT NOT NULL,
    pilot_district TEXT,
    local_body_count INTEGER,
    model_source TEXT,
    manifest_path TEXT,
    error TEXT,
    metadata_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_refresh_runs_time
    ON system_refresh_runs(started_at DESC);

CREATE TABLE IF NOT EXISTS data_assets (
    asset_id TEXT PRIMARY KEY,
    logical_key TEXT NOT NULL UNIQUE,
    asset_kind TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    source_name TEXT,
    is_critical INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS data_asset_versions (
    asset_version_id TEXT PRIMARY KEY,
    asset_id TEXT NOT NULL,
    version_label TEXT NOT NULL,
    issue_date TEXT,
    local_path TEXT,
    object_bucket TEXT,
    object_key TEXT,
    sha256 TEXT,
    size_bytes INTEGER,
    status TEXT NOT NULL DEFAULT 'discovered',
    verified_at TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (asset_id) REFERENCES data_assets(asset_id) ON DELETE CASCADE,
    UNIQUE (asset_id, version_label)
);

CREATE INDEX IF NOT EXISTS idx_asset_versions_lookup
    ON data_asset_versions(asset_id, issue_date DESC);

CREATE INDEX IF NOT EXISTS idx_asset_versions_sha
    ON data_asset_versions(sha256);
"""


@contextmanager
def connect(path: Path = SQLITE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=FULL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def initialize_database(path: Path = SQLITE_PATH) -> None:
    with connect(path) as conn:
        conn.executescript(SCHEMA_SQL)
        now = utc_now()
        conn.execute(
            """
            INSERT INTO app_meta(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            ("schema_version", "step17", now),
        )
        conn.execute(
            """
            INSERT INTO app_meta(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            ("project_root", str(PROJECT_ROOT), now),
        )


def upsert_locations(rows: Iterable[dict[str, Any]], path: Path = SQLITE_PATH) -> int:
    count = 0
    now = utc_now()
    with connect(path) as conn:
        for row in rows:
            conn.execute(
                """
                INSERT INTO locations(
                    local_body_code, state, district, block, panchayat,
                    source_subdistrict, source, is_active, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                ON CONFLICT(local_body_code) DO UPDATE SET
                    state=excluded.state,
                    district=excluded.district,
                    block=excluded.block,
                    panchayat=excluded.panchayat,
                    source_subdistrict=excluded.source_subdistrict,
                    source=excluded.source,
                    is_active=1,
                    updated_at=excluded.updated_at
                """,
                (
                    str(row["local_body_code"]),
                    str(row["state"]),
                    str(row["district"]),
                    str(row["block"]),
                    str(row["panchayat"]),
                    row.get("source_subdistrict"),
                    str(row.get("source", "LGD verified mapping")),
                    now,
                    now,
                ),
            )
            count += 1
    return count


def upsert_asset(
    *,
    logical_key: str,
    asset_kind: str,
    dataset_name: str,
    source_name: str | None,
    is_critical: bool,
    version_label: str,
    issue_date: str | None,
    local_path: str | None,
    object_bucket: str | None,
    object_key: str | None,
    sha256: str | None,
    size_bytes: int | None,
    status: str,
    verified_at: str | None,
    metadata: dict[str, Any] | None = None,
    path: Path = SQLITE_PATH,
) -> None:
    now = utc_now()
    asset_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"mausamsaathi:asset:{logical_key}"))
    version_id = str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"mausamsaathi:asset-version:{logical_key}:{version_label}",
        )
    )
    with connect(path) as conn:
        conn.execute(
            """
            INSERT INTO data_assets(
                asset_id, logical_key, asset_kind, dataset_name,
                source_name, is_critical, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(logical_key) DO UPDATE SET
                asset_kind=excluded.asset_kind,
                dataset_name=excluded.dataset_name,
                source_name=excluded.source_name,
                is_critical=excluded.is_critical,
                updated_at=excluded.updated_at
            """,
            (
                asset_id,
                logical_key,
                asset_kind,
                dataset_name,
                source_name,
                int(is_critical),
                now,
                now,
            ),
        )

        conn.execute(
            """
            INSERT INTO data_asset_versions(
                asset_version_id, asset_id, version_label, issue_date,
                local_path, object_bucket, object_key, sha256, size_bytes,
                status, verified_at, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(asset_id, version_label) DO UPDATE SET
                issue_date=excluded.issue_date,
                local_path=excluded.local_path,
                object_bucket=excluded.object_bucket,
                object_key=excluded.object_key,
                sha256=excluded.sha256,
                size_bytes=excluded.size_bytes,
                status=excluded.status,
                verified_at=excluded.verified_at,
                metadata_json=excluded.metadata_json
            """,
            (
                version_id,
                asset_id,
                version_label,
                issue_date,
                local_path,
                object_bucket,
                object_key,
                sha256,
                size_bytes,
                status,
                verified_at,
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
                now,
            ),
        )


def record_refresh_run(
    *,
    refresh_run_id: str,
    started_at: str,
    finished_at: str | None,
    issue_date: str | None,
    status: str,
    pilot_district: str | None,
    local_body_count: int | None,
    model_source: str | None,
    manifest_path: str | None,
    error: str | None,
    metadata: dict[str, Any] | None = None,
    path: Path = SQLITE_PATH,
) -> None:
    with connect(path) as conn:
        conn.execute(
            """
            INSERT INTO system_refresh_runs(
                refresh_run_id, started_at, finished_at, issue_date, status,
                pilot_district, local_body_count, model_source, manifest_path,
                error, metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(refresh_run_id) DO UPDATE SET
                finished_at=excluded.finished_at,
                issue_date=excluded.issue_date,
                status=excluded.status,
                pilot_district=excluded.pilot_district,
                local_body_count=excluded.local_body_count,
                model_source=excluded.model_source,
                manifest_path=excluded.manifest_path,
                error=excluded.error,
                metadata_json=excluded.metadata_json
            """,
            (
                refresh_run_id,
                started_at,
                finished_at,
                issue_date,
                status,
                pilot_district,
                local_body_count,
                model_source,
                manifest_path,
                error,
                json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True),
            ),
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def database_counts(path: Path = SQLITE_PATH) -> dict[str, int]:
    tables = [
        "locations",
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
    ]
    counts: dict[str, int] = {}
    with connect(path) as conn:
        for table in tables:
            counts[table] = int(
                conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            )
    return counts
