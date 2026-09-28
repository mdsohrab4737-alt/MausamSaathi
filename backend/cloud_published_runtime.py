from __future__ import annotations

from pathlib import Path
from typing import Any
import csv
import io
import json
import os
import shutil
import tempfile
import urllib.error
import urllib.parse
import urllib.request
import sys


BUCKET = os.environ.get("SUPABASE_BUCKET", "mausamsaathi-scientific").strip()
STATIC_PREFIX = "mausamsaathi"
PUBLISHED_RUNTIME_KEY = f"{STATIC_PREFIX}/published/runtime_manifest.json"
PUBLISHED_STATE_KEY = f"{STATIC_PREFIX}/published/current_state.json"

RUNTIME_FILES = {
    "prediction": "data_test/processed/current_climate_2026/mausam_current_ml_predictions.csv",
    "features": "data_test/processed/current_climate_2026/mausam_current_ml_features.csv",
    "anomaly": "data_test/processed/current_climate_2026/mausam_current_rainfall_anomaly.csv",
    "spatial": "data_test/processed/current_spatial_2026/mausam_current_spatial_current.geojson",
    "state": "data_test/processed/current_climate_2026/current_state.json",
}


def _load_local_env(project_root: Path) -> None:
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    try:
        for raw in env_path.read_text(encoding="utf-8-sig").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            name, value = line.split("=", 1)
            name, value = name.strip(), value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
                value = value[1:-1]
            if name and name not in os.environ:
                os.environ[name] = value
    except OSError:
        return


def cloud_runtime_configured(project_root: Path | None = None) -> bool:
    if project_root is not None:
        _load_local_env(project_root)
    project_url = os.environ.get("SUPABASE_URL", "").strip()
    key = (
        os.environ.get("SUPABASE_SECRET_KEY", "").strip()
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )
    return project_url.startswith("https://") and bool(key)


def _config(project_root: Path | None = None) -> tuple[str, str]:
    if project_root is not None:
        _load_local_env(project_root)
    project_url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    key = (
        os.environ.get("SUPABASE_SECRET_KEY", "").strip()
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )
    if not project_url.startswith("https://"):
        raise RuntimeError("SUPABASE_URL must be supplied and start with https://")
    if not key:
        raise RuntimeError("SUPABASE_SECRET_KEY is not configured.")
    return project_url, key


def _auth_headers(key: str) -> dict[str, str]:
    headers = {
        "apikey": key,
        "User-Agent": "MausamSaathi-backend/1.0",
    }
    if not key.startswith("sb_secret_"):
        headers["Authorization"] = f"Bearer {key}"
    return headers


def _storage_url(project_url: str, object_key: str) -> str:
    base = f"{project_url.rstrip('/')}/storage/v1/object"
    return (
        f"{base}/{urllib.parse.quote(BUCKET, safe='')}/"
        f"{urllib.parse.quote(object_key, safe='/._-')}"
    )


def _download(project_url: str, key: str, object_key: str) -> bytes:
    req = urllib.request.Request(
        _storage_url(project_url, object_key),
        headers=_auth_headers(key),
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = resp.read()
    except urllib.error.HTTPError as exc:
        body = exc.read()
        preview = body.decode("utf-8", errors="replace")[:700]
        raise RuntimeError(
            f"Supabase Storage download failed HTTP {exc.code}: {object_key}: {preview}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Supabase Storage download failed for {object_key}: {exc}"
        ) from exc
    if not payload:
        raise RuntimeError(f"Supabase Storage returned an empty object: {object_key}")
    return payload


def _json_object(project_url: str, key: str, object_key: str) -> dict[str, Any]:
    raw = _download(project_url, key, object_key)
    try:
        value = json.loads(raw.decode("utf-8-sig"))
    except Exception as exc:
        raise RuntimeError(f"Invalid JSON in published object: {object_key}") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"Published JSON object is not an object: {object_key}")
    return value


def published_state(project_root: Path | None = None) -> dict[str, Any]:
    project_url, key = _config(project_root)
    state = _json_object(project_url, key, PUBLISHED_STATE_KEY)
    if str(state.get("status", "")).upper() != "PASS":
        raise RuntimeError(
            f"Published current_state is not PASS: {state.get('status')!r}"
        )
    issue_date = str(state.get("last_successful_issue_date") or "").strip()
    if not issue_date:
        raise RuntimeError("Published current_state is missing last_successful_issue_date.")
    return state


def _published_runtime_manifest(project_root: Path | None = None) -> dict[str, Any]:
    project_url, key = _config(project_root)
    manifest = _json_object(project_url, key, PUBLISHED_RUNTIME_KEY)
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("Published runtime manifest has no valid 'files' list.")
    return manifest


def _resolve_object_key(manifest: dict[str, Any], relative_path: str) -> str:
    target = relative_path.replace("\\", "/").strip("/")
    matches: list[str] = []
    for row in manifest.get("files") or []:
        if not isinstance(row, dict):
            continue
        object_key = str(row.get("object_key") or "").replace("\\", "/").strip("/")
        if object_key.endswith("/" + target) or object_key == target:
            matches.append(object_key)
    if len(matches) != 1:
        raise RuntimeError(
            f"Published runtime does not uniquely contain {target}: {matches}"
        )
    return matches[0]


def published_runtime_object_key(relative_path: str, project_root: Path | None = None) -> str:
    return _resolve_object_key(
        _published_runtime_manifest(project_root),
        relative_path,
    )


def fetch_published_runtime_file(
    relative_path: str,
    project_root: Path | None = None,
) -> bytes:
    project_url, key = _config(project_root)
    manifest = _published_runtime_manifest(project_root)
    object_key = _resolve_object_key(manifest, relative_path)
    return _download(project_url, key, object_key)


def _local_issue_date(state_path: Path) -> str | None:
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    value = str(state.get("last_successful_issue_date") or "").strip()
    return value or None


def ensure_current_runtime(project_root: Path, current_climate_dir: Path) -> None:
    if not cloud_runtime_configured(project_root):
        return

    remote_state = published_state(project_root)
    remote_issue_date = str(
        remote_state.get("last_successful_issue_date") or ""
    ).strip()

    local_state_path = current_climate_dir / "current_state.json"
    if _local_issue_date(local_state_path) == remote_issue_date:
        return

    manifest = _published_runtime_manifest(project_root)
    project_url, key = _config(project_root)

    current_climate_dir.mkdir(parents=True, exist_ok=True)
    spatial_dir = project_root / "data_test" / "processed" / "current_spatial_2026"
    spatial_dir.mkdir(parents=True, exist_ok=True)

    targets = {
        "prediction": current_climate_dir / "mausam_current_ml_predictions.csv",
        "features": current_climate_dir / "mausam_current_ml_features.csv",
        "anomaly": current_climate_dir / "mausam_current_rainfall_anomaly.csv",
        "spatial": spatial_dir / "mausam_current_spatial_current.geojson",
        "state": local_state_path,
    }

    temp_dir = Path(tempfile.mkdtemp(prefix="mausamsaathi-cloud-cache-"))
    staged: dict[str, Path] = {}
    try:
        for name, destination in targets.items():
            relative_path = RUNTIME_FILES[name]
            object_key = _resolve_object_key(manifest, relative_path)
            payload = _download(project_url, key, object_key)
            stage = temp_dir / destination.name
            stage.write_bytes(payload)
            staged[name] = stage

        staged_state = json.loads(staged["state"].read_text(encoding="utf-8-sig"))
        staged_issue_date = str(
            staged_state.get("last_successful_issue_date") or ""
        ).strip()
        if staged_issue_date != remote_issue_date:
            raise RuntimeError(
                "Published state changed during synchronization: "
                f"expected {remote_issue_date}, got {staged_issue_date}"
            )

        for key_name in ("prediction", "anomaly", "features"):
            raw = staged[key_name].read_text(encoding="utf-8-sig")
            reader = csv.reader(io.StringIO(raw))
            if not next(reader, None):
                raise RuntimeError(
                    f"Published runtime CSV has no header: {RUNTIME_FILES[key_name]}"
                )

        for name, stage in staged.items():
            destination = targets[name]
            replacement = destination.with_name(
                destination.name + f".cloudsync.{os.getpid()}.part"
            )
            replacement.unlink(missing_ok=True)
            shutil.copy2(stage, replacement)
            os.replace(replacement, destination)
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def self_test(project_root: Path) -> None:
    import getpass

    _load_local_env(project_root)
    if not os.environ.get("SUPABASE_URL"):
        raise RuntimeError("SUPABASE_URL is not configured.")
    if not os.environ.get("SUPABASE_SECRET_KEY"):
        key = getpass.getpass("Supabase Secret Key: ").strip()
        if not key:
            raise RuntimeError("No Supabase Secret Key entered.")
        os.environ["SUPABASE_SECRET_KEY"] = key

    state = published_state(project_root)
    manifest = _published_runtime_manifest(project_root)

    print("=" * 76)
    print("MAUSAMSAATHI CLOUD PUBLISHED RUNTIME SELF-TEST")
    print("=" * 76)
    print("Bucket:", BUCKET)
    print("Published issue date:", state["last_successful_issue_date"])
    print("Published runtime files:", len(manifest["files"]))

    for name, relative_path in RUNTIME_FILES.items():
        object_key = _resolve_object_key(manifest, relative_path)
        payload = fetch_published_runtime_file(relative_path, project_root)
        print(f"{name}: PASS | {len(payload)} bytes | {object_key}")

    print("CLOUD PUBLISHED RUNTIME SELF-TEST: PASS")
    print("=" * 76)


if __name__ == "__main__":
    self_test(Path(__file__).resolve().parents[1])
