from __future__ import annotations

import argparse
import hashlib
import getpass
import json
import mimetypes
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib.error
import urllib.parse
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "backend" / "cloud_refresh_artifact_manifest.json"
REFRESH_SCRIPT = ROOT / "backend" / "refresh_current_bilaspur.py"

BUCKET = os.environ.get("SUPABASE_BUCKET", "mausamsaathi-scientific").strip()
STATIC_PREFIX = "mausamsaathi"
PUBLISHED_RUNTIME_KEY = f"{STATIC_PREFIX}/published/runtime_manifest.json"
PUBLISHED_STATE_KEY = f"{STATIC_PREFIX}/published/current_state.json"


def log(message: str) -> None:
    print(f"[cloud-worker] {message}", flush=True)


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def load_local_env() -> None:
    """Load simple KEY=VALUE entries from local .env files without requiring python-dotenv."""
    for env_path in (ROOT / ".env", ROOT / "backend" / ".env"):
        if not env_path.exists():
            continue
        try:
            for raw in env_path.read_text(encoding="utf-8").splitlines():
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except Exception as exc:
            log(f"Could not read {env_path}: {exc}")


def get_supabase_config() -> tuple[str, str]:
    load_local_env()
    project_url = os.environ.get("SUPABASE_URL", "").strip().rstrip("/")
    key = (
        os.environ.get("SUPABASE_SECRET_KEY", "").strip()
        or os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    )
    if not project_url.startswith("https://"):
        raise RuntimeError("SUPABASE_URL must be supplied and start with https://")

    placeholder_keys = {
        "sb_secret_REPLACE_WITH_YOUR_REAL_SECRET",
        "replace_with_your_real_secret",
    }
    if not key or key in placeholder_keys:
        print("Supabase secret key is not configured in the local .env.", flush=True)
        print("Enter the real Supabase Secret Key now. Input will be hidden.", flush=True)
        key = getpass.getpass("Supabase Secret Key: ").strip()

    if not key or key in placeholder_keys:
        raise RuntimeError("A real SUPABASE_SECRET_KEY is required.")

    return project_url, key


def auth_headers(key: str) -> dict[str, str]:
    headers = {
        "apikey": key,
        "User-Agent": "MausamSaathi-cloud-refresh/1.0",
    }
    # Legacy service-role JWTs are accepted through Authorization.
    # Modern sb_secret_* keys are opaque API keys and should not be sent as Bearer JWTs.
    if not key.startswith("sb_secret_"):
        headers["Authorization"] = f"Bearer {key}"
    return headers


def storage_url(project_url: str, object_key: str) -> str:
    # Use the canonical project API hostname for object REST calls. Supabase also
    # exposes a direct storage hostname, but the existing Step-17 MausamSaathi
    # backup implementation used the project URL and the existing bucket is
    # already proven through that path.
    base = f"{project_url.rstrip('/')}/storage/v1/object"
    return f"{base}/{urllib.parse.quote(BUCKET, safe='')}/{urllib.parse.quote(object_key, safe='/._-')}"


def request(
    project_url: str,
    key: str,
    object_key: str,
    *,
    method: str = "GET",
    body: bytes | None = None,
    content_type: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> tuple[int, bytes, dict[str, str]]:
    headers = auth_headers(key)
    if content_type:
        headers["Content-Type"] = content_type
    if extra_headers:
        headers.update(extra_headers)

    req = urllib.request.Request(
        storage_url(project_url, object_key),
        data=body,
        headers=headers,
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            return resp.status, resp.read(), dict(resp.headers.items())
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        return exc.code, payload, dict(exc.headers.items())
    except Exception:
        raise


def verify_bucket(
    project_url: str,
    key: str,
    probe_object_key: str | None = None,
) -> None:
    # Modern Supabase sb_secret_* keys are opaque API keys. They belong in the
    # apikey header, not Authorization: Bearer. The bucket metadata endpoint in
    # this environment is demanding an Authorization header, so for modern keys
    # we verify the actual Storage object route instead.
    if key.startswith(("sb_secret_", "sb_publishable_")):
        if not probe_object_key:
            raise RuntimeError("Storage probe object is required for modern Supabase API-key verification.")
        status, payload, _ = request(
            project_url,
            key,
            probe_object_key,
            method="GET",
            extra_headers={"Range": "bytes=0-0"},
        )
        if status in {200, 206}:
            log(f"Supabase Storage access: PASS (object readable, bucket={BUCKET})")
            return
        if status == 404:
            # A missing object is fine, but Supabase may also return a generic
            # bucket-not-found response for an unauthenticated/invalid key.
            # Keep the body available for diagnostics instead of mislabeling it.
            body_text = payload.decode("utf-8", errors="replace")
            if "NoSuchBucket" in body_text or "Bucket not found" in body_text:
                raise RuntimeError(
                    "Supabase rejected the Storage request as bucket-not-found. "
                    "Verify the real Secret Key for the project and retry."
                )
            log(f"Supabase Storage endpoint reachable; probe object absent (bucket={BUCKET})")
            return
        preview = payload.decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Supabase Storage access check failed HTTP {status}: {preview}")

    # Legacy service_role JWTs can use the bucket metadata admin route.
    parsed = urllib.parse.urlparse(project_url)
    if parsed.netloc.endswith(".supabase.co"):
        ref = parsed.netloc[: -len(".supabase.co")]
        host = f"https://{ref}.supabase.co"
    else:
        host = project_url
    url = f"{host}/storage/v1/bucket/{urllib.parse.quote(BUCKET, safe='')}"
    req = urllib.request.Request(
        url,
        headers=auth_headers(key),
        method="GET",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Supabase bucket check failed HTTP {exc.code}: {body}") from exc
    if payload.get("public") is not False:
        raise RuntimeError(
            f"Bucket {BUCKET!r} is not confirmed private. Refusing scientific-data publish."
        )
    log(f"Supabase bucket metadata: PASS (private, bucket={BUCKET})")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_rel_path(value: str) -> Path:
    p = Path(value.replace("\\", "/"))
    if p.is_absolute() or ".." in p.parts:
        raise RuntimeError(f"Unsafe relative path in artifact manifest: {value!r}")
    return p


def load_artifact_manifest() -> dict[str, Any]:
    if not MANIFEST_PATH.exists():
        raise RuntimeError(f"Cloud artifact manifest not found: {MANIFEST_PATH}")
    payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise RuntimeError("cloud_refresh_artifact_manifest.json has no 'artifacts' list")
    return payload


def download_object(
    project_url: str,
    key: str,
    object_key: str,
    destination: Path,
    *,
    allow_missing: bool = False,
) -> bool:
    """
    Download a Storage object using byte ranges.

    Uses a process-specific temporary filename and retries the final atomic
    replacement because Windows can transiently lock freshly-created files
    (for example while antivirus/indexing scans them).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    part = destination.with_name(
        destination.name
        + f".{os.getpid()}.part"
    )
    part.unlink(missing_ok=True)

    chunk_size = 8 * 1024 * 1024
    position = 0
    total_size: int | None = None
    first_request = True

    try:
        with part.open("wb") as out:
            while True:
                end = position + chunk_size - 1
                status, payload, headers = request(
                    project_url,
                    key,
                    object_key,
                    method="GET",
                    extra_headers={"Range": f"bytes={position}-{end}"},
                )

                if first_request and allow_missing and (
                    status == 404
                    or (status == 400 and b"NoSuchKey" in payload)
                    or (status == 400 and b"Object not found" in payload)
                    or (status == 400 and b"not_found" in payload)
                ):
                    return False

                first_request = False

                if status not in {200, 206}:
                    preview = payload.decode("utf-8", errors="replace")[:700]
                    raise RuntimeError(
                        f"Supabase ranged download failed HTTP {status}: "
                        f"{object_key}: {preview}"
                    )

                content_range = headers.get("Content-Range") or headers.get("content-range")
                if content_range and "/" in content_range:
                    try:
                        total_token = content_range.rsplit("/", 1)[1]
                        if total_token != "*":
                            total_size = int(total_token)
                    except (ValueError, TypeError):
                        pass

                out.write(payload)
                received = len(payload)

                if status == 200:
                    position += received
                    break

                position += received

                if total_size is not None and position >= total_size:
                    break

                if received == 0:
                    break

                if received < chunk_size and total_size is None:
                    break

        if not part.exists() or part.stat().st_size <= 0:
            raise RuntimeError(f"Empty download from Supabase: {object_key}")

        if total_size is not None and part.stat().st_size != total_size:
            actual = part.stat().st_size
            raise RuntimeError(
                f"Incomplete Supabase download: {object_key}; "
                f"expected {total_size} bytes, got {actual}"
            )

        # Windows can briefly lock a new file after close(). Retry the atomic
        # replace instead of failing an otherwise successful cloud restore.
        last_error: Exception | None = None
        for attempt in range(1, 16):
            try:
                os.replace(part, destination)
                last_error = None
                break
            except OSError as exc:
                last_error = exc
                if attempt == 15:
                    raise
                import time
                time.sleep(0.5)

        if last_error is not None:
            raise last_error

        return True

    except Exception:
        part.unlink(missing_ok=True)
        raise

def upload_file(
    project_url: str,
    key: str,
    local_path: Path,
    object_key: str,
    *,
    upsert: bool,
) -> None:
    if not local_path.exists() or not local_path.is_file():
        raise RuntimeError(f"Cannot upload missing file: {local_path}")

    data = local_path.read_bytes()
    mime = mimetypes.guess_type(local_path.name)[0] or "application/octet-stream"
    status, payload, _ = request(
        project_url,
        key,
        object_key,
        method="POST",
        body=data,
        content_type=mime,
        extra_headers={
            "x-upsert": "true" if upsert else "false",
            "cache-control": "3600",
        },
    )
    if status not in {200, 201}:
        preview = payload.decode("utf-8", errors="replace")[:700]
        # A non-upsert bootstrap upload can race with a previous run.
        if status in {400, 409} and b"already exists" in payload.lower():
            return
        raise RuntimeError(f"Supabase upload failed HTTP {status}: {object_key}: {preview}")


def prepare_cds_credentials(work_dir: Path) -> None:
    cds_rc = os.environ.get("CDSAPI_RC", "")
    if not cds_rc.strip():
        return
    home = work_dir / ".home"
    home.mkdir(parents=True, exist_ok=True)
    rc_path = home / ".cdsapirc"
    rc_path.write_text(cds_rc.strip() + "\n", encoding="utf-8")
    os.environ["HOME"] = str(home)
    os.environ["USERPROFILE"] = str(home)
    log("CDS credentials loaded from CDSAPI_RC into isolated HOME.")


def stage_source_code(work_dir: Path) -> None:
    """Stage refresh source files into the ephemeral workspace without copying local runtime data."""
    backend_src = ROOT / "backend"
    backend_dst = work_dir / "backend"
    backend_dst.mkdir(parents=True, exist_ok=True)

    if backend_src.exists():
        for src in backend_src.glob("*.py"):
            shutil.copy2(src, backend_dst / src.name)
        data_src = backend_src / "data"
        if data_src.exists():
            shutil.copytree(data_src, backend_dst / "data", dirs_exist_ok=True)

    for src in ROOT.glob("*.py"):
        shutil.copy2(src, work_dir / src.name)

    manifest_src = ROOT / "backend" / "cloud_refresh_artifact_manifest.json"
    if manifest_src.exists():
        shutil.copy2(manifest_src, work_dir / "backend" / manifest_src.name)

    frontend_public = ROOT / "frontend" / "public" / "mausamspatial"
    if frontend_public.exists():
        shutil.copytree(
            frontend_public,
            work_dir / "frontend" / "public" / "mausamspatial",
            dirs_exist_ok=True,
        )

    for secret_rel in (".env", "backend/.env", ".cdsapirc"):
        (work_dir / secret_rel).unlink(missing_ok=True)



def storage_list_prefix(
    project_url: str,
    key: str,
    prefix: str,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """List Storage objects under a prefix using Supabase's object-list endpoint."""
    parsed = urllib.parse.urlparse(project_url)
    if parsed.netloc.endswith(".supabase.co"):
        ref = parsed.netloc[: -len(".supabase.co")]
        base = f"https://{ref}.supabase.co"
    else:
        base = project_url

    url = f"{base}/storage/v1/object/list/{urllib.parse.quote(BUCKET, safe='')}"
    headers = auth_headers(key)
    headers["Content-Type"] = "application/json"

    items: list[dict[str, Any]] = []
    offset = 0
    while True:
        payload = json.dumps(
            {
                "prefix": prefix.rstrip("/") + "/",
                "limit": limit,
                "offset": offset,
                "sortBy": {"column": "name", "order": "asc"},
            },
            separators=(",", ":"),
        ).encode("utf-8")

        req = urllib.request.Request(
            url,
            data=payload,
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
                status = resp.status
        except urllib.error.HTTPError as exc:
            body = exc.read()
            status = exc.code

        if status != 200:
            preview = body.decode("utf-8", errors="replace")[:800]
            raise RuntimeError(
                f"Supabase Storage list failed HTTP {status}: {preview}"
            )

        batch = json.loads(body.decode("utf-8"))
        if not isinstance(batch, list):
            raise RuntimeError("Supabase Storage list response was not an array.")

        items.extend(x for x in batch if isinstance(x, dict))
        if len(batch) < limit:
            break
        offset += limit

    return items


def restore_legacy_raw_gpm_history(
    project_url: str,
    key: str,
    work_dir: Path,
) -> dict[str, Any]:
    """
    Restore the older Step-17 raw GPM GRD history when it exists in Storage.

    The refined 806-artifact manifest intentionally excludes raw GPM history,
    but the earlier 949-asset backup included it. Restoring those files prevents
    the cloud worker from trying to re-fetch the whole Aug/Sep history before
    it can begin a true incremental daily refresh.
    """
    prefix = f"{STATIC_PREFIX}/data_test/processed/current_climate_2026/raw_gpm"
    raw_dir = (
        work_dir
        / "data_test"
        / "processed"
        / "current_climate_2026"
        / "raw_gpm"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        items = storage_list_prefix(project_url, key, prefix)
    except Exception as exc:
        log(f"Legacy raw GPM listing unavailable: {exc}")
        return {
            "status": "NOT_AVAILABLE",
            "listed": 0,
            "restored": 0,
            "error": str(exc),
        }

    files = [
        item for item in items
        if str(item.get("name", "")).lower().endswith(".grd")
    ]

    if not files:
        log("No legacy raw GPM GRD objects found in Storage.")
        return {
            "status": "NOT_FOUND",
            "listed": len(items),
            "restored": 0,
        }

    restored = 0
    skipped = 0
    empty_remote = 0
    failed = 0
    for item in files:
        name = Path(str(item.get("name", ""))).name
        if not name:
            continue

        # The Step-17 archive may contain zero-byte placeholder objects for dates
        # that were unavailable from IMD. They are not usable GPM inputs and must
        # not abort a clean cloud refresh.
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        reported_size = metadata.get("size")
        try:
            reported_size_int = int(reported_size) if reported_size is not None else None
        except (TypeError, ValueError):
            reported_size_int = None

        if reported_size_int == 0:
            empty_remote += 1
            log(f"Legacy raw GPM: skipping zero-byte remote object {name}")
            continue

        destination = raw_dir / name
        if destination.exists() and destination.stat().st_size > 0:
            skipped += 1
            continue

        object_key = f"{prefix}/{name}"
        try:
            download_object(project_url, key, object_key, destination)
            if destination.stat().st_size <= 0:
                destination.unlink(missing_ok=True)
                empty_remote += 1
                log(f"Legacy raw GPM: skipping empty downloaded object {name}")
                continue
            restored += 1
        except Exception as exc:
            failed += 1
            raise RuntimeError(
                f"Failed restoring legacy raw GPM {object_key}: {exc}"
            ) from exc

    log(
        f"Legacy raw GPM restore: {restored} downloaded, "
        f"{skipped} already present, {empty_remote} empty placeholders skipped, "
        f"{len(files)} GRD objects listed."
    )
    return {
        "status": "PASS",
        "listed": len(items),
        "grd_objects": len(files),
        "restored": restored,
        "already_present": skipped,
        "empty_placeholders_skipped": empty_remote,
        "failed": failed,
    }


def restore_static_artifacts(
    project_url: str,
    key: str,
    work_dir: Path,
    manifest: dict[str, Any],
) -> dict[str, Any]:
    artifacts = manifest["artifacts"]
    restored = 0
    failures: list[dict[str, str]] = []

    log(f"Restoring {len(artifacts)} static/runtime artifacts from Supabase...")
    for index, asset in enumerate(artifacts, start=1):
        rel = safe_rel_path(str(asset.get("path", "")))
        if not str(rel):
            failures.append({"path": "", "error": "missing path"})
            continue
        destination = work_dir / rel
        object_key = f"{STATIC_PREFIX}/{rel.as_posix()}"
        try:
            download_object(project_url, key, object_key, destination)
            restored += 1
        except Exception as exc:
            failures.append({"path": rel.as_posix(), "error": str(exc)})
        if index % 50 == 0 or index == len(artifacts):
            log(f"Static restore progress: {index}/{len(artifacts)}")

    if failures:
        raise RuntimeError(
            f"Static artifact restore failed for {len(failures)} artifacts. "
            f"First failure: {failures[0]}"
        )
    return {"expected": len(artifacts), "restored": restored, "failures": failures}


def restore_prior_runtime(
    project_url: str,
    key: str,
    work_dir: Path,
) -> dict[str, Any] | None:
    pointer = work_dir / ".previous_runtime_manifest.json"
    log(f"Checking for previously published runtime: {PUBLISHED_RUNTIME_KEY}")
    if not download_object(
        project_url,
        key,
        PUBLISHED_RUNTIME_KEY,
        pointer,
        allow_missing=True,
    ):
        log("No published runtime manifest exists yet. Treating this as first cloud refresh.")
        return None

    runtime = json.loads(pointer.read_text(encoding="utf-8"))
    files = runtime.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Published runtime manifest has no valid 'files' list.")

    log(f"Restoring previous published runtime: {len(files)} files")
    for index, row in enumerate(files, start=1):
        rel = safe_rel_path(str(row["path"]))
        object_key = str(row["object_key"])
        destination = work_dir / rel
        download_object(project_url, key, object_key, destination)

        expected_sha = str(row.get("sha256", ""))
        if expected_sha and sha256_file(destination) != expected_sha:
            raise RuntimeError(f"Runtime SHA mismatch after restore: {rel.as_posix()}")

        if index % 50 == 0 or index == len(files):
            log(f"Runtime restore progress: {index}/{len(files)}")

    return runtime


def restore_recovery_db(
    project_url: str,
    key: str,
    work_dir: Path,
) -> bool:
    target = (
        work_dir
        / "data_test"
        / "processed"
        / "persistence"
        / "mausamsaathi_local.sqlite3"
    )
    if target.exists() and target.stat().st_size > 0:
        return True
    object_key = f"{STATIC_PREFIX}/recovery/mausamsaathi_local.sqlite3"
    if download_object(
        project_url,
        key,
        object_key,
        target,
        allow_missing=True,
    ):
        log("Restored Step-17 recovery SQLite DB.")
        return True
    return False


def prepare_cloud_environment(work_dir: Path) -> None:
    empty_gpm = work_dir / "cloud_external_gpm_import"
    empty_gpm.mkdir(parents=True, exist_ok=True)
    os.environ["MS_GPM_IMPORT_DIR"] = str(empty_gpm)
    os.environ["MAUSAMSAATHI_CLOUD_REFRESH"] = "1"
    prepare_cds_credentials(work_dir)


def run_existing_refresh(work_dir: Path) -> subprocess.CompletedProcess[str]:
    if not REFRESH_SCRIPT.exists():
        raise RuntimeError(f"Refresh entrypoint missing: {REFRESH_SCRIPT}")

    log(f"Running existing refresh pipeline: {REFRESH_SCRIPT.name}")
    log("Child stdout/stderr will be streamed below.")

    command = [
        sys.executable,
        str(work_dir / "backend" / "refresh_current_bilaspur.py"),
    ]

    proc = subprocess.Popen(
        command,
        cwd=work_dir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
        bufsize=1,
    )

    output_lines: list[str] = []
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            print("[refresh-child] " + line.rstrip(), flush=True)
            output_lines.append(line)
            # Keep a bounded in-memory copy for the parent failure report.
            if len(output_lines) > 3000:
                output_lines = output_lines[-3000:]
    finally:
        if proc.stdout is not None:
            proc.stdout.close()

    returncode = proc.wait(timeout=90 * 60)
    output = "".join(output_lines)

    if returncode != 0:
        failure_log = work_dir / "cloud_refresh_child_failure.log"
        failure_log.write_text(output, encoding="utf-8", errors="replace")
        log(f"Child refresh returned {returncode}.")
        log("========== CHILD FAILURE TAIL ==========")
        for line in output.splitlines()[-120:]:
            print("[refresh-child] " + line, flush=True)
        log("========== END CHILD FAILURE TAIL ==========")

    return subprocess.CompletedProcess(
        args=command,
        returncode=returncode,
        stdout=output,
        stderr="",
    )


def locate_current_state(work_dir: Path) -> Path:
    candidates = [
        work_dir / "data_test" / "processed" / "current_climate_2026" / "current_state.json",
        work_dir / "backend" / "data" / "current_state.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    matches = list(work_dir.glob("**/current_state.json"))
    if matches:
        return matches[0]
    raise RuntimeError("No current_state.json found after refresh.")


def validate_success_state(state_path: Path) -> dict[str, Any]:
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    status = str(payload.get("status", "")).upper()
    if status != "PASS":
        raise RuntimeError(f"Refresh produced non-PASS current_state: {status!r}")

    stale_error = payload.get("error")
    stale_trace = payload.get("traceback_tail")
    if stale_error or stale_trace:
        raise RuntimeError(
            "Refresh status is PASS but current_state still contains stale error/traceback fields."
        )
    if not payload.get("last_successful_issue_date"):
        raise RuntimeError("PASS current_state is missing last_successful_issue_date.")
    return payload


def should_include_runtime(path: Path, work_dir: Path) -> bool:
    rel = path.relative_to(work_dir)
    parts = rel.parts

    # Avoid raw GPM inputs and temporary OLR append files. The cumulative rainfall/
    # atmosphere files and rebuilt OLR base are preserved elsewhere in the runtime tree.
    if "raw_gpm" in parts:
        return False
    if "refresh_olr_daily" in parts:
        return False
    if "original_eofs" in parts:
        return False
    if "__pycache__" in parts:
        return False

    current_climate = Path("data_test/processed/current_climate_2026")
    current_omi = Path("data_test/processed/current_omi")
    geo_bilaspur = Path("data_test/geo/bilaspur")
    frontend_maps = Path("frontend/public/mausamspatial")

    if rel == Path("data_test/processed/persistence/mausamsaathi_local.sqlite3"):
        return True
    if rel.parts[: len(current_climate.parts)] == current_climate.parts:
        return True
    if rel.parts[: len(current_omi.parts)] == current_omi.parts:
        return True
    if rel.parts[: len(frontend_maps.parts)] == frontend_maps.parts:
        return True
    if rel.parts[: len(geo_bilaspur.parts)] == geo_bilaspur.parts:
        return path.suffix.lower() in {".geojson", ".json"} and any(
            token in path.name.lower() for token in ("current", "spatial", "audit")
        )
    return False


def collect_runtime_files(work_dir: Path) -> list[Path]:
    result: list[Path] = []
    for base in [
        work_dir / "data_test" / "processed" / "current_climate_2026",
        work_dir / "data_test" / "processed" / "current_omi",
        work_dir
        / "data_test"
        / "processed"
        / "persistence"
        / "mausamsaathi_local.sqlite3",
        work_dir / "frontend" / "public" / "mausamspatial",
        work_dir / "data_test" / "geo" / "bilaspur",
    ]:
        if base.is_file():
            if should_include_runtime(base, work_dir):
                result.append(base)
        elif base.is_dir():
            for path in base.rglob("*"):
                if path.is_file() and should_include_runtime(path, work_dir):
                    result.append(path)

    # Deduplicate while preserving deterministic order.
    unique = {p.resolve(): p for p in result}
    return sorted(unique.values(), key=lambda p: p.relative_to(work_dir).as_posix())


def upload_runtime_snapshot(
    project_url: str,
    key: str,
    work_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    issue_date = str(state["last_successful_issue_date"])
    run_id = f"refresh-{issue_date.replace('-', '')}-{uuid.uuid4().hex[:12]}"
    files = collect_runtime_files(work_dir)
    if not files:
        raise RuntimeError("No runtime files found for cloud publish.")

    log(f"Preparing immutable runtime snapshot {run_id}: {len(files)} files")
    entries: list[dict[str, Any]] = []

    for index, path in enumerate(files, start=1):
        rel = path.relative_to(work_dir).as_posix()
        object_key = f"{STATIC_PREFIX}/runtime/{run_id}/{rel}"
        digest = sha256_file(path)
        upload_file(project_url, key, path, object_key, upsert=False)
        entries.append(
            {
                "path": rel,
                "object_key": object_key,
                "sha256": digest,
                "size_bytes": path.stat().st_size,
            }
        )
        if index % 25 == 0 or index == len(files):
            log(f"Runtime upload progress: {index}/{len(files)}")

    manifest = {
        "schema": 1,
        "status": "PUBLISHED",
        "published_at": now_iso(),
        "run_id": run_id,
        "issue_date": issue_date,
        "file_count": len(entries),
        "files": entries,
    }

    version_manifest = work_dir / ".published_runtime_manifest.json"
    version_manifest.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    version_manifest_key = f"{STATIC_PREFIX}/runtime/{run_id}/runtime_manifest.json"
    upload_file(project_url, key, version_manifest, version_manifest_key, upsert=False)

    # The pointer is written last. If anything before this point fails, the old
    # published runtime remains untouched.
    manifest["version_manifest_object_key"] = version_manifest_key
    pointer_path = work_dir / ".published_runtime_pointer.json"
    pointer_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    upload_file(
        project_url,
        key,
        pointer_path,
        PUBLISHED_RUNTIME_KEY,
        upsert=True,
    )

    state_path = locate_current_state(work_dir)
    upload_file(
        project_url,
        key,
        state_path,
        PUBLISHED_STATE_KEY,
        upsert=True,
    )

    return manifest


def upload_run_audit(
    project_url: str,
    key: str,
    run_id: str,
    payload: dict[str, Any],
    work_dir: Path,
) -> None:
    audit_path = work_dir / f".run-audit-{run_id}.json"
    audit_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    upload_file(
        project_url,
        key,
        audit_path,
        f"{STATIC_PREFIX}/runs/{run_id}.json",
        upsert=True,
    )


def bootstrap_static_artifacts() -> int:
    project_url, key = get_supabase_config()
    manifest = load_artifact_manifest()
    artifacts = manifest["artifacts"]
    probe_rel = safe_rel_path(str(artifacts[0]["path"]))
    verify_bucket(project_url, key, f"{STATIC_PREFIX}/{probe_rel.as_posix()}")

    checked = 0
    uploaded = 0
    existing = 0

    recovery_db = ROOT / "data_test" / "persistence" / "mausamsaathi_local.sqlite3"

    for index, asset in enumerate(artifacts, start=1):
        rel = safe_rel_path(str(asset.get("path", "")))
        local_path = ROOT / rel
        if not local_path.exists() or not local_path.is_file():
            raise RuntimeError(f"Manifest artifact missing locally: {rel.as_posix()}")
        if local_path.stat().st_size <= 0:
            raise RuntimeError(f"Manifest artifact is empty: {rel.as_posix()}")

        object_key = f"{STATIC_PREFIX}/{rel.as_posix()}"
        status, payload, _ = request(
            project_url,
            key,
            object_key,
            method="GET",
            extra_headers={"Range": "bytes=0-0"},
        )
        if status in {200, 206}:
            existing += 1
        elif status == 404:
            upload_file(project_url, key, local_path, object_key, upsert=False)
            uploaded += 1
        elif status == 400 and b"NoSuchKey" in payload:
            # Supabase Storage can return HTTP 400 with a NoSuchKey body for
            # a missing object instead of HTTP 404.
            upload_file(project_url, key, local_path, object_key, upsert=False)
            uploaded += 1
        else:
            preview = payload.decode("utf-8", errors="replace")[:500]
            raise RuntimeError(f"Bootstrap existence check failed HTTP {status}: {object_key}: {preview}")

        checked += 1
        if index % 50 == 0 or index == len(artifacts):
            log(f"Bootstrap progress: {index}/{len(artifacts)}")

    if recovery_db.exists():
        recovery_key = f"{STATIC_PREFIX}/recovery/mausamsaathi_local.sqlite3"
        upload_file(project_url, key, recovery_db, recovery_key, upsert=True)

    bootstrap_report = {
        "status": "PASS",
        "timestamp": now_iso(),
        "manifest_artifact_count": len(artifacts),
        "checked": checked,
        "uploaded": uploaded,
        "already_existing": existing,
        "recovery_db_uploaded": recovery_db.exists(),
    }
    report_path = ROOT / "backend" / "cloud_bootstrap_last_report.json"
    report_path.write_text(
        json.dumps(bootstrap_report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    upload_file(
        project_url,
        key,
        report_path,
        f"{STATIC_PREFIX}/runs/bootstrap_last_report.json",
        upsert=True,
    )
    log(json.dumps(bootstrap_report, indent=2))
    return 0


def perform_cloud_refresh() -> int:
    project_url, key = get_supabase_config()
    manifest = load_artifact_manifest()

    host_tmp = Path(os.environ.get("MS_CLOUD_WORK_ROOT", "")) if os.environ.get("MS_CLOUD_WORK_ROOT") else None
    temp_parent = host_tmp if host_tmp and host_tmp.exists() else None

    with tempfile.TemporaryDirectory(prefix="mausamsaathi-cloud-refresh-", dir=str(temp_parent) if temp_parent else None) as temp_name:
        work_dir = Path(temp_name)

        audit: dict[str, Any] = {
            "started_at": now_iso(),
            "status": "RUNNING",
            "bucket": BUCKET,
            "static_manifest_path": str(MANIFEST_PATH),
            "static_artifact_count": len(manifest["artifacts"]),
            "work_dir": str(work_dir),
        }

        try:
            stage_source_code(work_dir)
            restore = restore_static_artifacts(project_url, key, work_dir, manifest)
            audit["static_restore"] = restore

            legacy_gpm = restore_legacy_raw_gpm_history(
                project_url,
                key,
                work_dir,
            )
            audit["legacy_raw_gpm_restore"] = legacy_gpm

            # For modern sb_secret_* keys, verify access using an object that has
            # just been restored from the verified 806-artifact set. This avoids
            # requiring a separate probe object before the clean workspace is built.
            first_artifact = safe_rel_path(str(manifest["artifacts"][0]["path"]))
            verify_bucket(
                project_url,
                key,
                f"{STATIC_PREFIX}/{first_artifact.as_posix()}",
            )

            prior_runtime = restore_prior_runtime(project_url, key, work_dir)
            audit["prior_runtime_restored"] = prior_runtime is not None

            # The published runtime may predate the persistence SQLite asset.
            # Always make sure the DB exists before the refresh starts so the
            # existing persistence hook can record the successful run and the
            # DB can be included in the next immutable runtime snapshot.
            runtime_db = (
                work_dir
                / "data_test"
                / "processed"
                / "persistence"
                / "mausamsaathi_local.sqlite3"
            )
            if not runtime_db.exists() or runtime_db.stat().st_size <= 0:
                audit["recovery_db_restored"] = restore_recovery_db(
                    project_url, key, work_dir
                )
            else:
                audit["recovery_db_restored"] = False
                audit["recovery_db_already_present"] = True

            if not runtime_db.exists() or runtime_db.stat().st_size <= 0:
                raise RuntimeError(
                    "Required persistence SQLite DB could not be restored: "
                    f"{runtime_db}"
                )

            prepare_cloud_environment(work_dir)

            proc = run_existing_refresh(work_dir)
            audit["refresh_returncode"] = proc.returncode
            audit["refresh_stdout_tail"] = proc.stdout[-12000:]
            audit["refresh_stderr_tail"] = proc.stderr[-12000:]

            if proc.returncode != 0:
                raise RuntimeError(
                    f"Existing refresh pipeline failed with return code {proc.returncode}"
                )

            state_path = locate_current_state(work_dir)
            state = validate_success_state(state_path)
            audit["current_state"] = {
                "path": str(state_path.relative_to(work_dir)),
                "status": state.get("status"),
                "issue_date": state.get("last_successful_issue_date"),
            }

            published = upload_runtime_snapshot(
                project_url,
                key,
                work_dir,
                state,
            )
            audit["published_runtime"] = {
                "run_id": published["run_id"],
                "issue_date": published["issue_date"],
                "file_count": published["file_count"],
            }
            audit["status"] = "PASS"
            audit["finished_at"] = now_iso()

            upload_run_audit(
                project_url,
                key,
                published["run_id"],
                audit,
                work_dir,
            )

            log(
                f"REFRESH STATUS: PASS | issue_date={published['issue_date']} "
                f"| runtime_files={published['file_count']}"
            )
            return 0

        except Exception as exc:
            audit["status"] = "FAILED"
            audit["finished_at"] = now_iso()
            audit["error"] = str(exc)
            audit["traceback_tail"] = traceback.format_exc()[-12000:]

            failure_id = f"failed-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
            try:
                upload_run_audit(project_url, key, failure_id, audit, work_dir)
            except Exception as upload_exc:
                log(f"Could not upload failure audit: {upload_exc}")

            # Crucially: never touch PUBLISHED_RUNTIME_KEY or PUBLISHED_STATE_KEY on failure.
            log(f"REFRESH STATUS: FAILED: {exc}")
            log("Last-good published runtime was left untouched.")
            return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="MausamSaathi cloud daily refresh worker")
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Upload/verify the 806-artifact static runtime set from the local project.",
    )
    args = parser.parse_args()

    if args.bootstrap:
        return bootstrap_static_artifacts()
    return perform_cloud_refresh()


if __name__ == "__main__":
    raise SystemExit(main())
