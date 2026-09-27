from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

try:
    from .persistence_service import (
        cloud_configured,
        create_sos,
        create_user,
        initialize,
        queue_notification,
        register_device,
        set_user_location,
        status,
        sync_user_to_cloud,
        upsert_crop_profile,
        user_context,
    )
except ImportError:
    from persistence_service import (
        cloud_configured,
        create_sos,
        create_user,
        initialize,
        queue_notification,
        register_device,
        set_user_location,
        status,
        sync_user_to_cloud,
        upsert_crop_profile,
        user_context,
    )


router = APIRouter(tags=["persistence"])


class UserCreate(BaseModel):
    name: str | None = None
    phone_e164: str | None = None
    preferred_language: str = "hi"
    external_auth_id: str | None = None


class DeviceCreate(BaseModel):
    platform: str | None = None
    app_version: str | None = None
    push_token: str | None = None
    device_id: str | None = None


class LocationCreate(BaseModel):
    local_body_code: str
    is_primary: bool = True


class CropCreate(BaseModel):
    local_body_code: str
    crop_name: str
    growth_stage: str | None = None
    area_hectare: float | None = Field(default=None, ge=0)
    notes: str | None = None
    crop_profile_id: str | None = None


class NotificationCreate(BaseModel):
    channel: str
    message: str
    local_body_code: str | None = None
    advisory_id: str | None = None
    language: str = "hi"
    scheduled_at: str | None = None


class SosCreate(BaseModel):
    device_id: str | None = None
    local_body_code: str | None = None
    latitude: float | None = None
    longitude: float | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


def _internal_key_ok(header: str | None) -> bool:
    # Local development stays usable when the internal key is not configured.
    # Production should set MAUSAMSAATHI_INTERNAL_API_KEY before exposing writes.
    import os
    expected = os.getenv("MAUSAMSAATHI_INTERNAL_API_KEY", "").strip()
    if not expected:
        return True
    return bool(header) and header == expected


initialize()


@router.get("/persistence/status")
def persistence_status():
    return status()


@router.post("/persistence/users")
def persistence_create_user(
    payload: UserCreate,
    x_persistence_key: str | None = Header(default=None),
):
    if not _internal_key_ok(x_persistence_key):
        raise HTTPException(status_code=401, detail="Unauthorized persistence request")
    try:
        return create_user(**payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.get("/persistence/users/{user_id}")
def persistence_user_context(user_id: str):
    try:
        return user_context(user_id)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=str(exc))


@router.post("/persistence/users/{user_id}/devices")
def persistence_register_device(
    user_id: str,
    payload: DeviceCreate,
    x_persistence_key: str | None = Header(default=None),
):
    if not _internal_key_ok(x_persistence_key):
        raise HTTPException(status_code=401, detail="Unauthorized persistence request")
    try:
        return register_device(user_id=user_id, **payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/persistence/users/{user_id}/locations")
def persistence_set_location(
    user_id: str,
    payload: LocationCreate,
    x_persistence_key: str | None = Header(default=None),
):
    if not _internal_key_ok(x_persistence_key):
        raise HTTPException(status_code=401, detail="Unauthorized persistence request")
    try:
        return set_user_location(user_id=user_id, **payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/persistence/users/{user_id}/crops")
def persistence_upsert_crop(
    user_id: str,
    payload: CropCreate,
    x_persistence_key: str | None = Header(default=None),
):
    if not _internal_key_ok(x_persistence_key):
        raise HTTPException(status_code=401, detail="Unauthorized persistence request")
    try:
        return upsert_crop_profile(user_id=user_id, **payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/persistence/users/{user_id}/notifications")
def persistence_queue_notification(
    user_id: str,
    payload: NotificationCreate,
    x_persistence_key: str | None = Header(default=None),
):
    if not _internal_key_ok(x_persistence_key):
        raise HTTPException(status_code=401, detail="Unauthorized persistence request")
    try:
        return queue_notification(user_id=user_id, **payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/persistence/sos")
def persistence_create_sos(
    user_id: str | None = None,
    payload: SosCreate | None = None,
    x_persistence_key: str | None = Header(default=None),
):
    if not _internal_key_ok(x_persistence_key):
        raise HTTPException(status_code=401, detail="Unauthorized persistence request")
    payload = payload or SosCreate()
    try:
        return create_sos(
            user_id=user_id,
            **payload.model_dump(),
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@router.post("/persistence/users/{user_id}/sync")
def persistence_sync_user(
    user_id: str,
    x_persistence_key: str | None = Header(default=None),
):
    if not _internal_key_ok(x_persistence_key):
        raise HTTPException(status_code=401, detail="Unauthorized persistence request")
    if not cloud_configured():
        return {
            "status": "deferred",
            "reason": "SUPABASE_URL and SUPABASE_SECRET_KEY are not configured.",
        }
    try:
        return sync_user_to_cloud(user_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
