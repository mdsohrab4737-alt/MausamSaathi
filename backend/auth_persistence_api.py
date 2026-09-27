from __future__ import annotations

from typing import Any
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel, Field

try:
    from .auth_service import extract_bearer_token, verify_supabase_access_token
    from .persistence_service import (
        cloud_configured, create_sos, create_user, queue_notification,
        register_device, set_user_location, sync_user_to_cloud,
        upsert_crop_profile, user_context,
    )
except ImportError:
    from auth_service import extract_bearer_token, verify_supabase_access_token
    from persistence_service import (
        cloud_configured, create_sos, create_user, queue_notification,
        register_device, set_user_location, sync_user_to_cloud,
        upsert_crop_profile, user_context,
    )

router = APIRouter(prefix="/persistence/me", tags=["authenticated-persistence"])

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

def _auth_user(authorization: str | None) -> dict[str, Any]:
    try:
        return verify_supabase_access_token(extract_bearer_token(authorization))
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))

def _ensure_local_user(auth_user: dict[str, Any]) -> str:
    metadata = auth_user.get("user_metadata") or {}
    user_id = str(auth_user["id"])
    create_user(
        name=metadata.get("full_name") or metadata.get("name") or auth_user.get("email"),
        phone_e164=auth_user.get("phone") or metadata.get("phone"),
        preferred_language=metadata.get("preferred_language", "hi"),
        external_auth_id=user_id,
    )
    return user_id

@router.get("")
def get_me(authorization: str | None = Header(default=None)):
    auth_user = _auth_user(authorization)
    user_id = _ensure_local_user(auth_user)
    return {
        "auth_user": {
            "id": auth_user["id"],
            "email": auth_user.get("email"),
            "phone": auth_user.get("phone"),
        },
        "local": user_context(user_id),
        "cloud_configured": cloud_configured(),
    }

@router.post("/bootstrap")
def bootstrap(authorization: str | None = Header(default=None)):
    auth_user = _auth_user(authorization)
    user_id = _ensure_local_user(auth_user)
    sync_result: dict[str, Any] | None = None
    if cloud_configured():
        try:
            sync_result = sync_user_to_cloud(user_id)
        except Exception as exc:
            sync_result = {"status": "deferred", "reason": str(exc)}
    return {"status": "PASS", "user_id": user_id, "cloud_sync": sync_result}

@router.post("/devices")
def add_device(payload: DeviceCreate, authorization: str | None = Header(default=None)):
    user_id = _ensure_local_user(_auth_user(authorization))
    try:
        return register_device(user_id=user_id, **payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/locations")
def add_location(payload: LocationCreate, authorization: str | None = Header(default=None)):
    user_id = _ensure_local_user(_auth_user(authorization))
    try:
        return set_user_location(user_id=user_id, **payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/crops")
def add_crop(payload: CropCreate, authorization: str | None = Header(default=None)):
    user_id = _ensure_local_user(_auth_user(authorization))
    try:
        return upsert_crop_profile(user_id=user_id, **payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/notifications")
def add_notification(payload: NotificationCreate, authorization: str | None = Header(default=None)):
    user_id = _ensure_local_user(_auth_user(authorization))
    try:
        return queue_notification(user_id=user_id, **payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/sos")
def add_sos(payload: SosCreate, authorization: str | None = Header(default=None)):
    user_id = _ensure_local_user(_auth_user(authorization))
    try:
        return create_sos(user_id=user_id, **payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))

@router.post("/sync")
def sync(authorization: str | None = Header(default=None)):
    user_id = _ensure_local_user(_auth_user(authorization))
    if not cloud_configured():
        return {"status": "deferred", "reason": "Cloud persistence is not configured."}
    try:
        return sync_user_to_cloud(user_id)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=str(exc))
