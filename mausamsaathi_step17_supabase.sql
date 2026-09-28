-- MAUSAMSAATHI SIH 2026
-- STEP 17 - PERSISTENCE FOUNDATION
-- Cloud schema for a dedicated Supabase/PostgreSQL project.
--
-- IMPORTANT:
-- This migration is additive. It does not import scientific CSV/GRD/NC files.
-- Large scientific/model artifacts remain in object storage and are referenced
-- by data_assets/data_asset_versions.
--
-- Apply only in the dedicated MAUSAMSAATHI Supabase project, not in another app.

create extension if not exists pgcrypto;
create extension if not exists postgis;

create table if not exists public.app_meta (
    key text primary key,
    value jsonb,
    updated_at timestamptz not null default now()
);

create table if not exists public.locations (
    local_body_code text primary key,
    state text not null,
    district text not null,
    block text not null,
    panchayat text not null,
    source_subdistrict text,
    source text not null,
    geom geometry(MultiPolygon, 4326),
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_locations_district_block
    on public.locations(district, block);

create index if not exists idx_locations_geom
    on public.locations using gist(geom);

create table if not exists public.profiles (
    user_id uuid primary key references auth.users(id) on delete cascade,
    name text,
    phone_e164 text,
    preferred_language text not null default 'hi',
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.devices (
    device_id uuid primary key default gen_random_uuid(),
    user_id uuid references public.profiles(user_id) on delete set null,
    platform text,
    app_version text,
    push_token text,
    last_seen_at timestamptz,
    is_active boolean not null default true,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_devices_user
    on public.devices(user_id);

create table if not exists public.user_locations (
    user_id uuid not null references public.profiles(user_id) on delete cascade,
    local_body_code text not null references public.locations(local_body_code) on delete restrict,
    is_primary boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key(user_id, local_body_code)
);

create table if not exists public.crop_profiles (
    crop_profile_id uuid primary key default gen_random_uuid(),
    user_id uuid not null references public.profiles(user_id) on delete cascade,
    local_body_code text not null references public.locations(local_body_code) on delete restrict,
    crop_name text not null,
    growth_stage text,
    area_hectare numeric,
    notes text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_crop_profiles_user
    on public.crop_profiles(user_id, local_body_code);

create table if not exists public.advisories (
    advisory_id uuid primary key default gen_random_uuid(),
    local_body_code text not null references public.locations(local_body_code) on delete restrict,
    crop_name text,
    issue_date date not null,
    horizon_days integer not null check (horizon_days in (7,14,21,30)),
    language text not null default 'hi',
    title text not null,
    summary text not null,
    actions_json jsonb not null default '[]'::jsonb,
    source text not null,
    model_version text,
    created_at timestamptz not null default now()
);

create index if not exists idx_advisories_location_date
    on public.advisories(local_body_code, issue_date desc);

create table if not exists public.notifications (
    notification_id uuid primary key default gen_random_uuid(),
    user_id uuid references public.profiles(user_id) on delete set null,
    local_body_code text references public.locations(local_body_code) on delete set null,
    advisory_id uuid references public.advisories(advisory_id) on delete set null,
    channel text not null check (channel in ('in_app','sms','whatsapp','voice')),
    language text not null default 'hi',
    message text not null,
    status text not null default 'queued',
    scheduled_at timestamptz,
    sent_at timestamptz,
    provider_message_id text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_notifications_user_status
    on public.notifications(user_id, status, created_at desc);

create table if not exists public.notification_deliveries (
    delivery_id uuid primary key default gen_random_uuid(),
    notification_id uuid not null references public.notifications(notification_id) on delete cascade,
    user_id uuid references public.profiles(user_id) on delete set null,
    attempt_no integer not null default 1,
    provider text,
    provider_message_id text,
    status text not null,
    error_message text,
    attempted_at timestamptz not null default now()
);

create index if not exists idx_delivery_notification
    on public.notification_deliveries(notification_id, attempted_at desc);

create table if not exists public.sos_events (
    sos_event_id uuid primary key default gen_random_uuid(),
    user_id uuid references public.profiles(user_id) on delete set null,
    device_id uuid references public.devices(device_id) on delete set null,
    local_body_code text references public.locations(local_body_code) on delete set null,
    latitude double precision,
    longitude double precision,
    location_geom geography(Point, 4326),
    status text not null default 'created',
    triggered_at timestamptz not null default now(),
    resolved_at timestamptz,
    payload_json jsonb
);

create index if not exists idx_sos_status_time
    on public.sos_events(status, triggered_at desc);

create table if not exists public.sync_queue (
    sync_id uuid primary key default gen_random_uuid(),
    user_id uuid references public.profiles(user_id) on delete cascade,
    entity_type text not null,
    entity_id text not null,
    operation text not null check (operation in ('upsert','delete')),
    payload_json jsonb not null,
    status text not null default 'pending',
    attempts integer not null default 0,
    next_attempt_at timestamptz,
    last_error text,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists idx_sync_queue_pending
    on public.sync_queue(status, next_attempt_at, created_at);

create table if not exists public.system_refresh_runs (
    refresh_run_id uuid primary key default gen_random_uuid(),
    started_at timestamptz not null,
    finished_at timestamptz,
    issue_date date,
    status text not null,
    pilot_district text,
    local_body_count integer,
    model_source text,
    manifest_path text,
    error text,
    metadata_json jsonb
);

create index if not exists idx_refresh_runs_time
    on public.system_refresh_runs(started_at desc);

create table if not exists public.data_assets (
    asset_id uuid primary key default gen_random_uuid(),
    logical_key text not null unique,
    asset_kind text not null,
    dataset_name text not null,
    source_name text,
    is_critical boolean not null default false,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create table if not exists public.data_asset_versions (
    asset_version_id uuid primary key default gen_random_uuid(),
    asset_id uuid not null references public.data_assets(asset_id) on delete cascade,
    version_label text not null,
    issue_date date,
    local_path text,
    object_bucket text,
    object_key text,
    sha256 text,
    size_bytes bigint,
    status text not null default 'discovered',
    verified_at timestamptz,
    metadata_json jsonb,
    created_at timestamptz not null default now(),
    unique(asset_id, version_label)
);

create index if not exists idx_asset_versions_lookup
    on public.data_asset_versions(asset_id, issue_date desc);

create index if not exists idx_asset_versions_sha
    on public.data_asset_versions(sha256);

-- RLS: public geography can be read by signed-in clients;
-- user-owned data is isolated by auth.uid().
alter table public.locations enable row level security;
alter table public.profiles enable row level security;
alter table public.devices enable row level security;
alter table public.user_locations enable row level security;
alter table public.crop_profiles enable row level security;
alter table public.advisories enable row level security;
alter table public.notifications enable row level security;
alter table public.notification_deliveries enable row level security;
alter table public.sos_events enable row level security;
alter table public.sync_queue enable row level security;
alter table public.system_refresh_runs enable row level security;
alter table public.data_assets enable row level security;
alter table public.data_asset_versions enable row level security;
alter table public.app_meta enable row level security;

drop policy if exists locations_select_authenticated on public.locations;
create policy locations_select_authenticated
    on public.locations for select
    to authenticated
    using (true);

drop policy if exists profiles_self_all on public.profiles;
create policy profiles_self_all
    on public.profiles for all
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists devices_self_all on public.devices;
create policy devices_self_all
    on public.devices for all
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists user_locations_self_all on public.user_locations;
create policy user_locations_self_all
    on public.user_locations for all
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists crop_profiles_self_all on public.crop_profiles;
create policy crop_profiles_self_all
    on public.crop_profiles for all
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists advisories_select_authenticated on public.advisories;
create policy advisories_select_authenticated
    on public.advisories for select
    to authenticated
    using (true);

drop policy if exists notifications_self_select on public.notifications;
create policy notifications_self_select
    on public.notifications for select
    to authenticated
    using (auth.uid() = user_id);

drop policy if exists notification_deliveries_self_select on public.notification_deliveries;
create policy notification_deliveries_self_select
    on public.notification_deliveries for select
    to authenticated
    using (auth.uid() = user_id);

drop policy if exists sos_events_self_all on public.sos_events;
create policy sos_events_self_all
    on public.sos_events for all
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists sync_queue_self_all on public.sync_queue;
create policy sync_queue_self_all
    on public.sync_queue for all
    to authenticated
    using (auth.uid() = user_id)
    with check (auth.uid() = user_id);

drop policy if exists refresh_runs_select_authenticated on public.system_refresh_runs;
create policy refresh_runs_select_authenticated
    on public.system_refresh_runs for select
    to authenticated
    using (true);

drop policy if exists data_assets_select_authenticated on public.data_assets;
create policy data_assets_select_authenticated
    on public.data_assets for select
    to authenticated
    using (true);

drop policy if exists data_asset_versions_select_authenticated on public.data_asset_versions;
create policy data_asset_versions_select_authenticated
    on public.data_asset_versions for select
    to authenticated
    using (true);

-- app_meta is intentionally not exposed to clients by default.
