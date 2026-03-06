-- Invitations table for admin-managed onboarding flow.
create extension if not exists pgcrypto;

create table if not exists public.invitations (
    id uuid primary key default gen_random_uuid(),
    email text not null,
    role text not null default 'Staff',
    token text not null unique,
    invited_by text,
    invited_at timestamptz not null default now(),
    expires_at timestamptz not null,
    used_at timestamptz,
    accepted_user_id uuid,
    created_at timestamptz not null default now(),
    updated_at timestamptz not null default now()
);

create index if not exists invitations_email_idx on public.invitations (email);
create index if not exists invitations_token_idx on public.invitations (token);

create or replace function public.set_invitations_updated_at()
returns trigger
language plpgsql
as $$
begin
    new.updated_at = now();
    return new;
end;
$$;

drop trigger if exists trg_set_invitations_updated_at on public.invitations;
create trigger trg_set_invitations_updated_at
before update on public.invitations
for each row
execute procedure public.set_invitations_updated_at();

alter table public.invitations enable row level security;

-- Admin users should use service-role key from Streamlit backend for insert/update operations.
-- Public users can read only active invite by token to pre-fill signup email.
drop policy if exists "public_read_active_invite_by_token" on public.invitations;
create policy "public_read_active_invite_by_token"
on public.invitations
for select
using (
    used_at is null
    and expires_at > now()
);

-- Authenticated user can mark only their own email invite as used.
drop policy if exists "auth_accept_own_invite" on public.invitations;
create policy "auth_accept_own_invite"
on public.invitations
for update
using (
    auth.role() = 'authenticated'
    and lower(email) = lower(coalesce(auth.jwt() ->> 'email', ''))
)
with check (
    auth.role() = 'authenticated'
    and lower(email) = lower(coalesce(auth.jwt() ->> 'email', ''))
);
