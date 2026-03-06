import secrets
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import smtplib
from typing import Optional

import streamlit as st
from supabase import Client, create_client


def _secret(name: str, default: Optional[str] = None) -> Optional[str]:
    if name in st.secrets:
        return st.secrets[name]
    return default


def get_supabase_client(admin: bool = False) -> Client:
    supabase_url = _secret("SUPABASE_URL")
    if not supabase_url:
        raise RuntimeError("SUPABASE_URL is missing from Streamlit secrets.")

    key_name = "SUPABASE_SERVICE_ROLE_KEY" if admin else "SUPABASE_ANON_KEY"
    fallback_name = "SUPABASE_ANON_KEY"

    supabase_key = _secret(key_name) or _secret(fallback_name)
    if not supabase_key:
        raise RuntimeError("Supabase key is missing from Streamlit secrets.")

    return create_client(supabase_url, supabase_key)


def generate_invite_token() -> str:
    return secrets.token_urlsafe(32)


def create_invitation(email: str, role: str, invited_by: str, expires_in_days: int = 7) -> dict:
    supabase = get_supabase_client(admin=True)
    token = generate_invite_token()
    expires_at = (datetime.now(timezone.utc) + timedelta(days=expires_in_days)).isoformat()

    payload = {
        "email": email.strip().lower(),
        "role": role,
        "token": token,
        "invited_by": invited_by,
        "expires_at": expires_at,
        "used": False,
    }
    res = supabase.table("invitations").insert(payload).execute()
    if not res.data:
        raise RuntimeError("Failed to create invitation record.")
    return res.data[0]


def get_active_invitation(token: str) -> Optional[dict]:
    supabase = get_supabase_client(admin=False)
    res = (
        supabase.table("invitations")
        .select("id,email,role,token,expires_at,used_at")
        .eq("token", token)
        .eq("used", False)
        .is_("used_at", "null")
        .gt("expires_at", datetime.now(timezone.utc).isoformat())
        .limit(1)
        .execute()
    )
    return res.data[0] if res.data else None


def mark_invitation_used(token: str, accepted_user_id: str, user_email: str) -> None:
    supabase = get_supabase_client(admin=True)
    updates = {
        "used": True,
        "used_at": datetime.now(timezone.utc).isoformat(),
        "accepted_user_id": accepted_user_id,
    }
    (
        supabase.table("invitations")
        .update(updates)
        .eq("token", token)
        .eq("email", user_email.strip().lower())
        .is_("used_at", "null")
        .execute()
    )


def send_invite_email(to_email: str, invite_link: str) -> None:
    smtp_host = _secret("SMTP_HOST")
    smtp_port = int(_secret("SMTP_PORT", "587"))
    smtp_user = _secret("SMTP_USER")
    smtp_password = _secret("SMTP_PASSWORD")
    from_email = _secret("SMTP_FROM", smtp_user)

    if not all([smtp_host, smtp_user, smtp_password, from_email]):
        raise RuntimeError("SMTP settings are missing (SMTP_HOST/SMTP_USER/SMTP_PASSWORD/SMTP_FROM).")

    msg = EmailMessage()
    msg["Subject"] = "You're invited to ReCapture Pro"
    msg["From"] = from_email
    msg["To"] = to_email
    msg.set_content(
        "You've been invited to ReCapture Pro. "
        f"Accept your invite here: {invite_link}\n\n"
        "If you didn't expect this invite, please ignore this email."
    )

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.send_message(msg)
