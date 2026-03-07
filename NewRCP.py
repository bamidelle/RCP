# ReCapture Pro
"""
ReCapture Pro - Single-file Streamlit app
- No front-page login: this is an admin backend (Admin access by default)
- User & Role management available in Settings
- SQLite persistence via SQLAlchemy —- Now Migrating SupaBase DB
- Internal ML training & scoring (no user tuning)
- Pipeline dashboard, Analytics, CPA/ROI, Exports/Imports, Alerts, SLA, Priority scoring, Audit
trail
"""
import os
from datetime import datetime, timedelta, date
import io, base64, traceback
import streamlit as st
import folium
from streamlit_folium import st_folium
import pandas as pd
import numpy as np
import plotly.express as px
import qrcode
from datetime import datetime
import streamlit as st
import joblib
from sqlalchemy import or_
import secrets
import uuid
import smtplib
from email.message import EmailMessage
import re
from uuid import uuid4
from datetime import datetime
import json
import jwt
from datetime import datetime, timedelta
import secrets
import resend
import smtplib
from email.message import EmailMessage
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from services.org_service import create_organization, get_organizations
#--------------------supabase---------------
from supabase import create_client
import os
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_ANON_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
#--------------------ends here-----------------------
from sqlalchemy import (
create_engine,
Column,
Integer,
String,
Float,
Boolean,
DateTime,
Text,
ForeignKey,
inspect,
text
)
from sqlalchemy.orm import declarative_base, sessionmaker, relationship
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sqlalchemy import inspect
import uuid
from sqlalchemy import ForeignKey
from sqlalchemy.orm import relationship
from passlib.context import CryptContext
import json
from datetime import datetime
from uuid import uuid4
from pathlib import Path
EVENT_LOG_FILE = Path("platform_events.log")
import uuid
from datetime import datetime

def is_valid_email(email: str) -> bool:
    if not email:
        return False
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email.strip()))

def generate_review_token():
    return f"rvw_{uuid.uuid4().hex[:10]}"
st.markdown("""
<style>
/* Page spacing */
.block-container {
padding-top: 1.5rem;
}
/* KPI cards */
.kpi-card {
background: white;
border-radius: 12px;
padding: 18px;
border-left: 5px solid #eee;
box-shadow: 0 4px 14px rgba(0,0,0,0.04);
}
.kpi-title {
font-size: 0.85rem;
color: #777;
}
.kpi-value {
font-size: 1.6rem;
font-weight: 700;
margin-top: 6px;
}
.kpi-sub {
font-size: 0.75rem;
color: #999;
}
/* Colored borders */
.kpi-purple { border-left-color: #8b5cf6; }
.kpi-blue { border-left-color: #3b82f6; }
.kpi-yellow { border-left-color: #f59e0b; }
.kpi-green { border-left-color: #22c55e; }
.kpi-red { border-left-color: #ef4444; }
/* Attention list */
.attention-item {
padding: 14px;
border-left: 4px solid;
margin-bottom: 12px;
background: #fff;
border-radius: 10px;
}
.attention-red { border-color: #ef4444; }
.attention-yellow { border-color: #f59e0b; }
.attention-blue { border-color: #3b82f6; }
.attention-green { border-color: #22c55e; }
/* Timeline */
.timeline-item {
padding: 10px 0;
border-bottom: 1px solid #eee;
font-size: 0.9rem;
}
/* Section headers */
.section-title {
font-weight: 600;
color: #555;
margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)
# -----------------------------
# DEV MODE: Unlock all features
# -----------------------------
DEV_MODE = True # Set False in production
# ----------------------
# APP CONFIG
# ----------------------
FRONTEND_URL = (
st.secrets.get("FRONTEND_URL")
if "FRONTEND_URL" in st.secrets
else "http://localhost:8501"
)
STAGE_MAX_DAYS = {
"New": 1,
"Contacted": 2,
"Inspection Scheduled": 3,
"Inspection": 5,
"Estimate Sent": 7,
}
# ======================
# BILLING PROVIDER (DEV)
# ======================
class DummyBillingProvider:
    def charge(self, user, amount):
        if not user:
            raise ValueError("Billing failed: user is None")
        email = getattr(user, "email", None) or getattr(user, "username", "unknown-user")
        print(f"[BILLING] Simulated charge: {email} → ${amount}")
        return True
# MUST EXIST BEFORE apply_plan_change IS DEFINED
BILLING_PROVIDER = DummyBillingProvider()
def apply_plan_change(user, new_plan, amount):
    if not user:
        raise ValueError("Cannot apply plan: user is None")
    BILLING_PROVIDER.charge(user, amount)
    user.plan = new_plan
    user.subscription_status = "active"
# ---------- Country list helper (robust) ----------
import requests
from functools import lru_cache
@lru_cache(maxsize=1)
def get_all_countries():
    """
Robust fetch of all countries. Uses restcountries.com.
Returns list of dicts: [{ 'name': 'United States', 'code': 'US' }, ...]
Falls back to a small built-in list if the request fails.
"""
    FALLBACK = [
        {"name": "United States", "code": "US"},
        {"name": "Canada", "code": "CA"},
        {"name": "United Kingdom", "code": "GB"},
        {"name": "Australia", "code": "AU"},
    ]
    url = "https://restcountries.com/v3.1/all"
    try:
        r = requests.get(url, timeout=8)
        r.raise_for_status()
        data = r.json()
        out = []
        for c in data:
            # ensure structure exists
            name = c.get("name", {}).get("common")
            code = c.get("cca2") or c.get("cca3") or None
            if name and code:
                out.append({"name": name, "code": code})

        if not out:
            # unexpected schema
            print("get_all_countries: empty result, using fallback")
            return FALLBACK

        # sort by name
        out = sorted(out, key=lambda x: x["name"])
        return out
    except Exception as e:
        # print to stdout so Streamlit logs show it
        print("get_all_countries() ERROR:", repr(e))
        # return fallback so UI still works
        return FALLBACK
# ---------- end helper ----------
# ---------- City search helper (GLOBAL) ----------
def search_cities(country_code, city_name, limit=10):
    """
Uses Open-Meteo geocoding to search cities globally by country.
Returns: [{name, admin1, lat, lon}, ...]
    """
    if not city_name:
        return []
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={
                "name": city_name,
                "count": limit,
                "language": "en",
                "format": "json",
                "country": country_code,
            },
            timeout=8,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        return [
            {
                "name": x.get("name"),
                "admin1": x.get("admin1"),
                "lat": x.get("latitude"),
                "lon": x.get("longitude"),
            }
            for x in results
            if x.get("latitude") and x.get("longitude")
        ]
    except Exception as e:
        print("search_cities ERROR:", repr(e))
        return []
# ---------- end helper ----------
# ---------- Weather helpers (Open-Meteo) ----------
@lru_cache(maxsize=128)
def fetch_weather(lat, lon, months):
    """
Historical daily weather for the past N months
    """
    end = date.today()
    start = end - timedelta(days=months * 30)
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude": lat,
            "longitude": lon,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "daily": "precipitation_sum,temperature_2m_mean",
            "timezone": "UTC",
        },
        timeout=10,
    )
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(d["time"]),
            "rainfall_mm": d["precipitation_sum"],
            "temperature_c": d["temperature_2m_mean"],
        }
    )
    return df.dropna().reset_index(drop=True)
@lru_cache(maxsize=128)
def fetch_forecast_weather(lat, lon, days):
    """
Short-term daily forecast (Open-Meteo limit ~14 days)
    """
    days = min(days, 14) # API limit
    r = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": lat,
            "longitude": lon,
            "daily": "precipitation_sum,temperature_2m_mean",
            "forecast_days": days,
            "timezone": "UTC",
        },
        timeout=10,
    )
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(d["time"]),
            "rainfall_mm": d["precipitation_sum"],
            "temperature_c": d["temperature_2m_mean"],
        }
    )
    return df.dropna().reset_index(drop=True)
# ---------- end weather helpers ----------
st.markdown("""
<style>
.kpi-card {
background-color: #000000;
padding: 18px 20px;
border-radius: 14px;
box-shadow: 0 4px 14px rgba(0,0,0,0.35);
min-height: 110px;
}
.kpi-label {
color: #9ca3af;
font-size: 0.85rem;
margin-bottom: 6px;
}
.kpi-value {
font-size: 1.8rem;
font-weight: 700;
}
.kpi-caption {
color: #d1d5db;
font-size: 0.85rem;
margin-top: 6px;
}
.blue { color: #3b82f6; }
.green { color: #22c55e; }
.orange { color: #f97316; }
.red { color: #ef4444; }
.cyan { color: #06b6d4; }
</style>
""", unsafe_allow_html=True)
# =============================================================
# =========================================================
# PRICING PLANS & FEATURE ACCESS
# =========================================================
PLANS = {
"starter": {
    "analytics": False,
    "business_intelligence": False,
    "max_leads_per_month": 50,
    "max_users": 1,
},
"pro": {
    "analytics": True,
    "business_intelligence": True,
    "max_leads_per_month": 300,
    "max_users": 5,
},
"business": {
    "analytics": True,
    "business_intelligence": True,
    "max_leads_per_month": 2000,
    "max_users": 10,
},
"enterprise": {"max_leads_per_month": None},
}
PLAN_LIMITS = {
"trial": {
    "max_users": 3,
    "max_leads": 50,
    "ai_requests_per_day": 20,
    "exports": False,
},
"starter": {
    "max_users": 5,
    "max_leads": 300,
    "ai_requests_per_day": 100,
    "exports": True,
},
"pro": {
    "max_users": 20,
    "max_leads": 3000,
    "ai_requests_per_day": 500,
    "exports": True,
},
"enterprise": {
    "max_users": 9999,
    "max_leads": 999999,
    "ai_requests_per_day": 999999,
    "exports": True,
},
}
# ----------------------
# CONFIG
# ----------------------
APP_TITLE = "ReCapture Pro"
DB_FILE = "titan_backend.db" # stored in app working directory
MODEL_FILE = "titan_model.joblib"
PIPELINE_STAGES = [
"New", "Contacted", "Inspection Scheduled", "Inspection Completed",
"Estimate Sent", "Qualified", "Won", "Lost"
]
DEFAULT_SLA_HOURS = 24

FONT_URL = "https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap"
st.markdown(f"<style>@import url('{FONT_URL}'); html, body {{ font-family: 'Comfortaa', sans-serif; }}</style>", unsafe_allow_html=True)
# KPI colors (numbers)
KPI_COLORS = ["#2563eb", "#0ea5a4", "#a855f7", "#f97316", "#ef4444", "#6d28d9",
"#22c55e"]
#-----------------------
# STRIPE DUMMY TEST
#-----------------------
STRIPE_ENABLED = False # turn ON later
STRIPE_PLANS = {
"starter": None,
"pro": "price_test_pro",
"business": "price_test_business",
}
# ----------------------
# DB SETUP
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, DB_FILE)
ENGINE_URL = f"sqlite:///{DB_PATH}"
engine = create_engine(
ENGINE_URL,
connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()
# ----------------------
# MODELS
# ----------------------
class Organization(Base):
    __tablename__ = "organizations"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    plan = Column(String, default="trial")
    max_users = Column(Integer, default=1)
    created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True)
    organization_id = Column(Integer, ForeignKey("organizations.id"))
    organization = relationship("Organization")

    email = Column(String, unique=True, nullable=False, index=True)
    email_verified = Column(Boolean, default=False)
    username = Column(String, unique=True, nullable=True)
    full_name = Column(String, default="")
    role = Column(String, default="Viewer")
    created_at = Column(DateTime, default=datetime.utcnow)
    activated_at = Column(DateTime, nullable=True)

    plan = Column(String, default="starter")
    subscription_status = Column(String, default="trial")
    trial_ends_at = Column(DateTime, nullable=True)

    is_active = Column(Boolean, default=True)
    last_login_at = Column(DateTime, nullable=True)

    password_hash = Column(String, nullable=True)
    reset_token = Column(String, nullable=True)
    reset_expires_at = Column(DateTime, nullable=True)
    activation_token = Column(String, nullable=True)
    activation_expires_at = Column(DateTime, nullable=True)

    otp_code = Column(String, nullable=True)
    otp_expires_at = Column(DateTime, nullable=True)
    otp_required = Column(Boolean, default=False)

    failed_login_attempts = Column(Integer, default=0)
    locked_until = Column(DateTime, nullable=True)


JWT_SECRET = os.environ.get("JWT_SECRET", "CHANGE_ME_NOW")
JWT_ALGO = "HS256"
JWT_EXP_MINUTES = 15


class Invoice(Base):
    __tablename__ = "invoices"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    amount = Column(Float)
    currency = Column(String, default="USD")
    status = Column(String)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class UserInvite(Base):
    __tablename__ = "user_invites"

    id = Column(Integer, primary_key=True)
    email = Column(String, nullable=False, index=True)
    token = Column(String, unique=True, nullable=False)
    role = Column(String, default="Staff")
    invited_by = Column(String, nullable=True)
    expires_at = Column(DateTime, nullable=False)
    accepted = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class LoginToken(Base):
    __tablename__ = "login_tokens"

    id = Column(Integer, primary_key=True)
    token = Column(String, unique=True, nullable=False, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    used = Column(Boolean, default=False)
    user = relationship("User")


class Lead(Base):
    __tablename__ = "leads"

    id = Column(Integer, primary_key=True)
    lead_id = Column(String, unique=True, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    source = Column(String, default="Other")
    source_details = Column(String, nullable=True)
    contact_name = Column(String, nullable=True)
    contact_phone = Column(String, nullable=True)
    contact_email = Column(String, nullable=True)
    property_address = Column(String, nullable=True)
    damage_type = Column(String, nullable=True)
    assigned_to = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    estimated_value = Column(Float, default=0.0)
    stage = Column(String, default="New")
    sla_hours = Column(Integer, default=DEFAULT_SLA_HOURS)
    sla_entered_at = Column(DateTime, default=datetime.utcnow)
    contacted = Column(Boolean, default=False)
    inspection_scheduled = Column(Boolean, default=False)
    inspection_scheduled_at = Column(DateTime, nullable=True)
    inspection_completed = Column(Boolean, default=False)
    estimate_submitted = Column(Boolean, default=False)
    estimate_submitted_at = Column(DateTime, nullable=True)
    awarded_date = Column(DateTime, nullable=True)
    awarded_invoice = Column(String, nullable=True)
    lost_date = Column(DateTime, nullable=True)
    qualified = Column(Boolean, default=False)
    ad_cost = Column(Float, default=0.0)
    converted = Column(Boolean, default=False)
    score = Column(Float, nullable=True)


class LeadHistory(Base):
    __tablename__ = "lead_history"

    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=False)
    changed_by = Column(String, nullable=True)
    field = Column(String, nullable=True)
    old_value = Column(String, nullable=True)
    new_value = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)


from sqlalchemy import DateTime as SA_DateTime


class Technician(Base):
    __tablename__ = "technicians"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String, default="")
    phone = Column(String, nullable=True)
    specialization = Column(String, nullable=True)
    status = Column(String, default="available")
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class InspectionAssignment(Base):
    __tablename__ = "inspection_assignments"

    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=False)
    technician_username = Column(String, nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="assigned")
    notes = Column(Text, nullable=True)


class LocationPing(Base):
    __tablename__ = "location_pings"

    id = Column(Integer, primary_key=True)
    tech_username = Column(String, nullable=False)
    lead_id = Column(String, nullable=True)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float, nullable=True)


class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=True)
    technician_username = Column(String, nullable=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String, default="open")
    due_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class Competitor(Base):
    __tablename__ = "competitors"

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    place_id = Column(String, unique=True, nullable=True)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    rating = Column(Float, default=0.0)
    total_reviews = Column(Integer, default=0)
    primary_category = Column(String, nullable=True)
    service_area = Column(String, nullable=True)
    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class CompetitorSnapshot(Base):
    __tablename__ = "competitor_snapshots"

    id = Column(Integer, primary_key=True)
    competitor_id = Column(Integer, ForeignKey("competitors.id"))
    rating = Column(Float)
    total_reviews = Column(Integer)
    captured_at = Column(DateTime, default=datetime.utcnow)
    competitor = relationship("Competitor")


class CompetitorAlert(Base):
    __tablename__ = "competitor_alerts"

    id = Column(Integer, primary_key=True)
    competitor_id = Column(Integer)
    alert_type = Column(String)
    message = Column(String)
    severity = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)


class ReviewSettings(Base):
    __tablename__ = "review_settings"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False, unique=True)
    review_link = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class ReviewEmailTemplate(Base):
    __tablename__ = "review_email_templates"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, nullable=False)
    subject = Column(String, nullable=True)
    body = Column(Text, nullable=True)
    footer = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class AIInsight(Base):
    __tablename__ = "ai_insights"

    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    insight_key = Column(String, index=True)
    message = Column(Text)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)

# ---------- END BLOCK A ----------
# Create tables if missing
from sqlalchemy import inspect
inspector = inspect(engine)
def safe_create_tables():
    """Best-effort table creation that won't crash startup on partial schema drift."""
    try:
        Base.metadata.create_all(bind=engine, checkfirst=True)
        return
    except Exception as e:
        print("safe_create_tables: bulk create_all failed, falling back:", e)

    inspector = inspect(engine)
    existing_tables = set(inspector.get_table_names())
    for table in Base.metadata.sorted_tables:
        if table.name in existing_tables:
            continue
        try:
            table.create(bind=engine, checkfirst=True)
        except Exception as inner_e:
            print(f"safe_create_tables: skipped table '{table.name}':", inner_e)

safe_create_tables()

# Safe migration attempt (best-effort add missing columns)
def safe_migrate():
    try:
        inspector = inspect(engine)
        if "users" in inspector.get_table_names():
            existing = [c["name"] for c in inspector.get_columns("users")]
            desired = {
                "plan": "TEXT",
                "trial_ends_at": "DATETIME",
                "subscription_status": "TEXT",
            }
            with engine.begin() as conn:
                # ---- Subscription / Plan Fields ----
                for col, typ in desired.items():
                    if col not in existing:
                        conn.execute(text(f"ALTER TABLE users ADD COLUMN {col} {typ}"))

                # ---- User security fields ----
                if "failed_login_attempts" not in existing:
                    conn.execute(text(
                        "ALTER TABLE users ADD COLUMN failed_login_attempts INTEGER DEFAULT 0"
                    ))
                if "locked_until" not in existing:
                    conn.execute(text("ALTER TABLE users ADD COLUMN locked_until DATETIME"))
    except Exception as e:
        print(" User migration skipped:", e)

safe_migrate()

def create_login_token(user, minutes=15):
    token = secrets.token_urlsafe(32)
    login_token = LoginToken(
        token=token,
        user_id=user.id,
        expires_at=pd.Timestamp.utcnow() + timedelta(minutes=minutes),
    )
    with SessionLocal() as s:
        s.add(login_token)
        s.commit()
    return token

from datetime import datetime
def verify_login_token(token: str):
    with SessionLocal() as s:
        login_token = (
    s.query(LoginToken)
    .filter(
        LoginToken.token == token,
        LoginToken.used == False,
        LoginToken.expires_at > pd.Timestamp.utcnow(),
    )
    .first()
    )
    if not login_token:
        return None
    login_token.used = True
    login_token.user.last_login_at = pd.Timestamp.utcnow()
    s.commit()
    return login_token.user
def login_user(user):
    st.session_state["user_id"] = user.id
    st.session_state["user_role"] = user.role
    st.session_state["user_email"] = user.email


def logout_user():
    st.session_state.clear()
from sqlalchemy import inspect, text
from sqlalchemy import inspect, text
def safe_migrate_new_tables():
    try:
        inspector = inspect(engine)
        # Ensure tables exist
        Base.metadata.create_all(engine)
        with engine.begin() as conn:
        # ==========================
        # ---- USERS TABLE ----
        # ==========================
            if "users" in inspector.get_table_names():
                cols = [c["name"] for c in inspector.get_columns("users")]
        if "email" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN email VARCHAR")
        )
        if "email_verified" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN email_verified BOOLEAN DEFAULT0")
        )
        if "is_active" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN is_active BOOLEAN DEFAULT 1")
        )
        if "last_login_at" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN last_login_at DATETIME")
        )
        # ---- INVITES / ACTIVATION ----
        if "invite_token_hash" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN invite_token_hash VARCHAR")
        )
        if "invite_expires_at" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN invite_expires_at DATETIME")
        )
        if "activated_at" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN activated_at DATETIME")
        )
        if "activation_token" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN activation_token VARCHAR")
        )
        if "activation_expires_at" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN activation_expires_at DATETIME")
        )
        # ---- PASSWORD / SECURITY ----
        if "password_hash" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN password_hash VARCHAR")
        )
        if "password_reset_token" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN password_reset_token VARCHAR")
        )
        if "password_reset_expires_at" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN password_reset_expires_atDATETIME")
        )
        if "reset_token" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN reset_token VARCHAR")
        )
        if "reset_expires_at" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN reset_expires_at DATETIME")
        )
        if "otp_code" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN otp_code VARCHAR")
        )
        if "otp_expires_at" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN otp_expires_at DATETIME")
        )
        if "otp_required" not in cols:
            conn.execute(
        text("ALTER TABLE users ADD COLUMN otp_required BOOLEAN DEFAULT 0")
        )
        # BACKFILL EMAIL IF MISSING
        conn.execute(
        text("""
        UPDATE users
        SET email = username
        WHERE email IS NULL OR email = ''
        """)
        )
        # ==========================
        # ---- TECHNICIANS TABLE ----
        # ==========================
        if "technicians" in inspector.get_table_names():
            cols = [c["name"] for c in inspector.get_columns("technicians")]
        if "status" not in cols:
            conn.execute(
        text(
            "ALTER TABLE technicians "
            "ADD COLUMN status VARCHAR DEFAULT 'available'"
        )
        )
    except Exception as e:
        print("Safe migration skipped:", e)
# Call migration
safe_migrate_new_tables()
def haversine_km(lat1, lon1, lat2, lon2):
    """
Calculate distance between two lat/lon points in KM
"""
    R = 6371
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 +         math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))

# ---------- BEGIN BLOCK D: COMPETITOR HELPERS ----------
@st.cache_data(ttl=86400)
def calculate_competitor_score(rating, reviews, distance_km):
    if distance_km <= 0:
        distance_km = 1
    return round((rating * reviews) / distance_km, 2)

def save_competitor_snapshot(competitor_id, rating, total_reviews):
    s = get_session()
    try:
        snap = CompetitorSnapshot(
            competitor_id=competitor_id,
            rating=rating,
            total_reviews=total_reviews,
        )
        s.add(snap)
        s.commit()
    except Exception:
        s.rollback()
    finally:
        s.close()

def seo_visibility_gap(you_reviews, you_rating, competitors_df):
    avg_comp_reviews = competitors_df["Reviews"].mean()
    avg_comp_rating = competitors_df["Rating"].mean()
    review_gap = avg_comp_reviews - you_reviews
    rating_gap = avg_comp_rating - you_rating
    return {
    "review_gap": round(review_gap, 1),
    "rating_gap": round(rating_gap, 2),
    "pressure": "HIGH" if review_gap > 20 or rating_gap > 0.3 else "MODERATE"
}
# ---------- END BLOCK D ----------
# ---------- BEGIN BLOCK F: GOOGLE PLACES INGESTION ----------
import requests
from datetime import datetime
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
import requests
def ingest_competitors_openstreetmap(lat, lon, keyword, radius=5000):
    """
Fetch competitors from OpenStreetMap around a point using Overpass API.
"""
    overpass_url = "https://overpass-api.de/api/interpreter"
    keyword = keyword.lower().replace(" ", "_")
    query = f"""
[out:json][timeout:25];
node
    ["name"]
    ["amenity"]
    (around:{radius},{lat},{lon});
out center;
"""
    try:
        response = requests.get(overpass_url, params={"data": query})
        response.raise_for_status()
        data = response.json()
        elements = data.get("elements", [])
        if not elements:
            st.warning("No competitors found in this area.")
            return

        s = get_session()
        try:
            for e in elements:
                name = e.get("tags", {}).get("name")
                category = e.get("tags", {}).get("amenity", "unknown")
                lat_ = e.get("lat") or e.get("center", {}).get("lat")
                lon_ = e.get("lon") or e.get("center", {}).get("lon")

                if not (name and lat_ and lon_):
                    continue

                exists = s.query(Competitor).filter_by(name=name).first()
                if not exists:
                    comp = Competitor(
                        name=name,
                        primary_category=category,
                        latitude=lat_,
                        longitude=lon_,
                        total_reviews=0,
                        rating=0.0,
                    )
                    s.add(comp)
            s.commit()
        finally:
            s.close()
    except Exception as e:
        raise RuntimeError(f"OSM competitor scan failed: {e}")


def review_velocity(competitor_id, days):
    s = get_session()
    try:
        since = pd.Timestamp.utcnow() - timedelta(days=days)
        count = (
            s.query(CompetitorSnapshot)
            .filter(
                CompetitorSnapshot.competitor_id == competitor_id,
                CompetitorSnapshot.captured_at >= since,
            )
            .count()
        )
        return round(count / max(days, 1), 2)
    finally:
        s.close()


def generate_competitor_alerts():
    s = get_session()
    try:
        competitors = s.query(Competitor).all()
        for c in competitors:
            v7 = review_velocity(c.id, 7)
            v30 = review_velocity(c.id, 30)
            if v7 >= 10:
                s.add(
                    CompetitorAlert(
                        competitor_id=c.id,
                        alert_type="REVIEW_SPIKE",
                        severity="high",
                        message=f"{c.name} gained {v7} reviews in 7 days.",
                    )
                )
            if v30 >= 25:
                s.add(
                    CompetitorAlert(
                        competitor_id=c.id,
                        alert_type="AGGRESSIVE_GROWTH",
                        severity="high",
                        message=f"{c.name} gained {v30} reviews in 30 days.",
                    )
                )
        s.commit()
    finally:
        s.close()

# ----------------------
# HELPERS: DB ops
# ----------------------
def get_session():
    return SessionLocal()
def leads_to_df(start_date=None, end_date=None):
    """Load leads into a DataFrame. Filter by optional start_date/end_date (date objects)."""
    s = get_session()
    try:
        rows = s.query(Lead).order_by(Lead.created_at.desc()).all()
        data = []
        for r in rows:
            data.append(
                {
                    "id": r.id,
                    "lead_id": r.lead_id,
                    "created_at": r.created_at,
                    "source": r.source or "Other",
                    "source_details": getattr(r, "source_details", None),
                    "contact_name": getattr(r, "contact_name", None),
                    "contact_phone": getattr(r, "contact_phone", None),
                    "contact_email": getattr(r, "contact_email", None),
                    "property_address": getattr(r, "property_address", None),
                    "damage_type": getattr(r, "damage_type", None),
                    "assigned_to": getattr(r, "assigned_to", None),
                    "notes": r.notes,
                    "estimated_value": float(r.estimated_value or 0.0),
                    "stage": r.stage or "New",
                    "sla_hours": int(r.sla_hours or DEFAULT_SLA_HOURS),
                    "sla_entered_at": r.sla_entered_at or r.created_at,
                    "contacted": bool(r.contacted),
                    "inspection_scheduled": bool(r.inspection_scheduled),
                    "inspection_scheduled_at": r.inspection_scheduled_at,
                    "inspection_completed": bool(r.inspection_completed),
                    "estimate_submitted": bool(r.estimate_submitted),
                    "awarded_date": r.awarded_date,
                    "lost_date": r.lost_date,
                    "qualified": bool(r.qualified),
                    "ad_cost": float(r.ad_cost or 0.0),
                    "converted": bool(r.converted),
                    "score": float(r.score) if r.score is not None else None,
                }
            )

        df = pd.DataFrame(data)
        if df.empty:
            cols = [
                "id", "lead_id", "created_at", "source", "source_details", "contact_name", "contact_phone", "contact_email",
                "property_address", "damage_type", "assigned_to", "notes", "estimated_value", "stage", "sla_hours",
                "sla_entered_at", "contacted", "inspection_scheduled", "inspection_scheduled_at", "inspection_completed",
                "estimate_submitted", "awarded_date", "lost_date", "qualified", "ad_cost", "converted", "score",
            ]
            return pd.DataFrame(columns=cols)

        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            df = df[df["created_at"] >= start_dt]
        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            df = df[df["created_at"] <= end_dt]
        return df.reset_index(drop=True)
    finally:
        s.close()

def set_logged_in_user(user: User):
    st.session_state["user_id"] = user.id
# ----------------------
# ACCOUNT REACTIVATION
# ----------------------
def reactivate_user_account(user: User, plan: str):
    """
Re-enables a locked account after successful upgrade
"""
    user.plan = plan
    user.subscription_status = "active"
    user.is_active = True
    user.email_verified = True  # safe if coming from Stripe / admin
    user.trial_ends_at = None
# ----------------------
# TRIAL REMINDER EMAILS
# ----------------------
TRIAL_REMINDER_DAYS = [7, 3, 1]  # days before expiration

def send_trial_expiry_reminders():
    """
Send reminder emails to users whose trials are expiring soon.
Safe to run multiple times (idempotent by date).
"""
    now = pd.Timestamp.utcnow()
    with SessionLocal() as s:
        users = (
            s.query(User)
            .filter(
                User.subscription_status == "trial",
                User.trial_ends_at.isnot(None),
                User.is_active == True,
            )
            .all()
        )

    for user in users:
        days_left = (user.trial_ends_at - now).days if user.trial_ends_at else None
        if days_left in TRIAL_REMINDER_DAYS:
            try:
                send_trial_reminder_email(user.email, days_left)
            except Exception as e:
                print(f"Failed reminder email for {user.email}: {e}")


# ----------------------
# BILLING PROVIDER ABSTRACTION
# ----------------------
class BillingProvider:
    """Interface for all payment providers."""

    def create_checkout(self, user, plan):
        raise NotImplementedError

    def verify_payment(self, payload):
        raise NotImplementedError

    def cancel_subscription(self, user):
        raise NotImplementedError


class DummyBillingProvider(BillingProvider):
    """Temporary provider for manual / offline payments."""

    def create_checkout(self, user, plan):
        return {"status": "pending", "message": "Payment instructions sent manually"}

    def verify_payment(self, payload):
        return True

    def cancel_subscription(self, user):
        return True


def upgrade_user_plan(user, new_plan):
    _checkout = BILLING_PROVIDER.create_checkout(user, new_plan)
    # Manual approval or webhook later
    user.plan = new_plan
    user.subscription_status = "active"
    user.trial_ends_at = None
    with SessionLocal() as s:
        s.merge(user)
        s.commit()


# ----------------------
# AUTH HELPERS
# ----------------------
# ----------------------
# DEV MODE (RESET & STABILIZE)
# ----------------------
def bootstrap_admin():
    """
Ensures at least one Admin user exists.
MUST NEVER crash the app.
"""
    from sqlalchemy import inspect

    try:
        inspector = inspect(engine)
        if "users" not in inspector.get_table_names():
            return None

        with SessionLocal() as s:
            admin = s.query(User).filter(User.role == "Admin").first()
            if admin:
                return admin

            admin = User(
                email="admin@recapture.local",
                username="admin",
                full_name="System Admin",
                role="Admin",
                plan="pro",
                is_active=True,
                email_verified=True,
            )
            s.add(admin)
            s.commit()
            s.refresh(admin)
            return admin
    except Exception as e:
        # NEVER crash auth bootstrap
        print(" bootstrap_admin skipped:", e)
        return None


def get_current_user():
    user_id = st.session_state.get("user_id")
    if not user_id:
        return None
    with SessionLocal() as s:
        user = s.query(User).get(user_id)
        if not user:
            st.session_state.clear()
            st.warning("Session expired. Please log in again.")
            st.stop()
        return user

def decode_wp_token(token: str):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None
def generate_invite_token():
    return secrets.token_urlsafe(32)


def generate_activation_token():
    return secrets.token_urlsafe(32)
def verify_invite_token(token: str) -> User:
    try:
        payload = jwt.decode(token, INVITE_SECRET, algorithms=["HS256"])
        email = payload.get("email")
        token_hash = hashlib.sha256(token.encode()).hexdigest()
        with SessionLocal() as s:
            user = (
        s.query(User)
        .filter(
        User.email == email,
        User.invite_token_hash == token_hash,
        User.invite_expires_at > pd.Timestamp.utcnow(),
        User.activated_at.is_(None),
        )
        .first()
        )
        if not user:
            raise ValueError("Invalid or expired invite")
        # Activate user
        user.is_active = True
        user.activated_at = pd.Timestamp.utcnow()
        user.invite_token_hash = None
        user.invite_expires_at = None
        s.commit()
        return user
    except jwt.ExpiredSignatureError:
        raise ValueError("Invite expired")
def hash_password(password: str) -> str:
    return bcrypt.hash(password)
def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.verify(password, hashed)
def generate_reset_token() -> str:
    return secrets.token_urlsafe(32)
import random
def generate_otp():
    return str(random.randint(100000, 999999))
def send_otp_email(email, otp):
    subject = "Your ReCapture Pro verification code"
    body = f"""
Your verification code is:
{otp}
This code expires in 5 minutes.
If you did not request this, ignore this email.
"""
    send_email(email, subject, body)


def has_feature(user, feature_key):
    if PUBLIC_FREE_LAUNCH:
        return True
    return feature_key in PLAN_FEATURES.get(user.plan, [])
# ----------------------
# PLAN LIMIT ENFORCEMENT
# ----------------------
#ENFORCE PLAN WAS REPROGRAMMED FOR FREEMIUM
def enforce_plan_limit(*args, **kwargs):
    if PUBLIC_FREE_LAUNCH:
        return True
# Admin bypass
#if user and user.role == "Admin":
    #return True
def enforce_org_seat_limit(current_user):
    """
Enforces organization-based seat limits.
Safe to call from anywhere.
"""
    # DEV MODE → never block
    if st.secrets.get("DEV_MODE") == "true":
        return

    with SessionLocal() as s:
        org = s.get(Organization, current_user.organization_id)
        if not org or not org.max_users:
            return  # unlimited or misconfigured org

        user_count = s.query(User).filter(User.organization_id == org.id).count()
        if user_count >= org.max_users:
            st.error("User limit reached for your plan.")
            st.stop()


# ----------------------
# BILLING PROVIDER (DEV / MANUAL)
# ----------------------
class ManualBillingProvider:
    def charge(self, user, amount):
        print(f"[BILLING] Simulated charge: {user.email} → ${amount}")
        return True

    def cancel(self, user):
        print(f"[BILLING] Simulated cancel for {user.email}")
        return True


BILLING_PROVIDER = ManualBillingProvider()


def send_email(to_email, subject, html_body):
    url = "https://api.resend.com/emails"
    headers = {
        "Authorization": f"Bearer {st.secrets['RESEND_API_KEY']}",
        "Content-Type": "application/json",
    }
    payload = {
        "from": st.secrets.get("EMAIL_FROM", "ReCapture Pro <onboarding@resend.dev>"),
        "to": [to_email],
        "subject": subject,
        "html": html_body,
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.status_code, response.text


def send_invite_email(email: str, invite_link: str):
    subject = "You're invited to ReCapture Pro"
    html = (
        "<p>You have been invited to join ReCapture Pro.</p>"
        f"<p><a href='{invite_link}'>Activate your account</a></p>"
    )
    status, response = send_email(email, subject, html)
    if status >= 400:
        raise RuntimeError(f"Invite email failed ({status}): {response}")

def build_review_email(
customer_name,
business_name,
job_type,
review_link,
custom_message,
footer
):
    return f"""
<p>Hi {customer_name},</p>
<p>{custom_message}</p>
<p>
    If you have a moment, please leave us a review here:<br>
        <a href="{review_link}">Leave a Google Review</a>
</p>
<p>
    Job completed: <strong>{job_type}</strong>
</p>
<br>
<p>{footer}</p>
<hr>
<small>{business_name}</small>
"""
def send_review_request_email(
    contact,
    template,
    review_link,
    business_name,
):
    html = build_review_email(
        customer_name=contact["name"],
        business_name=business_name,
        job_type=contact.get("job_type", "Recent service"),
        review_link=review_link,
        custom_message=template["body"],
        footer=template["footer"],
    )
    status, _response = send_email(
        to_email=contact["email"],
        subject=template["subject"],
        html_body=html,
    )
    return status == 200


def log_review_request(user_id, email, status):
    with SessionLocal() as s:
        s.add(
            ReviewRequestLog(
                user_id=user_id,
                recipient=email,
                status=status,
            )
        )
        s.commit()


def get_total_leads_for_account(user):
    """
Temporary single-tenant helper.
Returns total number of leads.
"""
    if not user:
        return 0
    with SessionLocal() as s:
        return s.query(Lead).count()


def sync_ai_insights(user_id, generated_insights):
    from models import AIInsight

    with SessionLocal() as s:
        existing = {
            i.insight_key: i
            for i in s.query(AIInsight)
            .filter(
                AIInsight.user_id == user_id,
                AIInsight.is_active == True,
            )
            .all()
        }
        generated_keys = set()

        for insight in generated_insights:
            key = insight["key"]
            generated_keys.add(key)
            if key not in existing:
                s.add(
                    AIInsight(
                        user_id=user_id,
                        insight_key=key,
                        message=insight["message"],
                    )
                )
            elif existing[key].message != insight["message"]:
                existing[key].message = insight["message"]

        for key, record in existing.items():
            if key not in generated_keys:
                record.is_active = False
                record.resolved_at = pd.Timestamp.utcnow()

        s.commit()

# ---------- BEGIN BLOCK C: DB HELPERS FOR TECHNICIANS / ASSIGNMENTS / PINGS
def create_task(title, technician_username=None, lead_id=None, due_at=None, description=None):
    s = get_session()
    try:
        task = Task(
            title=title,
            technician_username=technician_username,
            lead_id=lead_id,
            description=description,
            status="open",
            due_at=due_at,
        )
        s.add(task)
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def update_task_status(task_id: int, new_status: str):
    s = get_session()
    try:
        task = s.query(Task).filter(Task.id == task_id).first()
        if not task:
            return False
        task.status = new_status
        s.add(task)
        s.commit()
        return True
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_tasks_for_user(username):
    s = get_session()
    try:
        rows = s.query(Task).filter(Task.technician_username == username).all()
        return pd.DataFrame(
            [
                {
                    "id": r.id,
                    "title": r.title,
                    "status": r.status,
                    "lead_id": r.lead_id,
                    "due_at": r.due_at,
                }
                for r in rows
            ]
        )
    finally:
        s.close()


def page_tasks():
    require_role_access("tasks")
    st.markdown("## Technician Tasks")

    techs = get_technicians_df(active_only=True)
    if techs.empty:
        st.warning("No technicians available.")
        return

    tech_username = st.selectbox("Select Technician", techs["username"].tolist())
    tasks_df = get_tasks_for_user(tech_username)
    if tasks_df.empty:
        st.info(
            " No task assigned to a Technician yet! To assign a job task to a technician, go to:SETTINGS at the Navigation Menu, then click on the TECHNICIAN MANAGEMENT."
        )
        return

    for _, row in tasks_df.iterrows():
        with st.expander(f" {row['title']} — {row['status'].upper()}"):
            st.write(f"**Lead ID:** {row['lead_id'] or 'N/A'}")
            st.write(f"**Due:** {row['due_at'] or 'No due date'}")

            if row["status"] == "open":
                if st.button(" Start Task", key=f"start_{row['id']}"):
                    update_task_status(row["id"], "in_progress")
                    st.success("Task started")
                    st.rerun()
            elif row["status"] == "in_progress":
                if st.button(" Mark Complete", key=f"done_{row['id']}"):
                    update_task_status(row["id"], "done")
                    st.success("Task completed")
                    st.rerun()
            elif row["status"] == "done":
                st.success("✔ Completed")


def get_tasks_df():
    s = get_session()
    try:
        rows = s.query(Task).order_by(Task.created_at.desc()).all()
        return pd.DataFrame(
            [
                {
                    "id": r.id,
                    "title": r.title,
                    "technician_username": r.technician_username,
                    "lead_id": r.lead_id,
                    "status": r.status,
                    "due_at": r.due_at,
                    "created_at": r.created_at,
                }
                for r in rows
            ]
        )
    finally:
        s.close()


def add_technician(
    username: str,
    full_name: str = "",
    phone: str = "",
    specialization: str = "Tech",
    active: bool = True,
):
    s = get_session()
    try:
        existing = s.query(Technician).filter(Technician.username == username).first()
        if existing:
            existing.full_name = full_name
            existing.phone = phone
            existing.specialization = specialization
            existing.active = active
            s.add(existing)
            s.commit()
            return existing.username

        t = Technician(
            username=username,
            full_name=full_name,
            phone=phone,
            specialization=specialization,
            active=active,
        )
        s.add(t)
        s.commit()
        return t.username
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def update_technician_status(username: str, status: str):
    s = get_session()
    try:
        tech = s.query(Technician).filter_by(username=username).first()
        if not tech:
            return False
        tech.status = status
        s.commit()
        return True
    finally:
        s.close()


if "_save_location" in st.query_params:
    data = st.get_json()
    save_location_ping(
        data["username"],
        data["lat"],
        data["lon"],
        data.get("accuracy"),
    )
    st.stop()


def save_location_ping(username, lat, lon, accuracy=None):
    s = get_session()
    try:
        ping = LocationPing(
            tech_username=username,
            latitude=float(lat),
            longitude=float(lon),
            accuracy=accuracy,
            timestamp=pd.Timestamp.utcnow(),
        )
        s.add(ping)
        s.commit()
    finally:
        s.close()

def get_technicians_df(active_only=True):
    s = get_session()
    try:
        q = s.query(Technician)
        if active_only:
            q = q.filter(Technician.active == True)
        rows = q.all()
        return pd.DataFrame(
            [
                {
                    "username": t.username,
                    "full_name": t.full_name,
                    "phone": t.phone,
                    "specialization": t.specialization,
                    "active": t.active,
                }
                for t in rows
            ]
        )
    finally:
        s.close()


def save_location_ping(
    tech_username: str,
    latitude: float,
    longitude: float,
    lead_id: str | None = None,
    accuracy: float | None = None,
):
    s = get_session()
    try:
        ping = LocationPing(
            tech_username=tech_username,
            latitude=latitude,
            longitude=longitude,
            lead_id=lead_id,
            accuracy=accuracy,
        )
        s.add(ping)
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def get_leads_df():
    response = supabase.table("leads").select("*").execute()
    if not response.data:
        return pd.DataFrame()

    df = pd.DataFrame(response.data)

    defaults = {
        "estimated_value": 0,
        "ad_cost": 0,
        "stage": "new",
        "score": 0.5,
        "damage_type": "Unknown",
    }
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
        df[col] = df[col].fillna(val)

    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce", utc=True)

    df["estimated_value"] = pd.to_numeric(df["estimated_value"], errors="coerce").fillna(0)
    df["ad_cost"] = pd.to_numeric(df["ad_cost"], errors="coerce").fillna(0)
    return df


def get_jobs_for_period(start_dt, end_dt):
    df = get_leads_df()
    if df.empty:
        return df
    return df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)]


def compute_job_volume_metrics(df):
    return {
        "total_jobs": len(df),
        "job_types": safe_col(df, "job_type").value_counts().to_dict(),
        "lead_sources": safe_col(df, "lead_source").value_counts().to_dict(),
    }


def generate_weekly_business_pulse(df):
    stalled_revenue = df[df["stage"].isin(["Inspection", "Estimate Sent"])]["estimated_value"].sum()
    follow_ups = len(df[df["stage"].isin(["New", "Contacted"])])
    won = len(df[df["stage"] == "Won"])
    inspections = len(df[df["stage"] == "Inspection"])
    conversion = (won / inspections * 100) if inspections else 0
    return f"""
Weekly Business Pulse
    Stalled Revenue: ${stalled_revenue:,.0f}
    Leads needing follow-up: {follow_ups}
    Inspection → Won conversion: {conversion:.0f}%
Log in to ReCapture Pro to take action.
"""


def generate_ai_advice(df):
    insights = []

    follow_up_count = len(df[df["stage"].isin(["New", "Contacted"])])
    if follow_up_count >= 5:
        insights.append(
            {
                "key": "follow_up_delay",
                "message": (
                    "You have multiple new leads awaiting follow-up. "
                    "Reducing response time could significantly increase close rates."
                ),
            }
        )

    stalled = df[df["stage"].isin(["Inspection", "Estimate Sent"])]
    if not stalled.empty:
        insights.append(
            {
                "key": "stalled_revenue",
                "message": (
                    "Several high-value leads appear stalled. "
                    "Completing inspections and sending estimates could unlock revenue."
                ),
            }
        )

    if "created_at" in df.columns:
        avg_response = (
            (pd.Timestamp.utcnow() - pd.to_datetime(df["created_at"], errors="coerce"))
            .dt.total_seconds()
            .mean()
            / 3600
        )
        if avg_response and avg_response > 4:
            insights.append(
                {
                    "key": "slow_response_time",
                    "message": (
                        "Your average response time is slower than optimal. "
                        "Aim for under 2 hours to improve conversions."
                    ),
                }
            )

    return insights


def compute_revenue_metrics(df):
    revenue_col = safe_col(df, "estimated_value", default_dtype=float).fillna(0)
    total_revenue = revenue_col.sum()
    revenue_per_job = total_revenue / len(df) if len(df) > 0 else 0
    return {
        "total_revenue": float(total_revenue),
        "revenue_per_job": float(revenue_per_job),
    }


def compute_efficiency_metrics(df):
    revenue_col = safe_col(df, "estimated_value", default_dtype=float).fillna(0)
    jobs = len(df)
    total_revenue = revenue_col.sum()
    return {
        "jobs": jobs,
        "revenue_per_job": (total_revenue / jobs if jobs > 0 else 0),
    }


def generate_synthetic_signals(df):
    signals = []
    revenue_col = safe_col(df, "estimated_value", default_dtype=float).fillna(0)
    job_type_col = safe_col(df, "job_type")

    if len(df) == 0:
        signals.append(" No job activity detected in this period.")
        return signals

    avg_revenue = revenue_col.mean()
    if avg_revenue < 500:
        signals.append(" High job volume but low average revenue per job.")

    dominant_job = job_type_col.mode().iloc[0] if not job_type_col.empty else None
    if dominant_job:
        signals.append(f" Revenue is highly concentrated in '{dominant_job}' jobs.")

    return signals


def seasonal_baseline(all_jobs, start_date, end_date):
    same_months = list(range(start_date.month, end_date.month + 1))
    hist = all_jobs[all_jobs["created_at"].dt.month.isin(same_months)]
    if hist.empty:
        return None
    return {
        "avg_jobs": hist.groupby(hist["created_at"].dt.year).size().mean(),
        "avg_revenue": hist.groupby(hist["created_at"].dt.year)["estimated_value"].sum().mean(),
        "avg_rev_per_job": hist["estimated_value"].mean(),
    }


def shift_period(start, end):
    delta = end - start
    return start - delta, end - delta


def safe_col(df, col, default_dtype=None):
    if col not in df.columns:
        if default_dtype:
            return pd.Series(dtype=default_dtype)
        return pd.Series()

    s = df[col]
    if default_dtype:
        s = pd.to_numeric(s, errors="coerce")
    return s


def init_trial():
    if "trial_start" not in st.session_state:
        st.session_state["trial_start"] = pd.Timestamp.utcnow()
    st.session_state["plan"] = "trial"


def is_trial_active(days=14):
    start = st.session_state.get("trial_start")
    if not start:
        return False
    return (pd.Timestamp.utcnow() - start).days < days


def count_leads_this_month():
    start = pd.Timestamp.utcnow().replace(day=1, hour=0, minute=0, second=0)
    end = pd.Timestamp.utcnow()
    df = leads_to_df(start, end)
    return len(df)


def get_current_plan():
    user = st.session_state.get("user")
    if not user:
        return "starter"

    if not DEV_MODE:
        if user.trial_ends_at and user.trial_ends_at < pd.Timestamp.utcnow():
            st.error("Trial expired")
            st.stop()

    if user.subscription_status == "trial":
        if user.trial_ends_at and pd.Timestamp.utcnow() > user.trial_ends_at:
            return "expired"

    return user.plan


# ----------------------
# GET GOOGLE REVIEW SETTINGS
# ----------------------
def get_review_settings(org_id):
    with SessionLocal() as s:
        return s.query(ReviewSettings).filter_by(org_id=org_id).first()


# ----------------------
# SAVE / UPDATE REVIEW SETTINGS
# ----------------------
def save_review_settings(org_id, data):
    with SessionLocal() as s:
        settings = s.query(ReviewSettings).filter_by(org_id=org_id).first()
        if not settings:
            settings = ReviewSettings(org_id=org_id)

        for key, value in data.items():
            setattr(settings, key, value)

        s.add(settings)
        s.commit()


# ----------------------
# DELETE REVIEW EMAIL TEMPLATE
# ----------------------
def delete_email_template(template_id):
    with SessionLocal() as s:
        template = s.query(ReviewEmailTemplate).get(template_id)
        if template:
            s.delete(template)
            s.commit()


def has_access(feature_key: str) -> bool:
    plan = get_current_plan()
    if plan == "expired":
        return False
    return PLANS.get(plan, {}).get(feature_key, False)

# ----------------------
# ROLE-BASED ACCESS CONTROL
# ----------------------
ROLE_PERMISSIONS = {
"Admin": {
    "overview",
    "lead_capture",
    "pipeline",
    "analytics",
    "business_intelligence",
    "competitor_intelligence",
    "technicians",
    "settings",
    "billing",
},
"Manager": {
    "overview",
    "lead_capture",
    "pipeline",
    "analytics",
    "business_intelligence",
    "technicians",
},
"Staff": {
    "overview",
    "lead_capture",
    "pipeline",
},
"Viewer": {
    "overview",
    "pipeline",
},
}
# ----------------------
# PLAN CAPABILITIES (GLOBAL)
# ----------------------
PLAN_LIMITS = {
"starter": {
    "pages": {
    "overview",
    "lead_capture",
    "pipeline",
    "analytics",
    "tasks",
    },
    "max_users": 3,
    "max_leads": 100,
},
"pro": {
    "pages": {
    "overview",
    "lead_capture",
    "pipeline",
    "analytics",
    "tasks",
    "business_intelligence",
    "seasonal_trends",
    "exports",
    },
    "max_users": 10,
    "max_leads": 1000,
},
"enterprise": {
    "pages": {"*"},
    "max_users": 999,
    "max_leads": 999999,
},
}
# ----------------------
# PAGE ACCESS GUARD
# ----------------------
def require_role_access(page_key):
    user = get_current_user()



def get_user_settings_safe():
    """
Safe wrapper to prevent Request Review page crashes
"""
    try:
        return get_user_settings()
    except Exception:
        return {
            "google_review_url": "",
            "enable_nfc_review": True,
            "enable_qr_review": True,
        }


def generate_review_token():
    """
Generates a short-lived anonymous review token.
DB persistence can be added later.
"""
    return uuid.uuid4().hex

def log_event(
*,
event_type: str,
user_id: str | None = None,
org_id: str | None = None,
entity_type: str | None = None,
entity_id: str | None = None,
metadata: dict | None = None,
severity: str = "info",
source: str = "app"
):
    try:
        payload = {
        "id": str(uuid4()),
        "event_type": event_type,
        "metadata": json.dumps(metadata or {}),
        "created_at": pd.Timestamp.utcnow()
        }
        if user_id:
            payload.update({
        "user_id": user_id,
        "org_id": org_id,
        "entity_type": entity_type,
        "entity_id": entity_id
        })
        run_query(
        """
        INSERT INTO platform_user_events
        (id, user_id, org_id, event_type, entity_type, entity_id, metadata, created_at)
        VALUES (%(id)s, %(user_id)s, %(org_id)s, %(event_type)s, %(entity_type)s,
        %(entity_id)s, %(metadata)s, %(created_at)s)
        """,
        payload
        )

        payload.update({
        "severity": severity,
        "source": source
        })
        run_query(
        """
        INSERT INTO platform_events
        (id, event_type, severity, source, metadata, created_at)
        VALUES (%(id)s, %(event_type)s, %(severity)s, %(source)s, %(metadata)s,
        %(created_at)s)
        """,
        payload
        )
    except Exception:
    # NEVER crash the app because of logging
        pass
def analyze_job_types(df):
    if df.empty:
        return {}

    grouped = (
        df.groupby("damage_type")
        .agg(
            jobs=("id", "count"),
            revenue=("estimated_value", "sum"),
            avg_revenue=("estimated_value", "mean"),
            revenue_std=("estimated_value", "std"),
        )
        .reset_index()
    )

    total_jobs = grouped["jobs"].sum()
    total_revenue = grouped["revenue"].sum()
    grouped["job_share"] = grouped["jobs"] / total_jobs if total_jobs else 0
    grouped["revenue_share"] = grouped["revenue"] / total_revenue if total_revenue else 0
    grouped["volatility"] = grouped["revenue_std"].fillna(0)

    insights = []
    for _, r in grouped.iterrows():
        if r["job_share"] > 0.45:
            insights.append(f" Over-dependence on **{r['damage_type']}** jobs")
        if r["job_share"] > 0.3 and r["avg_revenue"] < grouped["avg_revenue"].mean():
            insights.append(f" **{r['damage_type']}** jobs are high-volume but low-value")
        if r["volatility"] > grouped["volatility"].mean() * 1.5:
            insights.append(f" **{r['damage_type']}** revenue is highly volatile")

    return {
        "table": grouped.sort_values("revenue", ascending=False),
        "insights": insights,
    }


def pct_change(current, previous):
    """
Safe percentage change calculation.
Returns 0 if previous is zero or missing.
"""
    try:
        if previous in (0, None):
            return 0
        return ((current - previous) / previous) * 100
    except Exception:
        return 0


def generate_executive_narrative(data):
    narrative = []
    risk_flags = []
    volume = data.get("volume", {})
    revenue = data.get("revenue", {})
    efficiency = data.get("efficiency", {})

    narrative.append(
        {
            "text": (
                f"Total jobs: {volume.get('total_jobs', 0)} | "
                f"Revenue: ${revenue.get('total_revenue', 0):,.0f} | "
                f"Revenue/job: ${efficiency.get('revenue_per_job', 0):,.0f}"
            ),
            "confidence": 85,
        }
    )

    health_score = 82
    return {
        "lines": narrative,
        "risk_flags": risk_flags,
        "health_score": health_score,
        "version": "G-hardened-v1",
    }

def page_business_intelligence():
    require_role_access("business_intelligence")

    st.markdown("## Business Intelligence")

    col1, col2 = st.columns([2, 3])
    with col1:
        range_key = st.selectbox(
            "Time Range",
            ["daily", "weekly", "30d", "90d", "6m", "12m", "custom"],
            index=2,
        )
    with col2:
        custom_start, custom_end = None, None
        if range_key == "custom":
            custom_start = st.date_input("Start date")
            custom_end = st.date_input("End date")

    data = compute_business_intelligence(range_key, custom_start, custom_end)
    df = data.get("raw_df")
    if df is None or df.empty:
        st.warning("No jobs found for this period.")
        return

    health = data.get("health")
    if health:
        st.metric("Overall Business Health", f"{health.get('score', 0)} / 100")

    v = data.get("volume", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Jobs", v.get("total_jobs", 0))
    c2.metric("Jobs / Day", v.get("jobs_per_day", 0))
    c3.metric("Lead Sources", len(v.get("lead_sources", {})))

    r = data.get("revenue", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${r.get('total_revenue', 0):,.0f}")
    c2.metric("Avg / Job", f"${r.get('avg_revenue_per_job', 0):,.0f}")
    c3.metric("Revenue Risk", f"{int(r.get('revenue_concentration', 0) * 100)}% Top Dependency")

def page_technician_map_tracking():
    st.markdown("## Technician Live Map")
    df = get_latest_location_pings()
    if df.empty:
        st.warning("No technician GPS data available.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    def classify_status(ts):
        if pd.isna(ts):
            return "offline"
        mins = (pd.Timestamp.utcnow() - ts).total_seconds() / 60
        if mins <= 10:
            return "active"
        if mins <= 30:
            return "idle"
        return "offline"

    df["status"] = df["timestamp"].apply(classify_status)
    status_color = {"active": "green", "idle": "orange", "offline": "red"}

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=8,
            color=status_color.get(r["status"], "gray"),
            fill=True,
            fill_opacity=0.85,
            popup=f"{r.get('tech_username', 'tech')} ({r['status']})",
        ).add_to(m)

    st_folium(m, width=900, height=500)


def page_technician_mobile():
    st.markdown("## Technician Mobile")
    st.info("Use the mobile endpoint to push technician GPS pings.")

def page_cpa_roi():
    require_role_access("analytics")
    st.markdown("<div class='header'> CPA & ROI</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Total Marketing Spend vs Conversions and ROI calculations.</em>",
        unsafe_allow_html=True,
    )

    df = leads_to_df()
    if df.empty:
        st.info("No leads")
        return

    total_spend = float(df.get("ad_cost", 0).sum())
    won_df = df[df["stage"] == "Won"] if "stage" in df.columns else pd.DataFrame()
    conversions = len(won_df)
    cpa = (total_spend / conversions) if conversions else 0.0
    revenue = float(won_df.get("estimated_value", 0).sum()) if not won_df.empty else 0.0
    roi = revenue - total_spend
    roi_pct = (roi / total_spend * 100) if total_spend else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Marketing Spend", f"${total_spend:,.2f}")
    c2.metric("Conversions (Won)", f"{conversions}")
    c3.metric("CPA", f"${cpa:,.2f}")
    c4.metric("ROI", f"${roi:,.2f} ({roi_pct:.1f}%)")


def page_ml_internal():
    st.markdown("<div class='header'> Internal ML — Lead Scoring</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Model runs internally and writes score back to leads. No user tuning exposed.</em>",
        unsafe_allow_html=True,
    )

    if st.button("Train model (internal)"):
        with st.spinner("Training..."):
            try:
                acc, msg = train_internal_model()
                if acc is None:
                    st.error(f"Training aborted: {msg}")
                else:
                    st.success(f"Model trained (accuracy approx): {acc:.3f}")
            except Exception as e:
                st.error("Training failed: " + str(e))

def page_ai_recommendations():
    require_role_access("business_intelligence")
    st.markdown("<div class='header'> AI Recommendations</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Heuristic recommendations and quick diagnostics for the pipeline.</em>",
        unsafe_allow_html=True,
    )

    try:
        df = leads_to_df()
    except Exception as e:
        st.error(f"Failed to load leads: {e}")
        df = pd.DataFrame()

    if df.empty:
        st.info("No leads to analyze.")
        return

    st.subheader("Top Overdue Leads")
    overdue_list = []
    for _, r in df.iterrows():
        rem_s, overdue_flag = calculate_remaining_sla(
            r.get("sla_entered_at") or r.get("created_at"),
            r.get("sla_hours"),
        )
        if overdue_flag and r.get("stage") not in ("Won", "Lost"):
            overdue_list.append(
                {
                    "lead_id": r.get("lead_id"),
                    "stage": r.get("stage"),
                    "assigned_to": r.get("assigned_to"),
                    "value": r.get("estimated_value") or 0.0,
                    "overdue_seconds": rem_s,
                }
            )

    over_df = pd.DataFrame(overdue_list)
    if not over_df.empty:
        over_df = over_df.sort_values("value", ascending=False)
        st.table(over_df[["lead_id", "stage", "assigned_to", "value"]].head(10))
    else:
        st.info("No overdue leads.")

pwd_context = CryptContext(
schemes=["bcrypt"],
deprecated="auto"
)
def hash_password(password: str) -> str:
    return pwd_context.hash(password)
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
# ----------------------
# WORDPRESS AUTH
# ----------------------
def reset_password_with_token(token: str, new_password: str):
    with SessionLocal() as s:
        user = s.query(User).filter(
    User.reset_token == token,
    User.reset_expires_at > pd.Timestamp.utcnow()
    ).first()
    if not user:
        raise Exception("Invalid or expired reset token")
    user.password_hash = hash_password(new_password)
    user.reset_token = None
    user.reset_expires_at = None
    user.email_verified = True
    s.commit()
def request_password_reset(email: str):
    with SessionLocal() as s:
        user = s.query(User).filter(User.email == email.lower()).first()
    if not user:
        return # Silent fail for security
    token = generate_password_reset_token()
    user.password_reset_token = token
    user.password_reset_expires_at = pd.Timestamp.utcnow() + timedelta(hours=1)
    s.commit()
    reset_link = f"{FRONTEND_URL}/reset-password?token={token}"
    send_password_reset_email(user.email, reset_link)
def send_password_reset_email(email: str, reset_link: str):
# TEMP: console output for testing
    print("PASSWORD RESET LINK:", reset_link)
# Later you can wire:
# SendGrid / Mailgun / SMTP / SES
#--------------------OPTIONAL PAGE RESET FOR STREAMLIT---------------
def page_reset_password():
    st.markdown("## Reset Your Password")

    token = st.query_params.get("token")
    if not token:
        st.error("Missing reset token")
        st.stop()

    pw1 = st.text_input("New Password", type="password")
    pw2 = st.text_input("Confirm Password", type="password")

    if st.button("Reset Password"):
        if not pw1 or len(pw1) < 8:
            st.error("Password must be at least 8 characters")
            st.stop()
        if pw1 != pw2:
            st.error("Passwords do not match")
            st.stop()

        try:
            reset_password_with_token(token, pw1)
            st.success("Password reset successful. You may now log in.")
            st.stop()
        except Exception as e:
            st.error(str(e))
def record_invoice(user, amount, description):
    with SessionLocal() as s:
        invoice = Invoice(
    user_id=user.id,
    amount=amount,
    status="paid",
    description=description,
    )
    s.add(invoice)
    s.commit()
def authenticate_user(email: str, password: str):
    with SessionLocal() as s:
        user = s.query(User).filter(
    User.email == email.lower(),
    User.is_active == True
    ).first()
    if not user:
        return None
    if not user.password_hash:
        return None
    # ---- Check if account is locked ----
    if user.locked_until and user.locked_until > pd.Timestamp.utcnow():
        raise Exception("Account locked. Try again later.")
    # ---- Verify password ----
    if not verify_password(password, user.password_hash):
        user.failed_login_attempts += 1
    if user.failed_login_attempts >= 5:
        user.locked_until = pd.Timestamp.utcnow() + timedelta(minutes=30)
    s.commit()
    raise Exception("Invalid credentials")
    # ---- SUCCESS: reset failed attempts ----
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = pd.Timestamp.utcnow()
    # Generate OTP for Admins or first-time login
    otp = generate_otp()
    user.otp_code = otp
    user.otp_expires_at = pd.Timestamp.utcnow() + timedelta(minutes=5)
    user.otp_required = True
    send_otp_email(user.email, otp)
    s.commit()
    # Mark user as pending OTP
    st.session_state["otp_user_id"] = user.id
    return "OTP_REQUIRED"
def request_password_reset(email: str):
    token = generate_reset_token()
    with SessionLocal() as s:
        user = s.query(User).filter(User.email == email).first()
        if not user:
            return  # silent fail (security)

        user.reset_token = token
        user.reset_expires_at = pd.Timestamp.utcnow() + timedelta(hours=1)
        s.commit()

    reset_link = f"{FRONTEND_URL}/reset-password?token={token}"
    send_password_reset_email(email, reset_link)


def admin_upgrade_user(user_id: int, plan: str):
    with SessionLocal() as s:
        user = s.query(User).filter(User.id == user_id).first()
        if not user:
            return False

        reactivate_user_account(user, plan)
        s.commit()
        return True

# -------------------------
# Settings Page
# -------------------------
def get_users_df():
    with SessionLocal() as s:
        rows = s.query(User).order_by(User.created_at.desc()).all()

    data = [
        {
            "id": u.id,
            "email": u.email,
            "username": u.username,
            "full_name": u.full_name,
            "role": u.role,
            "plan": u.plan,
            "is_active": bool(u.is_active),
            "email_verified": bool(u.email_verified),
            "created_at": u.created_at,
        }
        for u in rows
    ]
    return pd.DataFrame(data)


def get_user_settings_safe():
    """
Safe wrapper to prevent Request Review page crashes
"""
    try:
        return get_user_settings()
    except Exception:
        return {
            "google_review_url": "",
            "enable_nfc_review": True,
            "enable_qr_review": True,
        }


def generate_review_token():
    """
Generates a short-lived anonymous review token.
DB persistence can be added later.
"""
    return uuid.uuid4().hex

def log_event(
*,
event_type: str,
user_id: str | None = None,
org_id: str | None = None,
entity_type: str | None = None,
entity_id: str | None = None,
metadata: dict | None = None,
severity: str = "info",
source: str = "app"
):
    try:
        payload = {
        "id": str(uuid4()),
        "event_type": event_type,
        "metadata": json.dumps(metadata or {}),
        "created_at": pd.Timestamp.utcnow()
        }
        if user_id:
            payload.update({
        "user_id": user_id,
        "org_id": org_id,
        "entity_type": entity_type,
        "entity_id": entity_id
        })
        run_query(
        """
        INSERT INTO platform_user_events
        (id, user_id, org_id, event_type, entity_type, entity_id, metadata, created_at)
        VALUES (%(id)s, %(user_id)s, %(org_id)s, %(event_type)s, %(entity_type)s,
        %(entity_id)s, %(metadata)s, %(created_at)s)
        """,
        payload
        )

        payload.update({
        "severity": severity,
        "source": source
        })
        run_query(
        """
        INSERT INTO platform_events
        (id, event_type, severity, source, metadata, created_at)
        VALUES (%(id)s, %(event_type)s, %(severity)s, %(source)s, %(metadata)s,
        %(created_at)s)
        """,
        payload
        )
    except Exception:
    # NEVER crash the app because of logging
        pass
def analyze_job_types(df):
    if df.empty:
        return {}

    grouped = (
        df.groupby("damage_type")
        .agg(
            jobs=("id", "count"),
            revenue=("estimated_value", "sum"),
            avg_revenue=("estimated_value", "mean"),
            revenue_std=("estimated_value", "std"),
        )
        .reset_index()
    )

    total_jobs = grouped["jobs"].sum()
    total_revenue = grouped["revenue"].sum()
    grouped["job_share"] = grouped["jobs"] / total_jobs if total_jobs else 0
    grouped["revenue_share"] = grouped["revenue"] / total_revenue if total_revenue else 0
    grouped["volatility"] = grouped["revenue_std"].fillna(0)

    insights = []
    for _, r in grouped.iterrows():
        if r["job_share"] > 0.45:
            insights.append(f" Over-dependence on **{r['damage_type']}** jobs")
        if r["job_share"] > 0.3 and r["avg_revenue"] < grouped["avg_revenue"].mean():
            insights.append(f" **{r['damage_type']}** jobs are high-volume but low-value")
        if r["volatility"] > grouped["volatility"].mean() * 1.5:
            insights.append(f" **{r['damage_type']}** revenue is highly volatile")

    return {
        "table": grouped.sort_values("revenue", ascending=False),
        "insights": insights,
    }


def pct_change(current, previous):
    """
Safe percentage change calculation.
Returns 0 if previous is zero or missing.
"""
    try:
        if previous in (0, None):
            return 0
        return ((current - previous) / previous) * 100
    except Exception:
        return 0


def generate_executive_narrative(data):
    narrative = []
    risk_flags = []
    volume = data.get("volume", {})
    revenue = data.get("revenue", {})
    efficiency = data.get("efficiency", {})

    narrative.append(
        {
            "text": (
                f"Total jobs: {volume.get('total_jobs', 0)} | "
                f"Revenue: ${revenue.get('total_revenue', 0):,.0f} | "
                f"Revenue/job: ${efficiency.get('revenue_per_job', 0):,.0f}"
            ),
            "confidence": 85,
        }
    )

    health_score = 82
    return {
        "lines": narrative,
        "risk_flags": risk_flags,
        "health_score": health_score,
        "version": "G-hardened-v1",
    }

def page_business_intelligence():
    require_role_access("business_intelligence")

    st.markdown("## Business Intelligence")

    col1, col2 = st.columns([2, 3])
    with col1:
        range_key = st.selectbox(
            "Time Range",
            ["daily", "weekly", "30d", "90d", "6m", "12m", "custom"],
            index=2,
        )
    with col2:
        custom_start, custom_end = None, None
        if range_key == "custom":
            custom_start = st.date_input("Start date")
            custom_end = st.date_input("End date")

    data = compute_business_intelligence(range_key, custom_start, custom_end)
    df = data.get("raw_df")
    if df is None or df.empty:
        st.warning("No jobs found for this period.")
        return

    health = data.get("health")
    if health:
        st.metric("Overall Business Health", f"{health.get('score', 0)} / 100")

    v = data.get("volume", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Jobs", v.get("total_jobs", 0))
    c2.metric("Jobs / Day", v.get("jobs_per_day", 0))
    c3.metric("Lead Sources", len(v.get("lead_sources", {})))

    r = data.get("revenue", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${r.get('total_revenue', 0):,.0f}")
    c2.metric("Avg / Job", f"${r.get('avg_revenue_per_job', 0):,.0f}")
    c3.metric("Revenue Risk", f"{int(r.get('revenue_concentration', 0) * 100)}% Top Dependency")

def page_technician_map_tracking():
    st.markdown("## Technician Live Map")
    df = get_latest_location_pings()
    if df.empty:
        st.warning("No technician GPS data available.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    def classify_status(ts):
        if pd.isna(ts):
            return "offline"
        mins = (pd.Timestamp.utcnow() - ts).total_seconds() / 60
        if mins <= 10:
            return "active"
        if mins <= 30:
            return "idle"
        return "offline"

    df["status"] = df["timestamp"].apply(classify_status)
    status_color = {"active": "green", "idle": "orange", "offline": "red"}

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=8,
            color=status_color.get(r["status"], "gray"),
            fill=True,
            fill_opacity=0.85,
            popup=f"{r.get('tech_username', 'tech')} ({r['status']})",
        ).add_to(m)

    st_folium(m, width=900, height=500)


def page_technician_mobile():
    st.markdown("## Technician Mobile")
    st.info("Use the mobile endpoint to push technician GPS pings.")

def page_cpa_roi():
    require_role_access("analytics")
    st.markdown("<div class='header'> CPA & ROI</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Total Marketing Spend vs Conversions and ROI calculations.</em>",
        unsafe_allow_html=True,
    )

    df = leads_to_df()
    if df.empty:
        st.info("No leads")
        return

    total_spend = float(df.get("ad_cost", 0).sum())
    won_df = df[df["stage"] == "Won"] if "stage" in df.columns else pd.DataFrame()
    conversions = len(won_df)
    cpa = (total_spend / conversions) if conversions else 0.0
    revenue = float(won_df.get("estimated_value", 0).sum()) if not won_df.empty else 0.0
    roi = revenue - total_spend
    roi_pct = (roi / total_spend * 100) if total_spend else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Marketing Spend", f"${total_spend:,.2f}")
    c2.metric("Conversions (Won)", f"{conversions}")
    c3.metric("CPA", f"${cpa:,.2f}")
    c4.metric("ROI", f"${roi:,.2f} ({roi_pct:.1f}%)")


def page_ml_internal():
    st.markdown("<div class='header'> Internal ML — Lead Scoring</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Model runs internally and writes score back to leads. No user tuning exposed.</em>",
        unsafe_allow_html=True,
    )

    if st.button("Train model (internal)"):
        with st.spinner("Training..."):
            try:
                acc, msg = train_internal_model()
                if acc is None:
                    st.error(f"Training aborted: {msg}")
                else:
                    st.success(f"Model trained (accuracy approx): {acc:.3f}")
            except Exception as e:
                st.error("Training failed: " + str(e))

def page_ai_recommendations():
    require_role_access("business_intelligence")
    st.markdown("<div class='header'> AI Recommendations</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Heuristic recommendations and quick diagnostics for the pipeline.</em>",
        unsafe_allow_html=True,
    )

    try:
        df = leads_to_df()
    except Exception as e:
        st.error(f"Failed to load leads: {e}")
        df = pd.DataFrame()

    if df.empty:
        st.info("No leads to analyze.")
        return

    st.subheader("Top Overdue Leads")
    overdue_list = []
    for _, r in df.iterrows():
        rem_s, overdue_flag = calculate_remaining_sla(
            r.get("sla_entered_at") or r.get("created_at"),
            r.get("sla_hours"),
        )
        if overdue_flag and r.get("stage") not in ("Won", "Lost"):
            overdue_list.append(
                {
                    "lead_id": r.get("lead_id"),
                    "stage": r.get("stage"),
                    "assigned_to": r.get("assigned_to"),
                    "value": r.get("estimated_value") or 0.0,
                    "overdue_seconds": rem_s,
                }
            )

    over_df = pd.DataFrame(overdue_list)
    if not over_df.empty:
        over_df = over_df.sort_values("value", ascending=False)
        st.table(over_df[["lead_id", "stage", "assigned_to", "value"]].head(10))
    else:
        st.info("No overdue leads.")

pwd_context = CryptContext(
schemes=["bcrypt"],
deprecated="auto"
)
def hash_password(password: str) -> str:
    return pwd_context.hash(password)
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
# ----------------------
# WORDPRESS AUTH
# ----------------------
def reset_password_with_token(token: str, new_password: str):
    with SessionLocal() as s:
        user = s.query(User).filter(
    User.reset_token == token,
    User.reset_expires_at > pd.Timestamp.utcnow()
    ).first()
    if not user:
        raise Exception("Invalid or expired reset token")
    user.password_hash = hash_password(new_password)
    user.reset_token = None
    user.reset_expires_at = None
    user.email_verified = True
    s.commit()
def request_password_reset(email: str):
    with SessionLocal() as s:
        user = s.query(User).filter(User.email == email.lower()).first()
    if not user:
        return # Silent fail for security
    token = generate_password_reset_token()
    user.password_reset_token = token
    user.password_reset_expires_at = pd.Timestamp.utcnow() + timedelta(hours=1)
    s.commit()
    reset_link = f"{FRONTEND_URL}/reset-password?token={token}"
    send_password_reset_email(user.email, reset_link)
def send_password_reset_email(email: str, reset_link: str):
# TEMP: console output for testing
    print("PASSWORD RESET LINK:", reset_link)
# Later you can wire:
# SendGrid / Mailgun / SMTP / SES
#--------------------OPTIONAL PAGE RESET FOR STREAMLIT---------------
def page_reset_password():
    st.markdown("## Reset Your Password")

    token = st.query_params.get("token")
    if not token:
        st.error("Missing reset token")
        st.stop()

    pw1 = st.text_input("New Password", type="password")
    pw2 = st.text_input("Confirm Password", type="password")

    if st.button("Reset Password"):
        if not pw1 or len(pw1) < 8:
            st.error("Password must be at least 8 characters")
            st.stop()
        if pw1 != pw2:
            st.error("Passwords do not match")
            st.stop()

        try:
            reset_password_with_token(token, pw1)
            st.success("Password reset successful. You may now log in.")
            st.stop()
        except Exception as e:
            st.error(str(e))
def record_invoice(user, amount, description):
    with SessionLocal() as s:
        invoice = Invoice(
    user_id=user.id,
    amount=amount,
    status="paid",
    description=description,
    )
    s.add(invoice)
    s.commit()
def authenticate_user(email: str, password: str):
    with SessionLocal() as s:
        user = s.query(User).filter(
    User.email == email.lower(),
    User.is_active == True
    ).first()
    if not user:
        return None
    if not user.password_hash:
        return None
    # ---- Check if account is locked ----
    if user.locked_until and user.locked_until > pd.Timestamp.utcnow():
        raise Exception("Account locked. Try again later.")
    # ---- Verify password ----
    if not verify_password(password, user.password_hash):
        user.failed_login_attempts += 1
    if user.failed_login_attempts >= 5:
        user.locked_until = pd.Timestamp.utcnow() + timedelta(minutes=30)
    s.commit()
    raise Exception("Invalid credentials")
    # ---- SUCCESS: reset failed attempts ----
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = pd.Timestamp.utcnow()
    # Generate OTP for Admins or first-time login
    otp = generate_otp()
    user.otp_code = otp
    user.otp_expires_at = pd.Timestamp.utcnow() + timedelta(minutes=5)
    user.otp_required = True
    send_otp_email(user.email, otp)
    s.commit()
    # Mark user as pending OTP
    st.session_state["otp_user_id"] = user.id
    return "OTP_REQUIRED"
def request_password_reset(email: str):
    token = generate_reset_token()
    with SessionLocal() as s:
        user = s.query(User).filter(User.email == email).first()
        if not user:
            return  # silent fail (security)

        user.reset_token = token
        user.reset_expires_at = pd.Timestamp.utcnow() + timedelta(hours=1)
        s.commit()

    reset_link = f"{FRONTEND_URL}/reset-password?token={token}"
    send_password_reset_email(email, reset_link)


def admin_upgrade_user(user_id: int, plan: str):
    with SessionLocal() as s:
        user = s.query(User).filter(User.id == user_id).first()
        if not user:
            return False

        reactivate_user_account(user, plan)
        s.commit()
        return True

# -------------------------
# Settings Page
# -------------------------
def get_users_df():
    with SessionLocal() as s:
        rows = s.query(User).order_by(User.created_at.desc()).all()

    data = [
        {
            "id": u.id,
            "email": u.email,
            "username": u.username,
            "full_name": u.full_name,
            "role": u.role,
            "plan": u.plan,
            "is_active": bool(u.is_active),
            "email_verified": bool(u.email_verified),
            "created_at": u.created_at,
        }
        for u in rows
    ]
    return pd.DataFrame(data)


def page_technician_mobile():
    st.markdown("## Technician Mobile")
    techs = get_technicians_df(active_only=True)
    if techs.empty:
        st.warning("No active technicians found.")
        return

    tech = st.selectbox("Select Technician", techs["username"].tolist())
    st.markdown("### My Tasks")
    tasks = get_tasks_for_user(tech)
    if tasks.empty:
        st.info(
            "No task assigned to a Technician yet! To assign job task to a technician, go to:SETTINGS at the Navigation Menu, then click on the TECHNICIAN MANAGEMENT"
        )
    else:
        for _, t in tasks.iterrows():
            st.checkbox(
                f"{t['title']} (Lead: {t['lead_id']})",
                value=(t["status"] == "done"),
                key=f"task_{t['id']}",
            )

    st.markdown("### Send Location Ping")
    lat = st.number_input("Latitude", format="%.6f")
    lon = st.number_input("Longitude", format="%.6f")
    if st.button("Send Location Ping"):
        persist_location_ping(tech, lat, lon)
        st.success("Location sent")

# (loose technician block removed - managed inside page_settings)
#---------------------------Exports page--------------------------------------------
def add_time_windows(hist_df):
    """
Adds rolling time windows (3, 6, 12 months) for seasonal comparison
"""
    if hist_df.empty:
        return {}

    df = hist_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    latest_date = df["date"].max()
    windows = {
        "3_months": df[df["date"] >= latest_date - pd.DateOffset(months=3)],
        "6_months": df[df["date"] >= latest_date - pd.DateOffset(months=6)],
        "12_months": df[df["date"] >= latest_date - pd.DateOffset(months=12)],
    }
    return windows
# -------------------------------------------------------------
# SEASONAL TRENDS PAGE — SINGLE SOURCE OF TRUTH
# -------------------------------------------------------------
def page_seasonal_trends():
    require_role_access("business_intelligence")

    st.markdown("## Seasonal Trends & Weather-Based Damage Insights")
    st.caption("Location-based risk forecasting with business impact intelligence")
    st.divider()

    countries = get_all_countries()
    country = st.selectbox("Country", [c["name"] for c in countries])
    country_code = next(c["code"] for c in countries if c["name"] == country)

    city_query = st.text_input("City", placeholder="Type at least 3 letters")
    if len(city_query) < 3:
        st.info("Start typing a city name (3+ characters)")
        return

    matches = search_cities(country_code, city_query)
    if not matches:
        st.warning("No cities found")
        return

    labels = [f"{m['name']}, {m.get('admin1', '')}" for m in matches]
    selected = st.selectbox("Select City", labels)
    chosen = matches[labels.index(selected)]

    hist_range = st.selectbox("Historical Window", ["3 months", "6 months", "12 months"], index=1)
    months = {"3 months": 3, "6 months": 6, "12 months": 12}[hist_range]

    if not st.button("Generate Insights", use_container_width=True):
        return

    with st.spinner("Generating insights..."):
        hist_df = fetch_weather(chosen["lat"], chosen["lon"], months)

    if hist_df is None or hist_df.empty:
        st.error("No historical data available")
        return

    hist_df = hist_df.copy()
    hist_df["date"] = pd.to_datetime(hist_df["date"], errors="coerce")
    hist_df = hist_df.dropna(subset=["date"]).sort_values("date")

    st.line_chart(hist_df.set_index("date")[["rainfall_mm", "temperature_c"]])

def page_competitor_intelligence():
    st.title(" Competitor Intelligence")

    st.subheader(" Competitive Alerts")
    s = get_session()
    try:
        alerts = (
            s.query(CompetitorAlert)
            .order_by(CompetitorAlert.created_at.desc())
            .limit(5)
            .all()
        )
    finally:
        s.close()

    if not alerts:
        st.success("No competitive threats detected.")
    else:
        for a in alerts:
            if a.severity == "high":
                st.error(a.message)
            else:
                st.warning(a.message)

    st.divider()

    with st.expander(" Discover Competitors"):
        lat = st.number_input("Latitude", value=39.9612, key="comp_lat")
        lon = st.number_input("Longitude", value=-82.9988, key="comp_lon")
        keyword = st.text_input("Search keyword", "water damage restoration", key="comp_keyword")
        if st.button("Run Competitor Scan", key="run_comp_scan"):
            ingest_competitors_openstreetmap(lat, lon, keyword)
            st.success("Competitor scan completed.")

    s = get_session()
    try:
        competitors = s.query(Competitor).all()
    finally:
        s.close()

    if not competitors:
        st.info("No competitors tracked yet.")
        return

    rows = []
    hq_lat = st.session_state.get("hq_lat")
    hq_lon = st.session_state.get("hq_lon")
    for c in competitors:
        distance = 10
        if hq_lat and hq_lon and c.latitude and c.longitude:
            distance = haversine_km(hq_lat, hq_lon, c.latitude, c.longitude)
        score = calculate_competitor_score(c.rating or 0, c.total_reviews or 0, distance)
        rows.append(
            {
                "Name": c.name,
                "Rating": c.rating,
                "Reviews": c.total_reviews,
                "Category": c.primary_category,
                "Distance (km)": round(distance, 2),
                "Strength Score": score,
                "Velocity (7d)": review_velocity(c.id, 7),
                "Velocity (30d)": review_velocity(c.id, 30),
            }
        )

    df = pd.DataFrame(rows).sort_values("Strength Score", ascending=False)
    st.subheader("Top Competitors")
    st.dataframe(df, use_container_width=True, key="competitor_table")

def page_google_reviews():
    st.subheader(" Google Reviews")
    st.caption("Request Google reviews from completed jobs to boost reputation and local SEO.")

    st.subheader(" Google Review Link")
    review_link = st.text_input(
        "Paste your Google Review link",
        placeholder="https://g.page/your-business/review",
    )
    if st.button(" Save Review Link"):
        if not review_link:
            st.error("Review link is required")
        else:
            save_review_link_for_user(get_current_user(), review_link)
            st.success("Review link saved")

    st.divider()
    st.subheader(" Select Customer")
    contacts = get_completed_job_contacts(get_current_user())
    if not contacts:
        st.info("No completed job contacts yet.")
        return

    contact = st.selectbox(
        "Choose a customer",
        contacts,
        format_func=lambda c: f"{c['name']} ({c['email']})",
    )

    st.divider()
    st.subheader(" Send Review Request")
    if st.button("Send Google Review Request"):
        send_google_review_request(
            to_email=contact["email"],
            customer_name=contact["name"],
            review_link=review_link,
            job_name=contact.get("job_title", ""),
        )
        st.success("Review request sent")

def page_request_review():
    require_role_access("overview")
    st.markdown("## Request Google Review")
    st.caption("Instant on-site review request via Tap or QR")

    settings = get_user_settings_safe()
    review_url = settings.get("google_review_url")
    if not review_url:
        st.warning("Google Review link not set. Add it in Settings.")
        return

    lead_id = st.session_state.get("active_lead_id")
    token = generate_review_token()
    base_url = st.secrets.get("APP_BASE_URL", "http://localhost:8501")
    review_request_url = f"{base_url}/?page=review_redirect&token={token}"

    try:
        log_event(
            event_type="review_requested",
            entity_type="lead" if lead_id else "session",
            entity_id=lead_id,
            metadata={"method": "manual"},
        )
    except Exception:
        pass

    st.write("Share this review link with your customer:")
    st.code(review_request_url)

def page_review_redirect():
    token = st.query_params.get("token")
    settings = get_user_settings_safe()
    review_url = settings.get("google_review_url")
    if not review_url:
        st.error("Review link not configured.")
        return

    try:
        log_event(
            event_type="review_tap",
            entity_type="review",
            entity_id=token,
            metadata={"source": "nfc_or_qr"},
        )
    except Exception:
        pass

    st.markdown("Redirecting to review page…")
    st.markdown(
        f"<meta http-equiv='refresh' content='1;url={review_url}'>",
        unsafe_allow_html=True,
    )

#-----------------------START OF COMMAND CENTER---------------------
def get_user_settings_safe():
    """
Safe wrapper to prevent Request Review page crashes
"""
    try:
        return get_user_settings()
    except Exception:
        return {
            "google_review_url": "",
            "enable_nfc_review": True,
            "enable_qr_review": True,
        }


def generate_review_token():
    """
Generates a short-lived anonymous review token.
DB persistence can be added later.
"""
    return uuid.uuid4().hex

def log_event(
*,
event_type: str,
user_id: str | None = None,
org_id: str | None = None,
entity_type: str | None = None,
entity_id: str | None = None,
metadata: dict | None = None,
severity: str = "info",
source: str = "app"
):
    try:
        payload = {
        "id": str(uuid4()),
        "event_type": event_type,
        "metadata": json.dumps(metadata or {}),
        "created_at": pd.Timestamp.utcnow()
        }
        if user_id:
            payload.update({
        "user_id": user_id,
        "org_id": org_id,
        "entity_type": entity_type,
        "entity_id": entity_id
        })
        run_query(
        """
        INSERT INTO platform_user_events
        (id, user_id, org_id, event_type, entity_type, entity_id, metadata, created_at)
        VALUES (%(id)s, %(user_id)s, %(org_id)s, %(event_type)s, %(entity_type)s,
        %(entity_id)s, %(metadata)s, %(created_at)s)
        """,
        payload
        )

        payload.update({
        "severity": severity,
        "source": source
        })
        run_query(
        """
        INSERT INTO platform_events
        (id, event_type, severity, source, metadata, created_at)
        VALUES (%(id)s, %(event_type)s, %(severity)s, %(source)s, %(metadata)s,
        %(created_at)s)
        """,
        payload
        )
    except Exception:
    # NEVER crash the app because of logging
        pass
def analyze_job_types(df):
    if df.empty:
        return {}

    grouped = (
        df.groupby("damage_type")
        .agg(
            jobs=("id", "count"),
            revenue=("estimated_value", "sum"),
            avg_revenue=("estimated_value", "mean"),
            revenue_std=("estimated_value", "std"),
        )
        .reset_index()
    )

    total_jobs = grouped["jobs"].sum()
    total_revenue = grouped["revenue"].sum()
    grouped["job_share"] = grouped["jobs"] / total_jobs if total_jobs else 0
    grouped["revenue_share"] = grouped["revenue"] / total_revenue if total_revenue else 0
    grouped["volatility"] = grouped["revenue_std"].fillna(0)

    insights = []
    for _, r in grouped.iterrows():
        if r["job_share"] > 0.45:
            insights.append(f" Over-dependence on **{r['damage_type']}** jobs")
        if r["job_share"] > 0.3 and r["avg_revenue"] < grouped["avg_revenue"].mean():
            insights.append(f" **{r['damage_type']}** jobs are high-volume but low-value")
        if r["volatility"] > grouped["volatility"].mean() * 1.5:
            insights.append(f" **{r['damage_type']}** revenue is highly volatile")

    return {
        "table": grouped.sort_values("revenue", ascending=False),
        "insights": insights,
    }


def pct_change(current, previous):
    """
Safe percentage change calculation.
Returns 0 if previous is zero or missing.
"""
    try:
        if previous in (0, None):
            return 0
        return ((current - previous) / previous) * 100
    except Exception:
        return 0


def generate_executive_narrative(data):
    narrative = []
    risk_flags = []
    volume = data.get("volume", {})
    revenue = data.get("revenue", {})
    efficiency = data.get("efficiency", {})

    narrative.append(
        {
            "text": (
                f"Total jobs: {volume.get('total_jobs', 0)} | "
                f"Revenue: ${revenue.get('total_revenue', 0):,.0f} | "
                f"Revenue/job: ${efficiency.get('revenue_per_job', 0):,.0f}"
            ),
            "confidence": 85,
        }
    )

    health_score = 82
    return {
        "lines": narrative,
        "risk_flags": risk_flags,
        "health_score": health_score,
        "version": "G-hardened-v1",
    }

def page_business_intelligence():
    require_role_access("business_intelligence")

    st.markdown("## Business Intelligence")

    col1, col2 = st.columns([2, 3])
    with col1:
        range_key = st.selectbox(
            "Time Range",
            ["daily", "weekly", "30d", "90d", "6m", "12m", "custom"],
            index=2,
        )
    with col2:
        custom_start, custom_end = None, None
        if range_key == "custom":
            custom_start = st.date_input("Start date")
            custom_end = st.date_input("End date")

    data = compute_business_intelligence(range_key, custom_start, custom_end)
    df = data.get("raw_df")
    if df is None or df.empty:
        st.warning("No jobs found for this period.")
        return

    health = data.get("health")
    if health:
        st.metric("Overall Business Health", f"{health.get('score', 0)} / 100")

    v = data.get("volume", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Jobs", v.get("total_jobs", 0))
    c2.metric("Jobs / Day", v.get("jobs_per_day", 0))
    c3.metric("Lead Sources", len(v.get("lead_sources", {})))

    r = data.get("revenue", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${r.get('total_revenue', 0):,.0f}")
    c2.metric("Avg / Job", f"${r.get('avg_revenue_per_job', 0):,.0f}")
    c3.metric("Revenue Risk", f"{int(r.get('revenue_concentration', 0) * 100)}% Top Dependency")

def page_technician_map_tracking():
    st.markdown("## Technician Live Map")
    df = get_latest_location_pings()
    if df.empty:
        st.warning("No technician GPS data available.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    def classify_status(ts):
        if pd.isna(ts):
            return "offline"
        mins = (pd.Timestamp.utcnow() - ts).total_seconds() / 60
        if mins <= 10:
            return "active"
        if mins <= 30:
            return "idle"
        return "offline"

    df["status"] = df["timestamp"].apply(classify_status)
    status_color = {"active": "green", "idle": "orange", "offline": "red"}

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=8,
            color=status_color.get(r["status"], "gray"),
            fill=True,
            fill_opacity=0.85,
            popup=f"{r.get('tech_username', 'tech')} ({r['status']})",
        ).add_to(m)

    st_folium(m, width=900, height=500)


def page_technician_mobile():
    st.markdown("## Technician Mobile")
    st.info("Use the mobile endpoint to push technician GPS pings.")

def page_cpa_roi():
    require_role_access("analytics")
    st.markdown("<div class='header'> CPA & ROI</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Total Marketing Spend vs Conversions and ROI calculations.</em>",
        unsafe_allow_html=True,
    )

    df = leads_to_df()
    if df.empty:
        st.info("No leads")
        return

    total_spend = float(df.get("ad_cost", 0).sum())
    won_df = df[df["stage"] == "Won"] if "stage" in df.columns else pd.DataFrame()
    conversions = len(won_df)
    cpa = (total_spend / conversions) if conversions else 0.0
    revenue = float(won_df.get("estimated_value", 0).sum()) if not won_df.empty else 0.0
    roi = revenue - total_spend
    roi_pct = (roi / total_spend * 100) if total_spend else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Marketing Spend", f"${total_spend:,.2f}")
    c2.metric("Conversions (Won)", f"{conversions}")
    c3.metric("CPA", f"${cpa:,.2f}")
    c4.metric("ROI", f"${roi:,.2f} ({roi_pct:.1f}%)")


def page_ml_internal():
    st.markdown("<div class='header'> Internal ML — Lead Scoring</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Model runs internally and writes score back to leads. No user tuning exposed.</em>",
        unsafe_allow_html=True,
    )

    if st.button("Train model (internal)"):
        with st.spinner("Training..."):
            try:
                acc, msg = train_internal_model()
                if acc is None:
                    st.error(f"Training aborted: {msg}")
                else:
                    st.success(f"Model trained (accuracy approx): {acc:.3f}")
            except Exception as e:
                st.error("Training failed: " + str(e))

def page_ai_recommendations():
    require_role_access("business_intelligence")
    st.markdown("<div class='header'> AI Recommendations</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Heuristic recommendations and quick diagnostics for the pipeline.</em>",
        unsafe_allow_html=True,
    )

    try:
        df = leads_to_df()
    except Exception as e:
        st.error(f"Failed to load leads: {e}")
        df = pd.DataFrame()

    if df.empty:
        st.info("No leads to analyze.")
        return

    st.subheader("Top Overdue Leads")
    overdue_list = []
    for _, r in df.iterrows():
        rem_s, overdue_flag = calculate_remaining_sla(
            r.get("sla_entered_at") or r.get("created_at"),
            r.get("sla_hours"),
        )
        if overdue_flag and r.get("stage") not in ("Won", "Lost"):
            overdue_list.append(
                {
                    "lead_id": r.get("lead_id"),
                    "stage": r.get("stage"),
                    "assigned_to": r.get("assigned_to"),
                    "value": r.get("estimated_value") or 0.0,
                    "overdue_seconds": rem_s,
                }
            )

    over_df = pd.DataFrame(overdue_list)
    if not over_df.empty:
        over_df = over_df.sort_values("value", ascending=False)
        st.table(over_df[["lead_id", "stage", "assigned_to", "value"]].head(10))
    else:
        st.info("No overdue leads.")

pwd_context = CryptContext(
schemes=["bcrypt"],
deprecated="auto"
)
def hash_password(password: str) -> str:
    return pwd_context.hash(password)
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
# ----------------------
# WORDPRESS AUTH
# ----------------------
def reset_password_with_token(token: str, new_password: str):
    with SessionLocal() as s:
        user = s.query(User).filter(
    User.reset_token == token,
    User.reset_expires_at > pd.Timestamp.utcnow()
    ).first()
    if not user:
        raise Exception("Invalid or expired reset token")
    user.password_hash = hash_password(new_password)
    user.reset_token = None
    user.reset_expires_at = None
    user.email_verified = True
    s.commit()
def request_password_reset(email: str):
    with SessionLocal() as s:
        user = s.query(User).filter(User.email == email.lower()).first()
    if not user:
        return # Silent fail for security
    token = generate_password_reset_token()
    user.password_reset_token = token
    user.password_reset_expires_at = pd.Timestamp.utcnow() + timedelta(hours=1)
    s.commit()
    reset_link = f"{FRONTEND_URL}/reset-password?token={token}"
    send_password_reset_email(user.email, reset_link)
def send_password_reset_email(email: str, reset_link: str):
# TEMP: console output for testing
    print("PASSWORD RESET LINK:", reset_link)
# Later you can wire:
# SendGrid / Mailgun / SMTP / SES
#--------------------OPTIONAL PAGE RESET FOR STREAMLIT---------------
def page_reset_password():
    st.markdown("## Reset Your Password")

    token = st.query_params.get("token")
    if not token:
        st.error("Missing reset token")
        st.stop()

    pw1 = st.text_input("New Password", type="password")
    pw2 = st.text_input("Confirm Password", type="password")

    if st.button("Reset Password"):
        if not pw1 or len(pw1) < 8:
            st.error("Password must be at least 8 characters")
            st.stop()
        if pw1 != pw2:
            st.error("Passwords do not match")
            st.stop()

        try:
            reset_password_with_token(token, pw1)
            st.success("Password reset successful. You may now log in.")
            st.stop()
        except Exception as e:
            st.error(str(e))
def record_invoice(user, amount, description):
    with SessionLocal() as s:
        invoice = Invoice(
    user_id=user.id,
    amount=amount,
    status="paid",
    description=description,
    )
    s.add(invoice)
    s.commit()
def authenticate_user(email: str, password: str):
    with SessionLocal() as s:
        user = s.query(User).filter(
    User.email == email.lower(),
    User.is_active == True
    ).first()
    if not user:
        return None
    if not user.password_hash:
        return None
    # ---- Check if account is locked ----
    if user.locked_until and user.locked_until > pd.Timestamp.utcnow():
        raise Exception("Account locked. Try again later.")
    # ---- Verify password ----
    if not verify_password(password, user.password_hash):
        user.failed_login_attempts += 1
    if user.failed_login_attempts >= 5:
        user.locked_until = pd.Timestamp.utcnow() + timedelta(minutes=30)
    s.commit()
    raise Exception("Invalid credentials")
    # ---- SUCCESS: reset failed attempts ----
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = pd.Timestamp.utcnow()
    # Generate OTP for Admins or first-time login
    otp = generate_otp()
    user.otp_code = otp
    user.otp_expires_at = pd.Timestamp.utcnow() + timedelta(minutes=5)
    user.otp_required = True
    send_otp_email(user.email, otp)
    s.commit()
    # Mark user as pending OTP
    st.session_state["otp_user_id"] = user.id
    return "OTP_REQUIRED"
def request_password_reset(email: str):
    token = generate_reset_token()
    with SessionLocal() as s:
        user = s.query(User).filter(User.email == email).first()
        if not user:
            return  # silent fail (security)

        user.reset_token = token
        user.reset_expires_at = pd.Timestamp.utcnow() + timedelta(hours=1)
        s.commit()

    reset_link = f"{FRONTEND_URL}/reset-password?token={token}"
    send_password_reset_email(email, reset_link)


def admin_upgrade_user(user_id: int, plan: str):
    with SessionLocal() as s:
        user = s.query(User).filter(User.id == user_id).first()
        if not user:
            return False

        reactivate_user_account(user, plan)
        s.commit()
        return True

# -------------------------
# Settings Page
# -------------------------
def get_users_df():
    with SessionLocal() as s:
        rows = s.query(User).order_by(User.created_at.desc()).all()

    data = [
        {
            "id": u.id,
            "email": u.email,
            "username": u.username,
            "full_name": u.full_name,
            "role": u.role,
            "plan": u.plan,
            "is_active": bool(u.is_active),
            "email_verified": bool(u.email_verified),
            "created_at": u.created_at,
        }
        for u in rows
    ]
    return pd.DataFrame(data)


def is_company_email(email: str) -> bool:
    try:
        domain = email.split("@", 1)[1].lower().strip()
    except Exception:
        return False
    return domain not in FREE_EMAIL_DOMAINS


def upsert_profile_from_auth(user_obj):
    if not user_obj:
        return
    try:
        full_name = ""
        if getattr(user_obj, "user_metadata", None):
            full_name = user_obj.user_metadata.get("full_name", "")
        supabase.table("profiles").upsert(
            {
                "id": user_obj.id,
                "email": user_obj.email,
                "full_name": full_name,
            },
            on_conflict="id",
        ).execute()
    except Exception:
        # If DB trigger is configured this is not required; keep as safe fallback.
        pass


def set_auth_session(user_obj):
    st.session_state["authenticated"] = True
    st.session_state["supabase_user_id"] = user_obj.id
    st.session_state["user_email"] = user_obj.email
    st.session_state["page"] = "command_center"

    profile_payload = {
        "id": user_obj.id,
        "email": user_obj.email,
        "full_name": "",
        "role": "Technician",
        "company_id": None,
    }
    try:
        profile_res = (
            supabase.table("profiles")
            .select("id,email,full_name,role,company_id")
            .eq("id", user_obj.id)
            .limit(1)
            .execute()
        )
        if profile_res.data:
            profile_payload.update(profile_res.data[0])
    except Exception:
        pass

    st.session_state["user"] = {
        "id": profile_payload.get("id") or user_obj.id,
        "email": profile_payload.get("email") or user_obj.email,
        "full_name": profile_payload.get("full_name") or "",
        "role": profile_payload.get("role") or "Technician",
        "company_id": profile_payload.get("company_id"),
    }


def page_auth():
    st.title("Welcome to ReCapture Pro")
    st.caption("Login or create an account with your company email.")

    tab_login, tab_signup = st.tabs(["Login", "Sign up"])

    with tab_login:
        login_email = st.text_input("Email", key="auth_login_email")
        login_pw = st.text_input("Password", type="password", key="auth_login_pw")
        if st.button("Log in", key="auth_login_btn"):
            try:
                result = supabase.auth.sign_in_with_password(
                    {"email": login_email.strip().lower(), "password": login_pw}
                )
                if result and result.user:
                    set_auth_session(result.user)
                    st.success("Logged in successfully")
                    st.rerun()
                else:
                    st.error("Login failed")
            except Exception as e:
                st.error(f"Login failed: {e}")

    with tab_signup:
        signup_name = st.text_input("Full Name", key="auth_signup_name")
        signup_email = st.text_input("Company Email", key="auth_signup_email")
        signup_pw = st.text_input("Create Password", type="password", key="auth_signup_pw")
        if st.button("Create account", key="auth_signup_btn"):
            email_norm = signup_email.strip().lower()
            if not is_valid_email(email_norm):
                st.error("Please enter a valid email address.")
                return
            if not is_company_email(email_norm):
                st.error("Please use your company email (personal domains are not allowed).")
                return
            if len(signup_pw) < 8:
                st.error("Password must be at least 8 characters.")
                return

            try:
                result = supabase.auth.sign_up(
                    {
                        "email": email_norm,
                        "password": signup_pw,
                        "options": {"data": {"full_name": signup_name}},
                    }
                )
                if result and result.user:
                    upsert_profile_from_auth(result.user)
                    set_auth_session(result.user)
                    st.success("Signup successful. Check your email to confirm your account.")
                    st.rerun()
                else:
                    st.info("Signup request submitted. Check your email for confirmation.")
            except Exception as e:
                st.error(f"Signup failed: {e}")


# Prefer existing Supabase session when available
try:
    sess = supabase.auth.get_session()
    user_obj = sess.session.user if sess and sess.session else None
    if user_obj and "authenticated" not in st.session_state:
        set_auth_session(user_obj)
except Exception:
    pass

# Keep WordPress token bridge compatibility when token is explicitly present
if "token" in st.query_params and not st.session_state.get("authenticated"):
    wp_auth_bridge()
    st.stop()

if not st.session_state.get("authenticated"):
    page_auth()
    st.stop()

user = get_current_user()



def get_user_settings_safe():
    """
Safe wrapper to prevent Request Review page crashes
"""
    try:
        return get_user_settings()
    except Exception:
        return {
            "google_review_url": "",
            "enable_nfc_review": True,
            "enable_qr_review": True,
        }


def generate_review_token():
    """
Generates a short-lived anonymous review token.
DB persistence can be added later.
"""
    return uuid.uuid4().hex

def log_event(
*,
event_type: str,
user_id: str | None = None,
org_id: str | None = None,
entity_type: str | None = None,
entity_id: str | None = None,
metadata: dict | None = None,
severity: str = "info",
source: str = "app"
):
    try:
        payload = {
        "id": str(uuid4()),
        "event_type": event_type,
        "metadata": json.dumps(metadata or {}),
        "created_at": pd.Timestamp.utcnow()
        }
        if user_id:
            payload.update({
        "user_id": user_id,
        "org_id": org_id,
        "entity_type": entity_type,
        "entity_id": entity_id
        })
        run_query(
        """
        INSERT INTO platform_user_events
        (id, user_id, org_id, event_type, entity_type, entity_id, metadata, created_at)
        VALUES (%(id)s, %(user_id)s, %(org_id)s, %(event_type)s, %(entity_type)s,
        %(entity_id)s, %(metadata)s, %(created_at)s)
        """,
        payload
        )

        payload.update({
        "severity": severity,
        "source": source
        })
        run_query(
        """
        INSERT INTO platform_events
        (id, event_type, severity, source, metadata, created_at)
        VALUES (%(id)s, %(event_type)s, %(severity)s, %(source)s, %(metadata)s,
        %(created_at)s)
        """,
        payload
        )
    except Exception:
    # NEVER crash the app because of logging
        pass
def analyze_job_types(df):
    if df.empty:
        return {}

    grouped = (
        df.groupby("damage_type")
        .agg(
            jobs=("id", "count"),
            revenue=("estimated_value", "sum"),
            avg_revenue=("estimated_value", "mean"),
            revenue_std=("estimated_value", "std"),
        )
        .reset_index()
    )

    total_jobs = grouped["jobs"].sum()
    total_revenue = grouped["revenue"].sum()
    grouped["job_share"] = grouped["jobs"] / total_jobs if total_jobs else 0
    grouped["revenue_share"] = grouped["revenue"] / total_revenue if total_revenue else 0
    grouped["volatility"] = grouped["revenue_std"].fillna(0)

    insights = []
    for _, r in grouped.iterrows():
        if r["job_share"] > 0.45:
            insights.append(f" Over-dependence on **{r['damage_type']}** jobs")
        if r["job_share"] > 0.3 and r["avg_revenue"] < grouped["avg_revenue"].mean():
            insights.append(f" **{r['damage_type']}** jobs are high-volume but low-value")
        if r["volatility"] > grouped["volatility"].mean() * 1.5:
            insights.append(f" **{r['damage_type']}** revenue is highly volatile")

    return {
        "table": grouped.sort_values("revenue", ascending=False),
        "insights": insights,
    }


def pct_change(current, previous):
    """
Safe percentage change calculation.
Returns 0 if previous is zero or missing.
"""
    try:
        if previous in (0, None):
            return 0
        return ((current - previous) / previous) * 100
    except Exception:
        return 0


def generate_executive_narrative(data):
    narrative = []
    risk_flags = []
    volume = data.get("volume", {})
    revenue = data.get("revenue", {})
    efficiency = data.get("efficiency", {})

    narrative.append(
        {
            "text": (
                f"Total jobs: {volume.get('total_jobs', 0)} | "
                f"Revenue: ${revenue.get('total_revenue', 0):,.0f} | "
                f"Revenue/job: ${efficiency.get('revenue_per_job', 0):,.0f}"
            ),
            "confidence": 85,
        }
    )

    health_score = 82
    return {
        "lines": narrative,
        "risk_flags": risk_flags,
        "health_score": health_score,
        "version": "G-hardened-v1",
    }

def page_business_intelligence():
    require_role_access("business_intelligence")

    st.markdown("## Business Intelligence")

    col1, col2 = st.columns([2, 3])
    with col1:
        range_key = st.selectbox(
            "Time Range",
            ["daily", "weekly", "30d", "90d", "6m", "12m", "custom"],
            index=2,
        )
    with col2:
        custom_start, custom_end = None, None
        if range_key == "custom":
            custom_start = st.date_input("Start date")
            custom_end = st.date_input("End date")

    data = compute_business_intelligence(range_key, custom_start, custom_end)
    df = data.get("raw_df")
    if df is None or df.empty:
        st.warning("No jobs found for this period.")
        return

    health = data.get("health")
    if health:
        st.metric("Overall Business Health", f"{health.get('score', 0)} / 100")

    v = data.get("volume", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Jobs", v.get("total_jobs", 0))
    c2.metric("Jobs / Day", v.get("jobs_per_day", 0))
    c3.metric("Lead Sources", len(v.get("lead_sources", {})))

    r = data.get("revenue", {})
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${r.get('total_revenue', 0):,.0f}")
    c2.metric("Avg / Job", f"${r.get('avg_revenue_per_job', 0):,.0f}")
    c3.metric("Revenue Risk", f"{int(r.get('revenue_concentration', 0) * 100)}% Top Dependency")

def page_technician_map_tracking():
    st.markdown("## Technician Live Map")
    df = get_latest_location_pings()
    if df.empty:
        st.warning("No technician GPS data available.")
        return

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    def classify_status(ts):
        if pd.isna(ts):
            return "offline"
        mins = (pd.Timestamp.utcnow() - ts).total_seconds() / 60
        if mins <= 10:
            return "active"
        if mins <= 30:
            return "idle"
        return "offline"

    df["status"] = df["timestamp"].apply(classify_status)
    status_color = {"active": "green", "idle": "orange", "offline": "red"}

    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()
    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles="OpenStreetMap")

    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=8,
            color=status_color.get(r["status"], "gray"),
            fill=True,
            fill_opacity=0.85,
            popup=f"{r.get('tech_username', 'tech')} ({r['status']})",
        ).add_to(m)

    st_folium(m, width=900, height=500)


def page_technician_mobile():
    st.markdown("## Technician Mobile")
    st.info("Use the mobile endpoint to push technician GPS pings.")

def page_cpa_roi():
    require_role_access("analytics")
    st.markdown("<div class='header'> CPA & ROI</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Total Marketing Spend vs Conversions and ROI calculations.</em>",
        unsafe_allow_html=True,
    )

    df = leads_to_df()
    if df.empty:
        st.info("No leads")
        return

    total_spend = float(df.get("ad_cost", 0).sum())
    won_df = df[df["stage"] == "Won"] if "stage" in df.columns else pd.DataFrame()
    conversions = len(won_df)
    cpa = (total_spend / conversions) if conversions else 0.0
    revenue = float(won_df.get("estimated_value", 0).sum()) if not won_df.empty else 0.0
    roi = revenue - total_spend
    roi_pct = (roi / total_spend * 100) if total_spend else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Marketing Spend", f"${total_spend:,.2f}")
    c2.metric("Conversions (Won)", f"{conversions}")
    c3.metric("CPA", f"${cpa:,.2f}")
    c4.metric("ROI", f"${roi:,.2f} ({roi_pct:.1f}%)")


def page_ml_internal():
    st.markdown("<div class='header'> Internal ML — Lead Scoring</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Model runs internally and writes score back to leads. No user tuning exposed.</em>",
        unsafe_allow_html=True,
    )

    if st.button("Train model (internal)"):
        with st.spinner("Training..."):
            try:
                acc, msg = train_internal_model()
                if acc is None:
                    st.error(f"Training aborted: {msg}")
                else:
                    st.success(f"Model trained (accuracy approx): {acc:.3f}")
            except Exception as e:
                st.error("Training failed: " + str(e))

def page_ai_recommendations():
    require_role_access("business_intelligence")
    st.markdown("<div class='header'> AI Recommendations</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Heuristic recommendations and quick diagnostics for the pipeline.</em>",
        unsafe_allow_html=True,
    )

    try:
        df = leads_to_df()
    except Exception as e:
        st.error(f"Failed to load leads: {e}")
        df = pd.DataFrame()

    if df.empty:
        st.info("No leads to analyze.")
        return

    st.subheader("Top Overdue Leads")
    overdue_list = []
    for _, r in df.iterrows():
        rem_s, overdue_flag = calculate_remaining_sla(
            r.get("sla_entered_at") or r.get("created_at"),
            r.get("sla_hours"),
        )
        if overdue_flag and r.get("stage") not in ("Won", "Lost"):
            overdue_list.append(
                {
                    "lead_id": r.get("lead_id"),
                    "stage": r.get("stage"),
                    "assigned_to": r.get("assigned_to"),
                    "value": r.get("estimated_value") or 0.0,
                    "overdue_seconds": rem_s,
                }
            )

    over_df = pd.DataFrame(overdue_list)
    if not over_df.empty:
        over_df = over_df.sort_values("value", ascending=False)
        st.table(over_df[["lead_id", "stage", "assigned_to", "value"]].head(10))
    else:
        st.info("No overdue leads.")

pwd_context = CryptContext(
schemes=["bcrypt"],
deprecated="auto"
)
def hash_password(password: str) -> str:
    return pwd_context.hash(password)
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)
# ----------------------
# WORDPRESS AUTH
# ----------------------
def wp_auth_bridge():
    st.title("Authenticating...")

    token = st.query_params.get("token")
    if not token:
        st.error("Missing authentication token")
        st.stop()

    payload = decode_wp_token(token)
    if not payload:
        st.error("Invalid or expired token")
        st.stop()

    email = payload.get("email")
    name = payload.get("name", "")
    role = payload.get("role", "Viewer")
    if not email:
        st.error("Invalid token payload")
        st.stop()

    with SessionLocal() as s:
        user = s.query(User).filter(User.email == email).first()

        if not user:
            user = User(
                email=email.lower(),
                username=email.split("@")[0],
                full_name=name,
                role=role,
                plan="starter",
                subscription_status="trial",
                trial_ends_at=pd.Timestamp.utcnow() + timedelta(days=14),
                email_verified=True,
                is_active=True,
            )
            s.add(user)
            s.commit()
            s.refresh(user)

        if not user.is_active:
            st.error("Your account has been deactivated.")
            st.stop()
        if not user.email_verified:
            st.error("Please verify your email address before accessing the app.")
            st.stop()

        st.session_state["user_id"] = user.id
        st.session_state["authenticated"] = True
        st.session_state["user_email"] = user.email
        st.session_state["role"] = user.role
        st.session_state["plan"] = user.plan

    st.success("Authentication successful")
    st.query_params.clear()
    st.rerun()

def reset_password_with_token(token: str, new_password: str):
    with SessionLocal() as s:
        user = s.query(User).filter(
    User.reset_token == token,
    User.reset_expires_at > pd.Timestamp.utcnow()
    ).first()
    if not user:
        raise Exception("Invalid or expired reset token")
    user.password_hash = hash_password(new_password)
    user.reset_token = None
    user.reset_expires_at = None
    user.email_verified = True
    s.commit()
def request_password_reset(email: str):
    with SessionLocal() as s:
        user = s.query(User).filter(User.email == email.lower()).first()
    if not user:
        return # Silent fail for security
    token = generate_password_reset_token()
    user.password_reset_token = token
    user.password_reset_expires_at = pd.Timestamp.utcnow() + timedelta(hours=1)
    s.commit()
    reset_link = f"{FRONTEND_URL}/reset-password?token={token}"
    send_password_reset_email(user.email, reset_link)
def send_password_reset_email(email: str, reset_link: str):
# TEMP: console output for testing
    print("PASSWORD RESET LINK:", reset_link)
# Later you can wire:
# SendGrid / Mailgun / SMTP / SES
#--------------------OPTIONAL PAGE RESET FOR STREAMLIT---------------
def page_reset_password():
    st.markdown("## Reset Your Password")

    token = st.query_params.get("token")
    if not token:
        st.error("Missing reset token")
        st.stop()

    pw1 = st.text_input("New Password", type="password")
    pw2 = st.text_input("Confirm Password", type="password")

    if st.button("Reset Password"):
        if not pw1 or len(pw1) < 8:
            st.error("Password must be at least 8 characters")
            st.stop()
        if pw1 != pw2:
            st.error("Passwords do not match")
            st.stop()

        try:
            reset_password_with_token(token, pw1)
            st.success("Password reset successful. You may now log in.")
            st.stop()
        except Exception as e:
            st.error(str(e))
def record_invoice(user, amount, description):
    with SessionLocal() as s:
        invoice = Invoice(
    user_id=user.id,
    amount=amount,
    status="paid",
    description=description,
    )
    s.add(invoice)
    s.commit()
def authenticate_user(email: str, password: str):
    with SessionLocal() as s:
        user = s.query(User).filter(
    User.email == email.lower(),
    User.is_active == True
    ).first()
    if not user:
        return None
    if not user.password_hash:
        return None
    # ---- Check if account is locked ----
    if user.locked_until and user.locked_until > pd.Timestamp.utcnow():
        raise Exception("Account locked. Try again later.")
    # ---- Verify password ----
    if not verify_password(password, user.password_hash):
        user.failed_login_attempts += 1
    if user.failed_login_attempts >= 5:
        user.locked_until = pd.Timestamp.utcnow() + timedelta(minutes=30)
    s.commit()
    raise Exception("Invalid credentials")
    # ---- SUCCESS: reset failed attempts ----
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = pd.Timestamp.utcnow()
    # Generate OTP for Admins or first-time login
    otp = generate_otp()
    user.otp_code = otp
    user.otp_expires_at = pd.Timestamp.utcnow() + timedelta(minutes=5)
    user.otp_required = True
    send_otp_email(user.email, otp)
    s.commit()
    # Mark user as pending OTP
    st.session_state["otp_user_id"] = user.id
    return "OTP_REQUIRED"
def page_login():
    st.markdown("## Login")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        result = authenticate_user(email, password)
        if result == "OTP_REQUIRED":
            st.info("OTP sent to your email.")
            st.rerun()
        elif not result:
            st.error("Invalid credentials or inactive account")
        else:
            set_logged_in_user(result)
            st.success("Logged in successfully")
            st.rerun()


def request_password_reset(email: str):
    token = generate_reset_token()
    with SessionLocal() as s:
        user = s.query(User).filter(User.email == email).first()
        if not user:
            return  # silent fail (security)

        user.reset_token = token
        user.reset_expires_at = pd.Timestamp.utcnow() + timedelta(hours=1)
        s.commit()

    reset_link = f"{FRONTEND_URL}/reset-password?token={token}"
    send_password_reset_email(email, reset_link)


def admin_upgrade_user(user_id: int, plan: str):
    with SessionLocal() as s:
        user = s.query(User).filter(User.id == user_id).first()
        if not user:
            return False

        reactivate_user_account(user, plan)
        s.commit()
        return True

# -------------------------
# Settings Page
# -------------------------
def get_users_df():
    with SessionLocal() as s:
        rows = s.query(User).order_by(User.created_at.desc()).all()

    data = [
        {
            "id": u.id,
            "email": u.email,
            "username": u.username,
            "full_name": u.full_name,
            "role": u.role,
            "plan": u.plan,
            "is_active": bool(u.is_active),
            "email_verified": bool(u.email_verified),
            "created_at": u.created_at,
        }
        for u in rows
    ]
    return pd.DataFrame(data)


def page_settings():
    require_role_access("settings")
    if st.session_state.get("user") is None:
        st.session_state["page"] = "auth"
        st.warning("Please log in to access Settings.")
        st.rerun()

    session_user = st.session_state.get("user", {})
    user_role = session_user.get("role")
    company_id = session_user.get("company_id")

    st.markdown(
        "<div class='header'> Settings & User Management</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<em>Add team users, invite users, manage roles, billing and technicians.</em>",
        unsafe_allow_html=True,
    )

    tab_team, tab_invite, tab_roles = st.tabs(
        ["👥 Team Members", "✉️ Invite", "🔐 Roles & Permissions"]
    )

    with tab_team:
        if user_role != "Admin":
            st.warning("Only Admin users can access this tab.")
        else:
            response = (
                supabase.table("profiles")
                .select("id,full_name,email,role,is_active,company_id")
                .eq("company_id", company_id)
                .order("full_name")
                .execute()
            )
            members = response.data or []
            if not members:
                st.info("No team members found for your company.")
            else:
                headers = st.columns([2, 2, 1.4, 1.2])
                headers[0].markdown("**Full Name**")
                headers[1].markdown("**Email**")
                headers[2].markdown("**Role**")
                headers[3].markdown("**Status (Active/Inactive)**")

                role_options = ["Admin", "Manager", "Technician"]
                for member in members:
                    cols = st.columns([2, 2, 1.4, 1.2])
                    cols[0].write(member.get("full_name") or "—")
                    cols[1].write(member.get("email") or "—")
                    current_role = member.get("role") if member.get("role") in role_options else "Technician"

                    new_role = cols[2].selectbox(
                        "Role",
                        role_options,
                        index=role_options.index(current_role),
                        key=f"settings_role_{member['id']}",
                        label_visibility="collapsed",
                    )
                    is_active = cols[3].toggle(
                        "Active",
                        value=bool(member.get("is_active", True)),
                        key=f"settings_active_{member['id']}",
                        label_visibility="collapsed",
                    )

                    updates = {}
                    if new_role != current_role:
                        updates["role"] = new_role
                    if is_active != bool(member.get("is_active", True)):
                        updates["is_active"] = is_active
                    if updates:
                        supabase.table("profiles").update(updates).eq("id", member["id"]).execute()
                        st.rerun()

    with tab_invite:
        if user_role not in ["Admin", "Manager"]:
            st.warning("Only Admin and Manager users can access this tab.")
        else:
            with st.form("settings_invite_form"):
                invite_email = st.text_input("Email (required)")
                invite_role = st.selectbox("Role", ["Admin", "Manager", "Technician"])
                submit_invite = st.form_submit_button("Send Invite")

            if submit_invite:
                if not invite_email or not is_valid_email(invite_email):
                    st.error("Please enter a valid email address.")
                else:
                    token = secrets.token_urlsafe(32)
                    expires_at = (datetime.utcnow() + timedelta(hours=48)).isoformat()
                    supabase.table("invitations").insert(
                        {
                            "email": invite_email.strip().lower(),
                            "role": invite_role,
                            "company_id": company_id,
                            "token": token,
                            "expires_at": expires_at,
                            "used": False,
                        }
                    ).execute()
                    supabase.auth.invite_user_by_email(invite_email.strip().lower())
                    st.success("Invitation sent.")
                    st.rerun()

            st.markdown("### Pending Invites")
            pending = (
                supabase.table("invitations")
                .select("id,email,role,expires_at")
                .eq("company_id", company_id)
                .eq("used", False)
                .order("expires_at")
                .execute()
            )
            pending_rows = pending.data or []
            if not pending_rows:
                st.info("No pending invites.")
            else:
                for invite in pending_rows:
                    row = st.columns([2, 1.4, 1.8, 1])
                    row[0].write(invite.get("email") or "—")
                    row[1].write(invite.get("role") or "—")
                    row[2].write(invite.get("expires_at") or "—")
                    if row[3].button("Cancel", key=f"cancel_inv_{invite['id']}"):
                        supabase.table("invitations").delete().eq("id", invite["id"]).execute()
                        st.rerun()

    with tab_roles:
        permissions_df = pd.DataFrame(
            [
                {"Feature": "View all jobs", "Admin": "✅", "Manager": "✅", "Technician": "❌"},
                {"Feature": "Manage team", "Admin": "✅", "Manager": "✅", "Technician": "❌"},
                {"Feature": "Assign jobs", "Admin": "✅", "Manager": "✅", "Technician": "❌"},
                {"Feature": "View own jobs", "Admin": "✅", "Manager": "✅", "Technician": "✅"},
                {"Feature": "Update job status", "Admin": "✅", "Manager": "✅", "Technician": "✅"},
                {"Feature": "Access billing", "Admin": "✅", "Manager": "❌", "Technician": "❌"},
                {"Feature": "Access settings", "Admin": "✅", "Manager": "❌", "Technician": "❌"},
            ]
        )
        st.table(permissions_df)

    st.markdown("---")
    # ======================================================
    # ADMIN — TRIAL REMINDERS
    # ======================================================
    user = get_current_user()
    if user.role == "Admin":
        st.markdown("### Trial Management")
        if st.button(" Send Trial Reminder Emails (Admin)"):
            try:
                send_trial_expiry_reminders()
                st.success("Trial reminders sent")
            except Exception as e:
                if DEV_MODE:
                    st.warning(f"Trial reminder skipped (dev): {e}")
            else:
                raise
    st.markdown("---")
    # ======================================================
    # TECHNICIAN MANAGEMENT
    # ======================================================
    st.markdown("## Technician Management")
    with st.expander(" Add Technician"):
        tech_username = st.text_input("Username")
        tech_name = st.text_input("Full Name")
        tech_phone = st.text_input("Phone Number")
        tech_role = st.selectbox(
            "Specialization",
            ["Estimator", "Technician", "Inspector", "Adjuster", "Other"]
        )
        tech_active = st.checkbox("Active", True)
        if st.button("Save Technician"):
            add_technician(
                tech_username.strip(),
                full_name=tech_name.strip(),
                phone=tech_phone.strip(),
                specialization=tech_role,
                active=tech_active,
            )
            st.success("Technician saved")
            st.rerun()
    tech_df = get_technicians_df(active_only=False)
    if not tech_df.empty:
        for _, row in tech_df.iterrows():
            cols = st.columns([3, 2, 2])
            cols[0].write(f" **{row['full_name']}** (`{row['username']}`)")
            new_status = cols[1].selectbox(
                "Status",
                ["available", "assigned", "enroute", "onsite", "completed"],
                index=["available", "assigned", "enroute", "onsite", "completed"].index(
                    row.get("status", "available")
                ),
                key=f"status_{row['username']}",
            )
            if cols[2].button("Update", key=f"upd_{row['username']}"):
                update_technician_status(row["username"], new_status)
                st.success("Status updated")
                st.rerun()
    st.markdown("---")
    # ======================================================
    # CHANGE PASSWORD
    # ======================================================
    st.markdown("## Change Password")
    with st.form("change_password_form"):
        current_password = st.text_input("Current Password", type="password")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm New Password", type="password")
        if st.form_submit_button("Update Password"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
                st.stop()
            user = get_current_user()
            with SessionLocal() as s:
                db_user = s.query(User).filter(User.id == user.id).first()
                if not pwd_context.verify(current_password, db_user.password_hash):
                    st.error("Current password incorrect")
                    st.stop()
                db_user.password_hash = pwd_context.hash(new_password)
                s.commit()
            st.success("Password updated")
            st.rerun()
    st.markdown("---")
def page_technician_mobile():
    st.markdown("## Technician Mobile")
    techs = get_technicians_df(active_only=True)
    if techs.empty:
        st.warning("No active technicians found.")
        return

    tech = st.selectbox("Select Technician", techs["username"].tolist())
    st.markdown("### My Tasks")
    tasks = get_tasks_for_user(tech)
    if tasks.empty:
        st.info(
            "No task assigned to a Technician yet! To assign job task to a technician, go to:SETTINGS at the Navigation Menu, then click on the TECHNICIAN MANAGEMENT"
        )
    else:
        for _, t in tasks.iterrows():
            st.checkbox(
                f"{t['title']} (Lead: {t['lead_id']})",
                value=(t["status"] == "done"),
                key=f"task_{t['id']}",
            )

    st.markdown("### Send Location Ping")
    lat = st.number_input("Latitude", format="%.6f")
    lon = st.number_input("Longitude", format="%.6f")
    if st.button("Send Location Ping"):
        persist_location_ping(tech, lat, lon)
        st.success("Location sent")


#---------------------------Exports page--------------------------------------------
def page_exports():
    require_role_access("exports")
    st.markdown("<div class='header'>Exports & Imports</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Export leads, import CSV/XLSX. Imported rows upsert by lead_id.</em>",
        unsafe_allow_html=True
    )
    # =========================================================
    # EXPORT LEADS (AUTO-FALLBACK XLSX → CSV)
    # =========================================================
    df = leads_to_df(None, None)
    if not df.empty:
        towrite = io.BytesIO()
        try:
            # Preferred: Excel export
            df.to_excel(towrite, index=False, engine="openpyxl")
            file_type = "xlsx"
        except ModuleNotFoundError:
        # Fallback: CSV export
            towrite = io.StringIO()
        df.to_csv(towrite, index=False)
        file_type = "csv"
        st.download_button(
        label=f" Download leads ({file_type.upper()})",
        data=towrite.getvalue(),
        file_name=f"leads_export.{file_type}",
        mime="application/octet-stream"
        )
    else:
        st.info("No leads available to export.")
    st.divider()
    # =========================================================
    # IMPORT / UPSERT LEADS
    # =========================================================
    uploaded = st.file_uploader(
        "Upload leads (CSV/XLSX) for import/upsert",
        type=["csv", "xlsx"]
    )
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_in = pd.read_csv(uploaded)
            else:
                df_in = pd.read_excel(uploaded)

            if "lead_id" not in df_in.columns:
                st.error(" File must include a lead_id column")
            else:
                count = 0
                for _, r in df_in.iterrows():
                    try:
                        upsert_lead_record(
                            {
                                "lead_id": str(r["lead_id"]),
                                "created_at": (
                                    pd.to_datetime(r.get("created_at"))
                                    if r.get("created_at") is not None
                                    else pd.Timestamp.utcnow()
                                ),
                                "source": r.get("source"),
                                "contact_name": r.get("contact_name"),
                                "contact_phone": r.get("contact_phone"),
                                "contact_email": r.get("contact_email"),
                                "property_address": r.get("property_address"),
                                "damage_type": r.get("damage_type"),
                                "assigned_to": r.get("assigned_to"),
                                "notes": r.get("notes"),
                                "estimated_value": float(r.get("estimated_value") or 0.0),
                                "ad_cost": float(r.get("ad_cost") or 0.0),
                                "stage": r.get("stage") or "New",
                                "converted": bool(r.get("converted") or False),
                            },
                            actor="admin",
                        )
                        count += 1
                    except Exception:
                        continue

        except Exception as e:
            st.error(" Failed to import: " + str(e))
# ---------- BEGIN BLOCK F: FLASK API FOR LOCATION PINGS (optional but ready) ----------
try:
    from flask import Flask, request, jsonify
    import threading
    flask_app = Flask("recapture_pro_api")

    @flask_app.route("/api/ping_location", methods=["POST"])
    def api_ping_location():
        try:
            payload = request.get_json(force=True)
            tech = payload.get("tech_username") or payload.get("username")
            lat = payload.get("latitude") or payload.get("lat")
            lon = payload.get("longitude") or payload.get("lon")
            lead_id = payload.get("lead_id")
            accuracy = payload.get("accuracy")
            ts = payload.get("timestamp")
            ts_parsed = None
            if ts:
                try:
                    ts_parsed = datetime.fromisoformat(ts)
                except Exception:
                    ts_parsed = None
            if not tech or lat is None or lon is None:
                return jsonify({"error": "missing fields (tech_username, latitude, longitude)"}), 400
            pid = persist_location_ping(
                tech_username=str(tech), latitude=float(lat),
                longitude=float(lon), lead_id=lead_id,
                accuracy=accuracy, timestamp=ts_parsed
            )
            return jsonify({"ok": True, "ping_id": pid}), 200
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    def run_flask():
        try:
            # choose port 5001 to avoid Streamlit port conflicts
            flask_app.run(host="0.0.0.0", port=5001, debug=False, use_reloader=False)
        except Exception:
            pass

    # start flask in background daemon thread (only if not already started)
    t = threading.Thread(target=run_flask, daemon=True)
    t.start()
except Exception:
    # if Flask isn't available (not installed) the API simply won't start — harmless
    pass
# ---------- END BLOCK F ----------
demand = {}
season_score = 0.5
def add_time_windows(hist_df):
    """
Adds rolling time windows (3, 6, 12 months) for seasonal comparison
"""
    if hist_df.empty:
        return {}

    df = hist_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    latest_date = df["date"].max()
    windows = {
        "3_months": df[df["date"] >= latest_date - pd.DateOffset(months=3)],
        "6_months": df[df["date"] >= latest_date - pd.DateOffset(months=6)],
        "12_months": df[df["date"] >= latest_date - pd.DateOffset(months=12)],
    }
    return windows
# -------------------------------------------------------------
# SEASONAL TRENDS PAGE — SINGLE SOURCE OF TRUTH
# -------------------------------------------------------------
def page_seasonal_trends():
    require_role_access("business_intelligence")

    st.markdown("## Seasonal Trends & Weather-Based Damage Insights")
    st.caption("Location-based risk forecasting with business impact intelligence")
    st.divider()

    countries = get_all_countries()
    country = st.selectbox("Country", [c["name"] for c in countries])
    country_code = next(c["code"] for c in countries if c["name"] == country)

    city_query = st.text_input("City", placeholder="Type at least 3 letters")
    if len(city_query) < 3:
        st.info("Start typing a city name (3+ characters)")
        return

    matches = search_cities(country_code, city_query)
    if not matches:
        st.warning("No cities found")
        return

    labels = [f"{m['name']}, {m.get('admin1', '')}" for m in matches]
    selected = st.selectbox("Select City", labels)
    chosen = matches[labels.index(selected)]

    hist_range = st.selectbox("Historical Window", ["3 months", "6 months", "12 months"], index=1)
    months = {"3 months": 3, "6 months": 6, "12 months": 12}[hist_range]

    if not st.button("Generate Insights", use_container_width=True):
        return

    with st.spinner("Generating insights..."):
        hist_df = fetch_weather(chosen["lat"], chosen["lon"], months)

    if hist_df is None or hist_df.empty:
        st.error("No historical data available")
        return

    hist_df = hist_df.copy()
    hist_df["date"] = pd.to_datetime(hist_df["date"], errors="coerce")
    hist_df = hist_df.dropna(subset=["date"]).sort_values("date")

    st.line_chart(hist_df.set_index("date")[["rainfall_mm", "temperature_c"]])

def page_competitor_intelligence():
    st.title(" Competitor Intelligence")

    st.subheader(" Competitive Alerts")
    s = get_session()
    try:
        alerts = (
            s.query(CompetitorAlert)
            .order_by(CompetitorAlert.created_at.desc())
            .limit(5)
            .all()
        )
    finally:
        s.close()

    if not alerts:
        st.success("No competitive threats detected.")
    else:
        for a in alerts:
            if a.severity == "high":
                st.error(a.message)
            else:
                st.warning(a.message)

    st.divider()

    with st.expander(" Discover Competitors"):
        lat = st.number_input("Latitude", value=39.9612, key="comp_lat")
        lon = st.number_input("Longitude", value=-82.9988, key="comp_lon")
        keyword = st.text_input("Search keyword", "water damage restoration", key="comp_keyword")
        if st.button("Run Competitor Scan", key="run_comp_scan"):
            ingest_competitors_openstreetmap(lat, lon, keyword)
            st.success("Competitor scan completed.")

    s = get_session()
    try:
        competitors = s.query(Competitor).all()
    finally:
        s.close()

    if not competitors:
        st.info("No competitors tracked yet.")
        return

    rows = []
    hq_lat = st.session_state.get("hq_lat")
    hq_lon = st.session_state.get("hq_lon")
    for c in competitors:
        distance = 10
        if hq_lat and hq_lon and c.latitude and c.longitude:
            distance = haversine_km(hq_lat, hq_lon, c.latitude, c.longitude)
        score = calculate_competitor_score(c.rating or 0, c.total_reviews or 0, distance)
        rows.append(
            {
                "Name": c.name,
                "Rating": c.rating,
                "Reviews": c.total_reviews,
                "Category": c.primary_category,
                "Distance (km)": round(distance, 2),
                "Strength Score": score,
                "Velocity (7d)": review_velocity(c.id, 7),
                "Velocity (30d)": review_velocity(c.id, 30),
            }
        )

    df = pd.DataFrame(rows).sort_values("Strength Score", ascending=False)
    st.subheader("Top Competitors")
    st.dataframe(df, use_container_width=True, key="competitor_table")

def save_review_link_for_user(user, review_link):
    if not user or not review_link:
        return
    with SessionLocal() as s:
        existing = s.query(ReviewSettings).filter(
            ReviewSettings.user_id == user.id
        ).first()
        if existing:
            existing.review_link = review_link
        else:
            s.add(ReviewSettings(user_id=user.id, review_link=review_link))
        s.commit()
def page_google_reviews():
    st.subheader(" Google Reviews")
    st.caption("Request Google reviews from completed jobs to boost reputation and local SEO.")

    st.subheader(" Google Review Link")
    review_link = st.text_input(
        "Paste your Google Review link",
        placeholder="https://g.page/your-business/review",
    )
    if st.button(" Save Review Link"):
        if not review_link:
            st.error("Review link is required")
        else:
            save_review_link_for_user(get_current_user(), review_link)
            st.success("Review link saved")

    st.divider()
    st.subheader(" Select Customer")
    contacts = get_completed_job_contacts(get_current_user())
    if not contacts:
        st.info("No completed job contacts yet.")
        return

    contact = st.selectbox(
        "Choose a customer",
        contacts,
        format_func=lambda c: f"{c['name']} ({c['email']})",
    )

    st.divider()
    st.subheader(" Send Review Request")
    if st.button("Send Google Review Request"):
        send_google_review_request(
            to_email=contact["email"],
            customer_name=contact["name"],
            review_link=review_link,
            job_name=contact.get("job_title", ""),
        )
        st.success("Review request sent")

def send_google_review_request(
    to_email: str,
    customer_name: str,
    review_link: str,
    job_name: str = ""
):
    subject = "We'd love your Google review"
    job_line = f" regarding your recent {job_name}" if job_name else ""
    html_body = (
        f"<p>Hi {customer_name},</p>"
        f"<p>Thank you for choosing us{job_line}.</p>"
        f"<p>We'd appreciate a quick Google review: "
        f"<a href='{review_link}'>Leave a Review</a></p>"
        f"<p>Thank you,<br>The Team</p>"
    )
    send_email(to_email=to_email, subject=subject, html_body=html_body)

# ---------- END SETTINGS AND EMAIL INVITES ----------
def page_request_review_settings():
    require_role_access("settings")
    st.markdown("## Request Review Settings")
    st.caption("Configure how you collect Google reviews")
    st.markdown("---")
    settings = get_user_settings_safe()
    existing_review_link = settings.get("google_review_url", "")
    review_link = st.text_input(
        "Google Review Link",
        value=existing_review_link,
        placeholder="https://g.page/your-business/review"
    )
    st.markdown("ℹ This link will be used for NFC tap cards, QR codes, and manual review requests.")
    if st.button("Save Review Link"):
        settings["google_review_url"] = review_link
        st.success("Review link saved successfully.")
    st.markdown("---")
    st.markdown("### NFC & QR Review Tools")
    st.info("NFC tap cards and QR codes will redirect customers to your saved Google review link. No paid API required.")
def page_request_review():
    require_role_access("overview")
    st.markdown("## Request Google Review")
    st.caption("Instant on-site review request via Tap or QR")

    settings = get_user_settings_safe()
    review_url = settings.get("google_review_url")
    if not review_url:
        st.warning("Google Review link not set. Add it in Settings.")
        return

    lead_id = st.session_state.get("active_lead_id")
    token = generate_review_token()
    base_url = st.secrets.get("APP_BASE_URL", "http://localhost:8501")
    review_request_url = f"{base_url}/?page=review_redirect&token={token}"

    try:
        log_event(
            event_type="review_requested",
            entity_type="lead" if lead_id else "session",
            entity_id=lead_id,
            metadata={"method": "manual"},
        )
    except Exception:
        pass

    st.write("Share this review link with your customer:")
    st.code(review_request_url)

def page_review_redirect():
    token = st.query_params.get("token")
    settings = get_user_settings_safe()
    review_url = settings.get("google_review_url")
    if not review_url:
        st.error("Review link not configured.")
        return

    try:
        log_event(
            event_type="review_tap",
            entity_type="review",
            entity_id=token,
            metadata={"source": "nfc_or_qr"},
        )
    except Exception:
        pass

    st.markdown("Redirecting to review page…")
    st.markdown(
        f"<meta http-equiv='refresh' content='1;url={review_url}'>",
        unsafe_allow_html=True,
    )

#-----------------------START OF COMMAND CENTER---------------------
def page_command_center():
    require_role_access("overview")
    # =========================================================
    # LOAD DATA SAFELY
    # =========================================================
    try:
        df = get_leads_df()
    except Exception:
        df = pd.DataFrame()
    # =========================================================
    # HEADER
    # =========================================================
    st.markdown("## Command Center")
    st.caption("Real-time business health & priorities")
    st.divider()
    # =========================================================
    # EMPTY STATE
    # =========================================================
    if df.empty:
        st.info(
            "No activity yet.\n\n"
            "Once you start capturing leads, this dashboard will show:\n"
            "• Revenue at risk & recovered\n"
            "• Follow-ups due\n"
            "• Daily priorities\n"
            "• Business insights\n"
            "• Activity timeline"
        )
        if st.button("Capture your first lead", use_container_width=True):
            st.session_state.page = "Lead Capture"
            st.rerun()
        st.stop()
    # =========================================================
    # NORMALIZATION
    # =========================================================
    df = df.copy()
    df["estimated_value"] = df.get("estimated_value", 0).fillna(0)
    df["stage"] = df.get("stage", "New").fillna("New")
    df["created_at"] = pd.to_datetime(df.get("created_at"), errors="coerce")
    df["updated_at"] = pd.to_datetime(df.get("updated_at", df["created_at"]), errors="coerce")
    df["sla_hours"] = df.get("sla_hours", 24)
    now = pd.Timestamp.utcnow()
    today = now.date()
    yesterday = today - timedelta(days=1)
    df["lead_age_hours"] = (now - df["created_at"]).dt.total_seconds() / 3600
    # =========================================================
    # KPI CALCULATIONS
    # =========================================================
    inspection_count = len(df[df["stage"] == "Inspection"])
    won_count = len(df[df["stage"] == "Won"])
    inspection_conversion = (won_count / inspection_count) * 100 if inspection_count else 0
    follow_up_24h = df[
        (df["stage"].isin(["New", "Contacted"])) &
        (df["lead_age_hours"] >= 24)
    ]
    stalled_revenue = df[
        (df["stage"].isin(["Inspection", "Estimate Sent"])) &
        (df["lead_age_hours"] > df["sla_hours"])
    ]["estimated_value"].sum()
    recovered_today = df[(df["stage"] == "Won") & (df["updated_at"].dt.date == today)]["estimated_value"].sum()
    recovered_yesterday = df[(df["stage"] == "Won") & (df["updated_at"].dt.date == yesterday)]["estimated_value"].sum()
    recovered_delta = recovered_today - recovered_yesterday
    avg_response = df["lead_age_hours"].mean()
    # =========================================================
    # AI SUMMARY
    # =========================================================
    signals = []
    if stalled_revenue > 0:
        signals.append(f"${stalled_revenue:,.0f} stalled")
    if len(follow_up_24h) > 0:
        signals.append(f"{len(follow_up_24h)} follow-ups due")
    if inspection_conversion < 30:
        signals.append("low inspection conversion")
    summary = "Everything is operating smoothly." if not signals else "Attention needed: " + ", ".join(signals)
    st.markdown(
        f'<div style="background:#f8fafc;border-left:4px solid #2563eb;padding:14px 18px;border-radius:10px;font-size:0.95rem;">'
        f'🤖 <strong>Executive Insight:</strong> {summary}</div>',
        unsafe_allow_html=True
    )
    st.divider()
    # =========================================================
    # KPI OVERVIEW
    # =========================================================
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Stalled Revenue", f"${stalled_revenue:,.0f}")
    k2.metric("Revenue Recovered Today", f"${recovered_today:,.0f}", f"{'+' if recovered_delta >= 0 else ''}${recovered_delta:,.0f}")
    k3.metric("Follow-ups Due", len(follow_up_24h))
    k4.metric("Avg Response Time", f"{avg_response:.1f}h")
    st.divider()
    # =========================================================
    # TODAY'S PRIORITIES
    # =========================================================
    st.subheader("Today's Priorities")
    priorities = follow_up_24h.sort_values("created_at").head(5)
    if priorities.empty:
        st.success("No urgent actions required today")
    else:
        for _, lead in priorities.iterrows():
            st.markdown(f"**Follow up with Lead #{lead['lead_id']}** — Overdue {int(lead['lead_age_hours'])} hours")
    st.divider()
    # =========================================================
    # BUSINESS INSIGHTS
    # =========================================================
    st.subheader("Business Insights")
    if stalled_revenue > 0:
        st.warning(f"{inspection_count} inspections or estimates stalled with ${stalled_revenue:,.0f} at risk.")
    else:
        st.success("No operational bottlenecks detected.")
    if inspection_conversion < 30:
        st.info("Inspection → Won conversion is below target. Review recent inspections.")
    st.divider()
    # =========================================================
    # TODAY vs YESTERDAY
    # =========================================================
    st.subheader("Today vs Yesterday")
    c1, c2, c3 = st.columns(3)
    c1.metric("Revenue Recovered", f"${recovered_today:,.0f}", f"{'+' if recovered_delta >= 0 else ''}${recovered_delta:,.0f}")
    c2.metric("Follow-ups Due", len(follow_up_24h))
    c3.metric("Pipeline Health", "Stable" if stalled_revenue == 0 else "Needs Attention")
    st.divider()
    # =========================================================
    # RECENT ACTIVITY
    # =========================================================
    st.subheader("Recent Activity")
    st.caption("Latest movements across your pipeline")
    timeline_df = (
        df[["lead_id", "stage", "updated_at"]]
        .dropna()
        .sort_values("updated_at", ascending=False)
        .head(8)
    )
    for _, row in timeline_df.iterrows():
        st.markdown(
            f"• **Lead #{row['lead_id']}** → **{row['stage']}** — {row['updated_at'].strftime('%b %d, %Y %H:%M')}"
        )

# =============================================================
# MAIN ENTRY POINT
# =============================================================

# Restore existing Supabase session if available
try:
    sess = supabase.auth.get_session()
    user_obj = sess.session.user if sess and sess.session else None
    if user_obj and not st.session_state.get("authenticated"):
        set_auth_session(user_obj)
except Exception:
    pass

# Handle WordPress token bridge and password reset links
if "token" in st.query_params:
    if st.query_params.get("reset") == "1":
        page_reset_password()
    else:
        wp_auth_bridge()
    st.stop()

# Show login/signup if not authenticated
if not st.session_state.get("authenticated"):
    page_auth()
    st.stop()

# ---- AUTHENTICATED AREA ----

user = get_current_user()

# Trial banner
if user and getattr(user, "subscription_status", None) == "trial" and getattr(user, "trial_ends_at", None):
    try:
        days_left = max(0, (user.trial_ends_at - pd.Timestamp.utcnow()).days)
        st.sidebar.warning(f"Trial ends in {days_left} days")
    except Exception:
        pass

# ----------------------
# NAV ICONS
# ----------------------
NAV_ICONS = {
    "Command Center": "🏠",
    "Overview": "📊",
    "Lead Capture": "📥",
    "Pipeline Board": "📋",
    "Analytics": "📈",
    "CPA & ROI": "💰",
    "Tasks": "✅",
    "AI Recommendations": "🤖",
    "Seasonal Trends": "🌦",
    "Settings": "⚙️",
    "Request Review": "⭐",
    "Exports": "📤",
}

# ----------------------
# SIDEBAR NAVIGATION
# ----------------------
with st.sidebar:
    st.header("Navigation")
    st.markdown("---")

pages = [
    "Command Center",
    "Overview",
    "Lead Capture",
    "Pipeline Board",
    "Analytics",
    "CPA & ROI",
    "Tasks",
    "AI Recommendations",
    "Seasonal Trends",
    "Request Review",
    "Settings",
    "Exports",
]

# Default page is Command Center
if "page" not in st.session_state or st.session_state.page in ("command_center", "auth", ""):
    st.session_state.page = "Command Center"

if "ai_insights" not in st.session_state:
    st.session_state.ai_insights = []

page_labels = [f"{NAV_ICONS.get(p, '')} {p}" for p in pages]
current_index = pages.index(st.session_state.page) if st.session_state.page in pages else 0

selected_label = st.sidebar.radio(
    "Navigate",
    page_labels,
    index=current_index,
    label_visibility="collapsed",
)
selected_page = selected_label.split(" ", 1)[-1].strip()

if selected_page != st.session_state.page:
    st.session_state.page = selected_page
    st.rerun()

# ----------------------
# ROUTER
# ----------------------
page = st.session_state.page

if page == "Command Center":
    page_command_center()
elif page == "Overview":
    page_overview()
elif page == "Lead Capture":
    page_lead_capture()
elif page == "Pipeline Board":
    page_pipeline_board()
elif page == "Analytics":
    page_analytics()
elif page == "CPA & ROI":
    page_cpa_roi()
elif page == "Tasks":
    page_tasks()
elif page == "AI Recommendations":
    page_ai_recommendations()
elif page == "Seasonal Trends":
    page_seasonal_trends()
elif page == "Settings":
    page_settings()
elif page == "Request Review":
    page_request_review()
elif page == "review_redirect":
    page_review_redirect()
elif page == "Request Review Settings":
    page_request_review_settings()
elif page == "Exports":
    page_exports()
else:
    st.info("Page not implemented yet.")

# Footer
st.markdown("---")
st.markdown("<div style='font-size:0.8rem;color:#999;'>ReCapture Pro. Sales Intelligence and Conversion.</div>", unsafe_allow_html=True)
