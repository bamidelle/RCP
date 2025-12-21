# titan_backend.py
"""
TITAN Backend - Single-file Streamlit app
- No front-page login: this is an admin backend (Admin access by default)
- User & Role management available in Settings
- SQLite persistence via SQLAlchemy
- Internal ML training & scoring (no user tuning)
- Pipeline dashboard, Analytics, CPA/ROI, Exports/Imports, Alerts, SLA, Priority scoring, Audit trail
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

import joblib
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
        {"name":"United States","code":"US"},
        {"name":"Canada","code":"CA"},
        {"name":"United Kingdom","code":"GB"},
        {"name":"Australia","code":"AU"}
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

    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "rainfall_mm": d["precipitation_sum"],
        "temperature_c": d["temperature_2m_mean"],
    })

    return df.dropna().reset_index(drop=True)


@lru_cache(maxsize=128)
def fetch_forecast_weather(lat, lon, days):
    """
    Short-term daily forecast (Open-Meteo limit ~14 days)
    """
    days = min(days, 14)  # API limit

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

    df = pd.DataFrame({
        "date": pd.to_datetime(d["time"]),
        "rainfall_mm": d["precipitation_sum"],
        "temperature_c": d["temperature_2m_mean"],
    })

    return df.dropna().reset_index(drop=True)
# ---------- end weather helpers ----------



# =============================================================


# ----------------------
# CONFIG
# ----------------------
APP_TITLE = "ReCapture Pro"
DB_FILE = "titan_backend.db"   # stored in app working directory
MODEL_FILE = "titan_model.joblib"
PIPELINE_STAGES = [
    "New", "Contacted", "Inspection Scheduled", "Inspection Completed",
    "Estimate Sent", "Qualified", "Won", "Lost"
]
DEFAULT_SLA_HOURS = 72
COMFORTAA_IMPORT = "https://fonts.googleapis.com/css2?family=Comfortaa:wght@300;400;700&display=swap"


# KPI colors (numbers)
KPI_COLORS = ["#2563eb", "#0ea5a4", "#a855f7", "#f97316", "#ef4444", "#6d28d9", "#22c55e"]


# ----------------------
# DB SETUP
# ----------------------
DB_PATH = os.path.join(os.getcwd(), DB_FILE)
ENGINE_URL = f"sqlite:///{DB_PATH}"


# SQLAlchemy engine and session factory
engine = create_engine(ENGINE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


# ----------------------
# MODELS
# ----------------------
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String, default="")
    role = Column(String, default="Admin")  # Admin by default for backend
    created_at = Column(DateTime, default=datetime.utcnow)


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
    assigned_to = Column(String, nullable=True)  # username of owner
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
    ad_cost = Column(Float, default=0.0)  # cost to acquire
    converted = Column(Boolean, default=False)
    score = Column(Float, nullable=True)  # ML probability


class LeadHistory(Base):
    __tablename__ = "lead_history"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=False)
    changed_by = Column(String, nullable=True)
    field = Column(String, nullable=True)
    old_value = Column(String, nullable=True)
    new_value = Column(String, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    # ---------- BEGIN BLOCK A: NEW MODELS (Technician, InspectionAssignment, LocationPing) ----------
from sqlalchemy import DateTime as SA_DateTime


class Technician(Base):
    __tablename__ = "technicians"

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String, default="")
    phone = Column(String, nullable=True)
    specialization = Column(String, nullable=True)

    # ✅ ADD THIS LINE
    status = Column(String, default="available")  
    # available, assigned, enroute, onsite, completed

    active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)



class InspectionAssignment(Base):
    __tablename__ = "inspection_assignments"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=False)          # lead_id from Lead.lead_id
    technician_username = Column(String, nullable=False)
    assigned_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="assigned")      # assigned, enroute, onsite, completed, cancelled
    notes = Column(Text, nullable=True)


class LocationPing(Base):
    __tablename__ = "location_pings"
    id = Column(Integer, primary_key=True)
    tech_username = Column(String, nullable=False)
    lead_id = Column(String, nullable=True)          # optional - link to lead if assigned
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    accuracy = Column(Float, nullable=True)          # optional accuracy (meters)

class Task(Base):
    __tablename__ = "tasks"
    id = Column(Integer, primary_key=True)
    lead_id = Column(String, nullable=True)
    technician_username = Column(String, nullable=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    status = Column(String, default="open")  # open, in_progress, done
    due_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

# ---------- BEGIN BLOCK A2: COMPETITOR INTELLIGENCE MODELS ----------

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
    severity = Column(String)  # low / medium / high
    created_at = Column(DateTime, default=datetime.utcnow)

# ---------- END BLOCK A2 ----------

# ---------- END BLOCK A ----------




# Create tables if missing
from sqlalchemy import inspect


def safe_create_tables():
    inspector = inspect(engine)
    existing_tables = inspector.get_table_names()


    for table in Base.metadata.sorted_tables:
        if table.name not in existing_tables:
            table.create(bind=engine)


safe_create_tables()




# Safe migration attempt (best-effort add missing columns)
def safe_migrate():
    try:
        inspector = inspect(engine)
        if "leads" in inspector.get_table_names():
            existing = [c['name'] for c in inspector.get_columns("leads")]
            desired = {
                "score": "FLOAT",
                "ad_cost": "FLOAT",
                "source_details": "TEXT",
                "contact_name": "TEXT",
                "assigned_to": "TEXT",
            }
            conn = engine.connect()
            for col, typ in desired.items():
                if col not in existing:
                    try:
                        conn.execute(f"ALTER TABLE leads ADD COLUMN {col} {typ}")
                    except Exception:
                        pass
            conn.close()
    except Exception:
        pass


from sqlalchemy import inspect, text

def safe_migrate_new_tables():
    """Add missing columns and create tables safely."""
    try:
        inspector = inspect(engine)

        # Check if "status" column exists in "technicians"
        cols = [c["name"] for c in inspector.get_columns("technicians")]
        if "status" not in cols:
            with engine.connect() as conn:
                conn.execute(text("ALTER TABLE technicians ADD COLUMN status VARCHAR DEFAULT 'available'"))
                conn.commit()

        # Create any missing tables
        Base.metadata.create_all(bind=engine)
    except Exception as e:
        print(f"Safe migration failed: {e}")

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

    a = math.sin(dphi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2

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
            total_reviews=total_reviews
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
    # Replace spaces in keyword for OSM query
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
                if name and lat_ and lon_:
                    # Avoid duplicates
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


# ---------- END BLOCK F ----------
def review_velocity(competitor_id, days):
    s = get_session()
    try:
        since = datetime.utcnow() - timedelta(days=days)
        count = (
            s.query(CompetitorSnapshot)
            .filter(
                CompetitorSnapshot.competitor_id == competitor_id,
                CompetitorSnapshot.captured_at >= since
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
                s.add(CompetitorAlert(
                    competitor_id=c.id,
                    alert_type="REVIEW_SPIKE",
                    severity="high",
                    message=f"{c.name} gained {v7} reviews in 7 days."
                ))

            if v30 >= 25:
                s.add(CompetitorAlert(
                    competitor_id=c.id,
                    alert_type="AGGRESSIVE_GROWTH",
                    severity="high",
                    message=f"{c.name} gained {v30} reviews in 30 days."
                ))

        s.commit()
    finally:
        s.close()


# ----------------------
# HELPERS: DB ops
# ----------------------
def get_session():
    return SessionLocal()


def leads_to_df(start_date=None, end_date=None):
    """Load leads into a DataFrame. Filter by optional start_date/end_date (date objects)"""
    s = get_session()
    try:
        rows = s.query(Lead).order_by(Lead.created_at.desc()).all()
        data = []
        for r in rows:
            data.append({
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
                "score": float(r.score) if r.score is not None else None
            })
        df = pd.DataFrame(data)
        if df.empty:
            # return empty with expected columns
            cols = ["id","lead_id","created_at","source","source_details","contact_name","contact_phone","contact_email",
                    "property_address","damage_type","assigned_to","notes","estimated_value","stage","sla_hours","sla_entered_at",
                    "contacted","inspection_scheduled","inspection_scheduled_at","inspection_completed","estimate_submitted",
                    "awarded_date","lost_date","qualified","ad_cost","converted","score"]
            return pd.DataFrame(columns=cols)
        # apply date filters
        if start_date:
            start_dt = datetime.combine(start_date, datetime.min.time())
            df = df[df["created_at"] >= start_dt]
        if end_date:
            end_dt = datetime.combine(end_date, datetime.max.time())
            df = df[df["created_at"] <= end_dt]
        return df.reset_index(drop=True)
    finally:
        s.close()
# ---------- BEGIN BLOCK C: DB HELPERS FOR TECHNICIANS / ASSIGNMENTS / PINGS ----------
def create_task(title, technician_username=None, lead_id=None, due_at=None, description=None):
    s = get_session()
    try:
        task = Task(
            title=title,
            technician_username=technician_username,
            lead_id=lead_id,
            description=description,
            status="open",
            due_at=due_at
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
        return pd.DataFrame([
            {
                "id": r.id,
                "title": r.title,
                "status": r.status,
                "lead_id": r.lead_id,
                "due_at": r.due_at
            } for r in rows
        ])
    finally:
        s.close()
def page_tasks():
    st.markdown("## ✅ Technician Tasks")

    techs = get_technicians_df(active_only=True)

    if techs.empty:
        st.warning("No technicians available.")
        return

    tech_username = st.selectbox(
        "Select Technician",
        techs["username"].tolist()
    )

    tasks_df = get_tasks_for_user(tech_username)

    if tasks_df.empty:
        st.info("No tasks assigned.")
        return

    for _, row in tasks_df.iterrows():
        with st.expander(f"🧾 {row['title']} — {row['status'].upper()}"):
            st.write(f"**Lead ID:** {row['lead_id'] or 'N/A'}")
            st.write(f"**Due:** {row['due_at'] or 'No due date'}")

            if row["status"] == "open":
                if st.button("▶️ Start Task", key=f"start_{row['id']}"):
                    update_task_status(row["id"], "in_progress")
                    st.success("Task started")
                    st.rerun()

            elif row["status"] == "in_progress":
                if st.button("✅ Mark Complete", key=f"done_{row['id']}"):
                    update_task_status(row["id"], "done")
                    st.success("Task completed")
                    st.rerun()

            elif row["status"] == "done":
                st.success("✔ Completed")


def get_tasks_df():
    s = get_session()
    try:
        rows = s.query(Task).order_by(Task.created_at.desc()).all()
        return pd.DataFrame([
            {
                "id": r.id,
                "title": r.title,
                "technician_username": r.technician_username,
                "lead_id": r.lead_id,
                "status": r.status,
                "due_at": r.due_at,
                "created_at": r.created_at
            } for r in rows
        ])
    finally:
        s.close()


def add_technician(username: str, full_name: str = "", phone: str = "", specialization: str = "Tech", active: bool = True):
    s = get_session()
    try:
        existing = s.query(Technician).filter(Technician.username == username).first()
        if existing:
            existing.full_name = full_name
            existing.phone = phone
            existing.specialization = specialization
            existing.active = active
            s.add(existing); s.commit()
            return existing.username
        t = Technician(username=username, full_name=full_name, phone=phone, specialization=specialization, active=active)
        s.add(t); s.commit()
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
        data.get("accuracy")
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
            timestamp=datetime.utcnow()
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

        return pd.DataFrame([
            {
                "username": t.username,
                "full_name": t.full_name,
                "phone": t.phone,
                "specialization": t.specialization,
                "active": t.active
            }
            for t in rows
        ])
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
            accuracy=accuracy
        )
        s.add(ping)
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()



def get_leads_df():
    s = get_session()
    try:
        rows = s.query(Lead).all()

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                "lead_id": r.lead_id,
                "stage": r.stage,
                "estimated_value": r.estimated_value or 0,
                "assigned_to": r.assigned_to,
                "score": r.score if r.score is not None else 0.5,
                "sla_hours": r.sla_hours,
                "sla_entered_at": r.sla_entered_at,
                "created_at": r.created_at,
                "source": r.source,
                "damage_type": r.damage_type
            }
            for r in rows
        ])
    finally:
        s.close()


def get_jobs_for_period(start_dt, end_dt):
    df = get_leads_df()  # existing function

    if df.empty:
        return df

    df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
    df = df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)]

    return df

def compute_job_volume_metrics(df):
    return {
        "total_jobs": len(df),
        "job_types": safe_col(df, "job_type").value_counts().to_dict(),
        "lead_sources": safe_col(df, "lead_source").value_counts().to_dict(),
    }



def compute_revenue_metrics(df):
    revenue_col = safe_col(df, "estimated_value", default_dtype=float).fillna(0)

    total_revenue = revenue_col.sum()

    revenue_per_job = (
        total_revenue / len(df)
        if len(df) > 0 else 0
    )

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
        "revenue_per_job": (
            total_revenue / jobs if jobs > 0 else 0
        ),
    }


def generate_synthetic_signals(df):
    signals = []

    revenue_col = safe_col(df, "estimated_value", default_dtype=float).fillna(0)
    job_type_col = safe_col(df, "job_type")

    if len(df) == 0:
        signals.append("🧊 No job activity detected in this period.")
        return signals

    avg_revenue = revenue_col.mean()

    if avg_revenue < 500:
        signals.append("⚠️ High job volume but low average revenue per job.")

    dominant_job = job_type_col.value_counts().idxmax() if not job_type_col.dropna().empty else None

    if dominant_job:
        signals.append(f"🎯 Revenue is highly concentrated in '{dominant_job}' jobs.")

    return signals


def seasonal_baseline(all_jobs, start_date, end_date):
    same_months = list(range(start_date.month, end_date.month + 1))

    hist = all_jobs[
        all_jobs["created_at"].dt.month.isin(same_months)
    ]

    if hist.empty:
        return None

    return {
        "avg_jobs": hist.groupby(hist["created_at"].dt.year).size().mean(),
        "avg_revenue": hist.groupby(hist["created_at"].dt.year)["estimated_value"].sum().mean(),
        "avg_rev_per_job": hist["estimated_value"].mean()
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

    
def analyze_job_types(df):
    if df.empty:
        return {}

    grouped = df.groupby("damage_type").agg(
        jobs=("id", "count"),
        revenue=("estimated_value", "sum"),
        avg_revenue=("estimated_value", "mean"),
        revenue_std=("estimated_value", "std")
    ).reset_index()

    total_jobs = grouped["jobs"].sum()
    total_revenue = grouped["revenue"].sum()

    grouped["job_share"] = grouped["jobs"] / total_jobs
    grouped["revenue_share"] = grouped["revenue"] / total_revenue
    grouped["volatility"] = grouped["revenue_std"].fillna(0)

    insights = []

    for _, r in grouped.iterrows():
        if r["job_share"] > 0.45:
            insights.append(f"⚠️ Over-dependence on **{r['damage_type']}** jobs")

        if r["job_share"] > 0.3 and r["avg_revenue"] < grouped["avg_revenue"].mean():
            insights.append(
                f"💸 **{r['damage_type']}** jobs are high-volume but low-value"
            )

        if r["volatility"] > grouped["volatility"].mean() * 1.5:
            insights.append(
                f"📉 **{r['damage_type']}** revenue is highly volatile"
            )

    return {
        "table": grouped.sort_values("revenue", ascending=False),
        "insights": insights
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


def compute_business_intelligence(range_key, custom_start=None, custom_end=None):
    # -----------------------------
    # Current Period
    # -----------------------------
    start, end = resolve_time_window(range_key, custom_start, custom_end)
    df = get_jobs_for_period(start, end)

    # -----------------------------
    # Previous Period (Same Length)
    # -----------------------------
    prev_start, prev_end = shift_period(start, end)
    prev_df = get_jobs_for_period(prev_start, prev_end)

    # -----------------------------
    # Metrics (Current)
    # -----------------------------
    volume = compute_job_volume_metrics(df)
    revenue = compute_revenue_metrics(df)
    efficiency = compute_efficiency_metrics(df)

    # -----------------------------
    # Metrics (Previous)
    # -----------------------------
    prev_volume = compute_job_volume_metrics(prev_df)
    prev_revenue = compute_revenue_metrics(prev_df)
    prev_efficiency = compute_efficiency_metrics(prev_df)

    # -----------------------------
    # Trends / Deltas (%)
    # -----------------------------
    volume["trend"] = pct_change(
        volume.get("total_jobs", 0),
        prev_volume.get("total_jobs", 0)
    )

    revenue["trend"] = pct_change(
        revenue.get("total_revenue", 0),
        prev_revenue.get("total_revenue", 0)
    )

    efficiency["trend"] = pct_change(
        efficiency.get("revenue_per_job", 0),
        prev_efficiency.get("revenue_per_job", 0)
    )

    # -----------------------------
    # Synthetic Signals
    # -----------------------------
    signals = generate_synthetic_signals(df)

    # -----------------------------
    # Final Output Object
    # -----------------------------
    output = {
        "window": (start, end),
        "previous_window": (prev_start, prev_end),
        "volume": volume,
        "revenue": revenue,
        "efficiency": efficiency,
        "signals": signals,
        "raw_df": df,
        "prev_df": prev_df,
    }

    # -----------------------------
    # STEP G-2️⃣ — Narrative Engine Feed
    # -----------------------------
    output["executive_narrative"] = generate_executive_narrative(output)

    return output






def add_time_windows(df, date_col="date"):
    df[date_col] = pd.to_datetime(df[date_col])
    now = df[date_col].max()

    return {
        "3m": df[df[date_col] >= now - pd.DateOffset(months=3)],
        "6m": df[df[date_col] >= now - pd.DateOffset(months=6)],
        "12m": df[df[date_col] >= now - pd.DateOffset(months=12)],
    }
def generate_seasonal_insights(leads_df, hist_df):
    insights = []

    # Safety check
    if leads_df.empty or "damage_type" not in leads_df.columns:
        return ["ℹ️ No lead data available to generate insights."]

    # Example logic
    top_damage = leads_df["damage_type"].value_counts().idxmax()
    insights.append(f"Most common damage type among recent leads: {top_damage}")

    # You can add more insights based on hist_df
    avg_rain = hist_df["rainfall_mm"].mean()
    if avg_rain > 50:
        insights.append("🌧️ High rainfall observed, water damage risk elevated.")
    else:
        insights.append("🌤️ Rainfall is moderate, typical seasonal risk.")

    return insights



def create_inspection_assignment(lead_id: str, technician_username: str, notes: str = None):
    s = get_session()
    try:
        ia = InspectionAssignment(lead_id=lead_id, technician_username=technician_username, notes=notes)
        s.add(ia)
        # optional: also set lead.assigned_to to technician_username (non-destructive)
        lead = s.query(Lead).filter(Lead.lead_id == lead_id).first()
        if lead:
            lead.assigned_to = technician_username
            s.add(lead)
            s.add(LeadHistory(lead_id=lead.lead_id, changed_by="system", field="assigned_to", old_value=str(getattr(lead, "assigned_to", "")), new_value=str(technician_username)))
        s.commit()
        return ia.id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_assignments_for_lead(lead_id: str):
    s = get_session()
    try:
        rows = s.query(InspectionAssignment).filter(InspectionAssignment.lead_id == lead_id).order_by(InspectionAssignment.assigned_at.desc()).all()
        return rows
    finally:
        s.close()


def persist_location_ping(tech_username: str, latitude: float, longitude: float, lead_id: str = None, accuracy: float = None, timestamp: datetime = None):
    s = get_session()
    try:
        ping = LocationPing(tech_username=tech_username, latitude=float(latitude), longitude=float(longitude), lead_id=lead_id, accuracy=accuracy, timestamp=timestamp or datetime.utcnow())
        s.add(ping)
        s.commit()
        return ping.id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()

def get_latest_location_pings():
    s = get_session()
    try:
        rows = s.execute(
            text("""
                SELECT lp.tech_username, lp.latitude, lp.longitude, lp.timestamp
                FROM location_pings lp
                INNER JOIN (
                    SELECT tech_username, MAX(timestamp) AS max_ts
                    FROM location_pings
                    GROUP BY tech_username
                ) latest
                ON lp.tech_username = latest.tech_username
                AND lp.timestamp = latest.max_ts
            """)
        ).fetchall()

        return pd.DataFrame(rows, columns=[
            "tech_username", "latitude", "longitude", "timestamp"
        ])
    finally:
        s.close()



def classify_tech_status(ts):
    if ts is None:
        return "offline"

    if isinstance(ts, str):
        try:
            ts = pd.to_datetime(ts)
        except Exception:
            return "offline"

    now = datetime.utcnow()
    delta = (now - ts).total_seconds() / 60

    if delta <= 2:
        return "🟢 Live"
    elif delta <= 10:
        return "🟡 Idle"
    else:
        return "🔴 Offline"
#-----------TIME WINDOW--------------

from datetime import datetime, timedelta

def resolve_time_window(range_key: str, custom_start=None, custom_end=None):
    now = datetime.utcnow()

    if range_key == "daily":
        return now - timedelta(days=1), now
    if range_key == "weekly":
        return now - timedelta(days=7), now
    if range_key == "30d":
        return now - timedelta(days=30), now
    if range_key == "90d":
        return now - timedelta(days=90), now
    if range_key == "6m":
        return now - timedelta(days=180), now
    if range_key == "12m":
        return now - timedelta(days=365), now
    if range_key == "custom" and custom_start and custom_end:
        return custom_start, custom_end

    # fallback
    return now - timedelta(days=30), now



# ---------- END BLOCK C ----------




def upsert_lead_record(payload: dict, actor="admin"):
    """
    payload must include lead_id (string)
    other fields optional
    """
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.lead_id == payload.get("lead_id")).first()
        if lead is None:
            # create
            lead = Lead(
                lead_id=payload.get("lead_id"),
                created_at=payload.get("created_at", datetime.utcnow()),
                source=payload.get("source"),
                source_details=payload.get("source_details"),
                contact_name=payload.get("contact_name"),
                contact_phone=payload.get("contact_phone"),
                contact_email=payload.get("contact_email"),
                property_address=payload.get("property_address"),
                damage_type=payload.get("damage_type"),
                assigned_to=payload.get("assigned_to"),
                notes=payload.get("notes"),
                estimated_value=float(payload.get("estimated_value") or 0.0),
                stage=payload.get("stage") or "New",
                sla_hours=int(payload.get("sla_hours") or DEFAULT_SLA_HOURS),
                sla_entered_at=payload.get("sla_entered_at") or datetime.utcnow(),
                ad_cost=float(payload.get("ad_cost") or 0.0),
                converted=bool(payload.get("converted") or False),
                score=payload.get("score")
            )
            s.add(lead)
            s.commit()
            s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor, field="create", old_value=None, new_value=str(lead.stage)))
            s.commit()
            return lead.lead_id
        else:
            # update fields and log changes
            changed = []
            for key in ["source","source_details","contact_name","contact_phone","contact_email","property_address",
                        "damage_type","assigned_to","notes","estimated_value","stage","sla_hours","sla_entered_at","ad_cost","converted","score"]:
                if key in payload:
                    new = payload.get(key)
                    old = getattr(lead, key)
                    # normalize numeric conversions
                    if key in ("estimated_value","ad_cost"):
                        try:
                            new_val = float(new or 0.0)
                        except Exception:
                            new_val = old
                    elif key in ("sla_hours",):
                        try:
                            new_val = int(new or old)
                        except Exception:
                            new_val = old
                    elif key in ("converted",):
                        new_val = bool(new)
                    else:
                        new_val = new
                    if new_val is not None and old != new_val:
                        changed.append((key, old, new_val))
                        setattr(lead, key, new_val)
            # persist
            s.add(lead)
            for (f, old, new) in changed:
                s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor, field=f, old_value=str(old), new_value=str(new)))
            s.commit()
            return lead.lead_id
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def delete_lead_record(lead_id: str, actor="admin"):
    s = get_session()
    try:
        lead = s.query(Lead).filter(Lead.lead_id == lead_id).first()
        if not lead:
            return False
        s.add(LeadHistory(lead_id=lead.lead_id, changed_by=actor, field="delete", old_value=str(lead.stage), new_value="deleted"))
        s.delete(lead)
        s.commit()
        return True
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_users_df():
    s = get_session()
    try:
        users = s.query(User).order_by(User.created_at.desc()).all()
        data = [{"id":u.id, "username":u.username, "full_name":u.full_name, "role":u.role, "created_at":u.created_at} for u in users]
        return pd.DataFrame(data)
    finally:
        s.close()
def get_leads_for_task_dropdown():
    """
    Returns DataFrame of leads for task assignment dropdown
    """
    s = get_session()
    try:
        rows = (
            s.query(
                Lead.lead_id,
                Lead.contact_name,
                Lead.property_address
            )
            .order_by(Lead.created_at.desc())
            .all()
        )

        return pd.DataFrame([
            {
                "lead_id": r.lead_id,
                "label": f"{r.lead_id} — {r.contact_name or 'No Name'} ({r.property_address or 'No Address'})"
            }
            for r in rows
        ])
    finally:
        s.close()


def add_user(username: str, full_name: str = "", role: str = "Admin"):
    s = get_session()
    try:
        existing = s.query(User).filter(User.username == username).first()
        if existing:
            existing.full_name = full_name
            existing.role = role
            s.add(existing); s.commit()
            return existing.username
        u = User(username=username, full_name=full_name, role=role)
        s.add(u); s.commit()
        return u.username
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


# ----------------------
# ML - internal only
# ----------------------
def train_internal_model():
    df = leads_to_df()
    if df.empty or df["converted"].nunique() < 2:
        return None, "Not enough labeled data to train"
    df2 = df.copy()
    df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
    X = pd.get_dummies(df2[["source","stage"]].astype(str), drop_first=False)
    X["ad_cost"] = df2["ad_cost"]
    X["estimated_value"] = df2["estimated_value"]
    X["age_days"] = df2["age_days"]
    y = df2["converted"].astype(int)
    X = X.fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=120, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    joblib.dump({"model": model, "columns": X.columns.tolist()}, MODEL_FILE)
    return acc, "trained"


def load_internal_model():
    if not os.path.exists(MODEL_FILE):
        return None, None
    try:
        obj = joblib.load(MODEL_FILE)
        return obj.get("model"), obj.get("columns")
    except Exception:
        return None, None


def score_dataframe(df, model, cols):
    if model is None or df.empty:
        df["score"] = np.nan
        return df
    df2 = df.copy()
    df2["age_days"] = (datetime.utcnow() - df2["created_at"]).dt.days
    X = pd.get_dummies(df2[["source","stage"]].astype(str), drop_first=False)
    X["ad_cost"] = df2["ad_cost"]
    X["estimated_value"] = df2["estimated_value"]
    X["age_days"] = df2["age_days"]
    for c in cols:
        if c not in X.columns:
            X[c] = 0
    X = X[cols].fillna(0)
    try:
        df["score"] = model.predict_proba(X)[:,1]
    except Exception:
        df["score"] = model.predict(X)
    return df


# ----------------------
# Priority & SLA utilities
# ----------------------
def calculate_remaining_sla(sla_entered_at, sla_hours):
    try:
        if sla_entered_at is None:
            sla_entered_at = datetime.utcnow()
        if isinstance(sla_entered_at, str):
            sla_entered_at = datetime.fromisoformat(sla_entered_at)
        deadline = sla_entered_at + timedelta(hours=int(sla_hours or DEFAULT_SLA_HOURS))
        remain = deadline - datetime.utcnow()
        return max(remain.total_seconds(), 0.0), (remain.total_seconds() <= 0)
    except Exception:
        return float("inf"), False


def compute_priority_for_row(row, weights=None):
    # row: Series/dict
    if weights is None:
        weights = {"score_w":0.6, "value_w":0.3, "sla_w":0.1, "value_baseline":5000.0}
    try:
        s = float(row.get("score") or 0.0)
    except Exception:
        s = 0.0
    try:
        val = float(row.get("estimated_value") or 0.0)
        vnorm = min(1.0, val / max(1.0, weights["value_baseline"]))
    except Exception:
        vnorm = 0.0
    try:
        sla_entered = row.get("sla_entered_at") or row.get("created_at")
        if sla_entered is None:
            sla_score = 0.0
        else:
            if isinstance(sla_entered, str):
                sla_entered = datetime.fromisoformat(sla_entered)
            time_left_h = max((sla_entered + timedelta(hours=row.get("sla_hours") or DEFAULT_SLA_HOURS) - datetime.utcnow()).total_seconds()/3600.0, 0.0)
            sla_score = max(0.0, (72.0 - min(time_left_h,72.0)) / 72.0)
    except Exception:
        sla_score = 0.0
    total = s*weights["score_w"] + vnorm*weights["value_w"] + sla_score*weights["sla_w"]
    return max(0.0, min(1.0, total))


# ----------------------
# UI CSS and layout
# ----------------------
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.markdown(f"<link href='{COMFORTAA_IMPORT}' rel='stylesheet'>", unsafe_allow_html=True)


APP_CSS = """
<style>
body, .stApp { background: #ffffff; color: #0b1220; font-family: 'Comfortaa', sans-serif; }
.header { font-weight:800; font-size:20px; margin-bottom:6px; }
.kpi-grid { display:flex; gap:12px; flex-wrap:wrap; }
.kpi-card {
    background:#000;
    color:white;
    border-radius:12px;
    padding:12px;
    min-width:220px;
    box-shadow:0 8px 22px rgba(16,24,40,0.06);


    /* ✅ add spacing */
    margin-top: 12px;
    margin-bottom: 12px;
}


.kpi-title { font-size:12px; opacity:0.95; margin-bottom:6px; }
.kpi-number { font-size:22px; font-weight:900; margin-bottom:8px; }
.progress-bar { height:8px; border-radius:8px; background:#e6e6e6; overflow:hidden; }
.progress-fill { height:100%; border-radius:8px; transition:width .4s ease; }
.lead-card {
    background: #000000; /* ✅ Black */
    color: #ffffff; /* text stays white */
    border:1px solid #eef2ff;
    border-radius:10px;
    padding:12px;
    margin-bottom:8px;
}


.priority-time { color:#dc2626; font-weight:700; }
.priority-money { color:#22c55e; font-weight:800; }
.alert-bubble { background:#111; color:white; padding:10px; border-radius:10px; }
.small-muted { color:#F5F5F5; font-size:12px; }
</style>
"""
st.markdown(APP_CSS, unsafe_allow_html=True)


# ----------------------
# Sidebar controls (Admin backend - no front login)
# ----------------------
with st.sidebar:
    st.header("TITAN Backend (Admin)")
    st.markdown("You are using the backend admin interface. User accounts and roles are managed in Settings.")
    page = st.radio(
    "Navigate", 
    [
        "Overview",
        "Lead Capture",
        "Pipeline Board",
        "Analytics",
        "CPA & ROI",
        "ML (internal)",
        "Technician Mobile",
        "Technician Map Tracking",   # << added this 
        "Tasks",
        "AI Recommendations",
        "Seasonal Trends",
        "Competitor Intelligence",
        "Settings",
        "Exports"
    ], 
    index=0
)


    st.markdown("---")
    st.markdown("Date range for reports")
    quick = st.selectbox("Quick range", ["Today","Last 7 days","Last 30 days","90 days","All","Custom"], index=4)
    if quick == "Today":
        st.session_state.start_date = date.today()
        st.session_state.end_date = date.today()
    elif quick == "Last 7 days":
        st.session_state.start_date = date.today() - timedelta(days=6)
        st.session_state.end_date = date.today()
    elif quick == "Last 30 days":
        st.session_state.start_date = date.today() - timedelta(days=29)
        st.session_state.end_date = date.today()
    elif quick == "90 days":
        st.session_state.start_date = date.today() - timedelta(days=89)
        st.session_state.end_date = date.today()
    elif quick == "All":
        st.session_state.start_date = None
        st.session_state.end_date = None
    else:
        sd, ed = st.date_input("Start / End", [date.today() - timedelta(days=29), date.today()])
        st.session_state.start_date = sd
        st.session_state.end_date = ed


    st.markdown("---")
    st.markdown("Internal ML runs silently. Use ML page to train/score.")
    if st.button("Refresh data"):
        # clear caches and refresh
        try:
            st.rerun()
        except Exception:
            pass


# Utility: date filters
start_dt = st.session_state.get("start_date", None)
end_dt = st.session_state.get("end_date", None)


# Load leads
try:
    leads_df = leads_to_df(start_dt, end_dt)
except OperationalError as exc:
    st.error("Database error — ensure file is writable and accessible.")
    st.stop()


# Load model (if exists)
model, model_cols = load_internal_model()
if model is not None and not leads_df.empty:
    try:
        leads_df = score_dataframe(leads_df.copy(), model, model_cols)
    except Exception:
        # if scoring fails, continue without scores
        pass


# ----------------------
# Alerts bell (top-right)
# ----------------------
def alerts_ui():
    overdue = []
    for _, r in leads_df.iterrows():
        rem_s, overdue_flag = calculate_remaining_sla(
            r.get("sla_entered_at") or r.get("created_at"),
            r.get("sla_hours")
        )
        if overdue_flag and r.get("stage") not in ("Won", "Lost"):
            overdue.append(r)


    if overdue:
        col1, col2 = st.columns([1, 10])


        # BELL BUTTON (needs unique key)
        with col1:
            if st.button(f"🔔 {len(overdue)}", key="alerts_button"):
                st.session_state.show_alerts = not st.session_state.get(
                    "show_alerts", False
                )


        with col2:
            st.markdown("")


        # EXPANDED ALERTS POPUP
        if st.session_state.get("show_alerts", False):
            with st.expander("SLA Alerts (click to close)", expanded=True):


                # list overdue leads
                for r in overdue:
                    st.markdown(
                        f"**{r['lead_id']}** — Stage: {r['stage']} — "
                        f"<span style='color:#22c55e;'>${r['estimated_value']:,.0f}</span> — "
                        f"<span style='color:#dc2626;'>OVERDUE</span>",
                        unsafe_allow_html=True
                    )


                # CLOSE BUTTON (needs unique key)
                if st.button("Close Alerts", key="alerts_close_button"):
                    st.session_state.show_alerts = False


    else:
        st.markdown("")




def page_overview():
    import plotly.express as px
    st.markdown("<div class='header'>TOTAL LEAD PIPELINE — KEY PERFORMANCE INDICATOR</div>", unsafe_allow_html=True)
    st.markdown("<em>High-level pipeline performance at a glance. Use filters and cards to drill into details.</em>", unsafe_allow_html=True)


    # Call alerts_ui() like your design; support both signatures
    try:
        alerts_ui()
    except TypeError:
        try:
            alerts_ui(leads_df if 'leads_df' in globals() else leads_to_df())
        except Exception:
            pass


    # Prefer leads_df if present, otherwise fall back to leads_to_df()
    try:
        if 'leads_df' in globals() and isinstance(leads_df, pd.DataFrame):
            df = leads_df.copy()
        else:
            df = leads_to_df()
    except Exception:
        df = leads_to_df()


    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame()


    total_leads = len(df)
    qualified_leads = int(df[df["qualified"] == True].shape[0]) if not df.empty else 0
    sla_success_count = int(df[df["contacted"] == True].shape[0]) if not df.empty else 0
    awarded_count = int(df[df["stage"] == "Won"].shape[0]) if not df.empty else 0
    lost_count = int(df[df["stage"] == "Lost"].shape[0]) if not df.empty else 0
    closed = awarded_count + lost_count
    conversion_rate = (awarded_count / closed * 100) if closed else 0.0
    inspection_count = int(df[df["inspection_scheduled"] == True].shape[0]) if not df.empty else 0
    inspection_pct = (inspection_count / qualified_leads * 100) if qualified_leads else 0.0
    estimate_sent_count = int(df[df["estimate_submitted"] == True].shape[0]) if not df.empty else 0
    pipeline_job_value = float(df["estimated_value"].sum()) if not df.empty else 0.0
    active_leads = total_leads - (awarded_count + lost_count)
    sla_success_pct = (sla_success_count / total_leads * 100) if total_leads else 0.0
    qualification_pct = (qualified_leads / total_leads * 100) if total_leads else 0.0


    try:
        KPI_COLORS
    except Exception:
        KPI_COLORS = ["#0ea5e9","#34d399","#f59e0b","#ef4444","#8b5cf6","#06b6d4","#10b981"]


    KPI_ITEMS = [
        ("Active Leads", f"{active_leads}", KPI_COLORS[0], "Leads currently in pipeline"),
        ("SLA Success", f"{sla_success_pct:.1f}%", KPI_COLORS[1], "Leads contacted within SLA"),
        ("Qualification Rate", f"{qualification_pct:.1f}%", KPI_COLORS[2], "Leads marked qualified"),
        ("Conversion Rate", f"{conversion_rate:.1f}%", KPI_COLORS[3], "Won / Closed"),
        ("Inspections Booked", f"{inspection_pct:.1f}%", KPI_COLORS[4], "Qualified → Scheduled"),
        ("Estimates Sent", f"{estimate_sent_count}", KPI_COLORS[5], "Estimates submitted"),
        ("Pipeline Job Value", f"${pipeline_job_value:,.0f}", KPI_COLORS[6], "Total pipeline job value")
    ]


    # render 2 rows: first 4 then next 3
    r1 = st.columns(4)
    r2 = st.columns(3)
    cols = r1 + r2
    for col, (title, value, color, note) in zip(cols, KPI_ITEMS):
        pct = min(100, max(10, (abs(hash(title)) % 80) + 20))
        col.markdown(f"""
            <div class='kpi-card'>
              <div class='kpi-title'>{title}</div>
              <div class='kpi-number' style='color:{color};'>{value}</div>
              <div class='progress-bar'><div class='progress-fill' style='width:{pct}%; background:{color};'></div></div>
              <div class='small-muted'>{note}</div>
            </div>
        """, unsafe_allow_html=True)


    st.markdown("---")
    st.markdown("### Lead Pipeline Stages")
    st.markdown("<em>Distribution of leads across pipeline stages.</em>", unsafe_allow_html=True)


    if df.empty:
        st.info("No leads yet. Create one in Lead Capture.")
    else:
        try:
            stages = PIPELINE_STAGES
        except Exception:
            stages = ["New","Contacted","Inspection Scheduled","Inspection","Estimate Sent","Won","Lost"]
        stage_counts = df["stage"].value_counts().reindex(stages, fill_value=0)
        pie_df = pd.DataFrame({"status": stage_counts.index, "count": stage_counts.values})
        try:
            fig = px.pie(pie_df, names="status", values="count", hole=0.45, color="status")
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        except Exception:
            st.bar_chart(stage_counts)


        st.markdown("---")


    # TOP 5 PRIORITY LEADS
    st.markdown("---")
    st.markdown("### TOP 5 PRIORITY LEADS")
    st.markdown("<em>Highest urgency leads by priority score (0–1). Address these first.</em>", unsafe_allow_html=True)
    if df.empty:
        st.info("No priority leads to display.")
    else:
        if 'compute_priority_for_row' in globals():
            scorer = compute_priority_for_row
        else:
            def scorer(r):
                rem_s, overdue = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
                score = 0.0
                if overdue:
                    score += 0.6
                try:
                    score += min(0.4, float(r.get("estimated_value") or 0.0) / max(1.0, pipeline_job_value))
                except Exception:
                    pass
                return min(1.0, score)


        df = df.copy()
        df["priority_score"] = df.apply(lambda r: scorer(r), axis=1)
        pr_df = df.sort_values("priority_score", ascending=False).head(5)
        for _, r in pr_df.iterrows():
            sla_sec, overdue = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
            hleft = int(sla_sec / 3600) if sla_sec not in (None, float("inf")) else 9999
            sla_html = f"<span class='priority-time'>❗ OVERDUE</span>" if overdue else f"<span class='small-muted'>⏳ {hleft}h left</span>"
            val_html = f"<span class='priority-money'>${r['estimated_value']:,.0f}</span>"
            st.markdown(f"""
                <div class='lead-card'>
                  <div style='display:flex; justify-content:space-between; align-items:center;'>
                    <div>
                      <div style='font-weight:800;'>#{r['lead_id']} — {r.get('contact_name') or 'No name'}</div>
                      <div class='small-muted'>{r.get('damage_type') or ''} • {r.get('source') or ''}</div>
                    </div>
                    <div style='text-align:right;'>
                      <div style='font-size:20px; font-weight:900; color:#111;'>{r['priority_score']:.2f}</div>
                      <div style='margin-top:8px;'>{val_html}<br>{sla_html}</div>
                    </div>
                  </div>
                </div>
            """, unsafe_allow_html=True)


    st.markdown("---")
    st.markdown("### 📋 All Leads (expand a card to edit / change status)")
    st.markdown("<em>Expand a lead to edit details, change status, assign owner, and create estimates.</em>", unsafe_allow_html=True)

        # ---- SAFELY PREPARE "stages" ----
    if not df.empty and "stage" in df.columns:
        stages = sorted(df["stage"].dropna().unique().tolist())
    else:
        stages = []
    # ---------------------------------
    # Quick filters
    q1, q2, q3 = st.columns([3,2,3])
    with q1:
        search_q = st.text_input("Search (lead_id, contact name, address, notes)", key="overview_search")
    with q2:
        filter_src = st.selectbox("Source filter", options=["All"] + sorted(df["source"].dropna().unique().tolist()) if not df.empty else ["All"], key="overview_filter_src")
    with q3:
        filter_stage = st.selectbox("Stage filter", options=["All"] + stages, key="overview_filter_stage")


    df_view = df.copy()
    if search_q:
        sq = search_q.lower()
        df_view = df_view[df_view.apply(lambda r: sq in str(r.get("lead_id","")).lower() or sq in str(r.get("contact_name","")).lower() or sq in str(r.get("property_address","")).lower() or sq in str(r.get("notes","")).lower(), axis=1)]
    if filter_src and filter_src != "All":
        df_view = df_view[df_view["source"] == filter_src]
    if filter_stage and filter_stage != "All":
        df_view = df_view[df_view["stage"] == filter_stage]


    if df_view.empty:
        st.info("No leads to show.")
        return



# iterate leads (most recent first)


    # iterate leads (most recent first)
    for _, lead in df_view.sort_values("created_at", ascending=False).head(200).iterrows():
        exp_key = f"exp_{lead['lead_id']}"
        with st.expander(f"#{lead['lead_id']} — {lead.get('contact_name') or 'No name'} — {lead.get('stage')}", expanded=False):
            left, right = st.columns([3,1])
            with left:
                st.write(f"**Source:** {lead.get('source') or ''}  |  **Assigned:** {lead.get('assigned_to') or ''}")
                st.write(f"**Address:** {lead.get('property_address') or ''}")
                st.write(f"**Contact:** {lead.get('contact_name') or ''} / {lead.get('contact_phone') or ''} / {lead.get('contact_email') or ''}")
                st.write(f"**Notes:** {lead.get('notes') or ''}")
                st.write(f"**Created:** {lead.get('created_at')}")
            with right:
                sla_sec, overdue = calculate_remaining_sla(lead.get("sla_entered_at") or lead.get("created_at"), lead.get("sla_hours"))
                if overdue:
                    st.markdown("<div style='color:#dc2626;font-weight:700;'>❗ OVERDUE</div>", unsafe_allow_html=True)
                else:
                    try:
                        hours = int(sla_sec // 3600) if sla_sec is not None else 0
                        mins = int((sla_sec % 3600) // 60) if sla_sec is not None else 0
                    except Exception:
                        hours, mins = 0, 0
                    st.markdown(f"<div class='small-muted'>⏳ {hours}h {mins}m left</div>", unsafe_allow_html=True)
            # update form
            c1, c2 = st.columns(2)
            with st.form(f"update_{lead['lead_id']}", clear_on_submit=False):
                new_stage = st.selectbox("Status", PIPELINE_STAGES, index=PIPELINE_STAGES.index(lead.get("stage")) if lead.get("stage") in PIPELINE_STAGES else 0, key=f"stage_{lead['lead_id']}")
                new_assigned = st.text_input("Assigned to (username)", value=lead.get("assigned_to") or "", key=f"assigned_{lead['lead_id']}")
                new_est = st.number_input("Estimated value (USD)", value=float(lead.get("estimated_value") or 0.0), min_value=0.0, step=100.0, key=f"estval_{lead['lead_id']}")
                new_cost = st.number_input("Cost to acquire lead (USD)", value=float(lead.get("ad_cost") or 0.0), min_value=0.0, step=1.0, key=f"cost_{lead['lead_id']}")
                new_notes = st.text_area("Notes", value=lead.get("notes") or "", key=f"notes_{lead['lead_id']}")
                submitted = st.form_submit_button("Save changes", key=f"save_changes_{lead['lead_id']}")
                if submitted:
                    try:
                        upsert_lead_record({
                            "lead_id": lead["lead_id"],
                            "stage": new_stage,
                            "assigned_to": new_assigned or None,
                            "estimated_value": new_est,
                            "ad_cost": new_cost,
                            "notes": new_notes
                        }, actor="admin")
                        st.success("Lead updated")
                        st.rerun()
                    except Exception as e:
                        st.error("Failed to update lead: " + str(e))
                        st.write(traceback.format_exc())
            # Technician assignment (outside form)
            st.markdown("### Technician Assignment")
            techs_df = get_technicians_df(active_only=True)
            tech_options = [""] + (techs_df["username"].tolist() if not techs_df.empty else [])
            selected_tech = st.selectbox("Assign Technician (active)", options=tech_options, index=0, key=f"tech_select_{lead['lead_id']}")
            assign_notes = st.text_area("Assignment notes (optional)", value="", key=f"tech_notes_{lead['lead_id']}")
            if st.button(f"Assign Technician to {lead['lead_id']}", key=f"assign_btn_{lead['lead_id']}"):
                if not selected_tech:
                    st.error("Select a technician")
                else:
                    try:
                        create_inspection_assignment(lead_id=lead["lead_id"], technician_username=selected_tech, notes=assign_notes)
                        upsert_lead_record({
                            "lead_id": lead["lead_id"],
                            "inspection_scheduled": True,
                            "stage": "Inspection Scheduled"
                        }, actor="admin")
                        st.success(f"Assigned {selected_tech} to lead {lead['lead_id']}")
                        st.rerun()
                    except Exception as e:
                        st.error("Failed to assign: " + str(e))
    # end for








# ------------------------------------------------------------
# NEXT PAGE STARTS HERE
# ------------------------------------------------------------


def page_lead_capture():
    st.markdown("<div class='header'>📇 Lead Capture</div>", unsafe_allow_html=True)
    st.markdown("<em>Create or upsert a lead. All inputs are saved for reporting and CPA calculations.</em>", unsafe_allow_html=True)
    with st.form("lead_capture_form", clear_on_submit=True):
        lead_id = st.text_input("Lead ID", value=f"L{int(datetime.utcnow().timestamp())}")
        source = st.selectbox("Lead Source", ["Google Ads","Organic Search","Referral","Phone","Insurance","Facebook","Instagram","LinkedIn","Other"])
        source_details = st.text_input("Source details (UTM / notes)", placeholder="utm_source=google...")
        contact_name = st.text_input("Contact name")
        contact_phone = st.text_input("Contact phone")
        contact_email = st.text_input("Contact email")
        property_address = st.text_input("Property address")
        damage_type = st.selectbox("Damage type", ["water","fire","mold","contents","reconstruction","other"])
        assigned_to = st.text_input("Assigned to (username)")
        estimated_value = st.number_input("Estimated value (USD)", min_value=0.0, value=0.0, step=100.0)
        ad_cost = st.number_input("Cost to acquire lead (USD)", min_value=0.0, value=0.0, step=1.0)
        sla_hours = st.number_input("SLA hours (first response)", min_value=1, value=DEFAULT_SLA_HOURS, step=1)
        notes = st.text_area("Notes")
        submitted = st.form_submit_button("Create / Update Lead")
        if submitted:
            try:
                upsert_lead_record({
                    "lead_id": lead_id.strip(),
                    "created_at": datetime.utcnow(),
                    "source": source,
                    "source_details": source_details,
                    "contact_name": contact_name,
                    "contact_phone": contact_phone,
                    "contact_email": contact_email,
                    "property_address": property_address,
                    "damage_type": damage_type,
                    "assigned_to": assigned_to or None,
                    "estimated_value": float(estimated_value or 0.0),
                    "ad_cost": float(ad_cost or 0.0),
                    "sla_hours": int(sla_hours or DEFAULT_SLA_HOURS),
                    "sla_entered_at": datetime.utcnow(),
                    "notes": notes
                }, actor="admin")
                st.success(f"Lead {lead_id} saved.")
                st.rerun()
            except Exception as e:
                st.error("Failed to save lead: " + str(e))
                st.write(traceback.format_exc())


    st.markdown("---")
    st.subheader("Recent leads")
    df = leads_to_df(None, None)
    if df.empty:
        st.info("No leads yet.")
    else:
        st.dataframe(df.sort_values("created_at", ascending=False).head(50))

# Pipeline Board 
# =============================
# PIPELINE BOARD — HYBRID VIEW
# Priority Intelligence + Stage Overview
# =============================

def page_pipeline_board():
    st.markdown("<div class='header'>📊 Pipeline Intelligence Board</div>", unsafe_allow_html=True)
    st.markdown(
        "<em>Operational pipeline combining urgency, value, and stage visibility.</em>",
        unsafe_allow_html=True,
    )

    # ---------- LOAD DATA ----------
    leads_df = get_leads_df()  # must return SLA, stage, score, estimated_value, assigned_to

    st.subheader("🧱 Seasonal Damage Type Distribution")
    
    if "created_at" not in leads_df.columns:
        st.warning("⚠️ Seasonal analysis unavailable: missing created_at data, Please add new Leads.")
        return

    leads_df = leads_df.copy()
    leads_df["created_at"] = pd.to_datetime(leads_df["created_at"], errors="coerce")
    leads_df = leads_df.dropna(subset=["created_at"])
    
    if leads_df.empty:
        st.warning("⚠️ No valid dates available for seasonal analysis.")
        return
    
    leads_df["month"] = leads_df["created_at"].dt.month_name()

    
    damage_month = (
        leads_df
        .groupby(["month", "damage_type"])
        .size()
        .reset_index(name="jobs")
    )
    
    fig_damage = px.bar(
        damage_month,
        x="month",
        y="jobs",
        color="damage_type",
        title="Damage Types by Month (Seasonal Impact)"
    )
    
    st.plotly_chart(fig_damage, use_container_width=True)

    if leads_df.empty:
        st.info("No leads in pipeline yet.")
        return

    # ---------- DERIVED METRICS ----------
    now = datetime.utcnow()
    leads_df["sla_remaining_hr"] = (
        leads_df["sla_entered_at"] + pd.to_timedelta(leads_df["sla_hours"], unit="h") - now
    ).dt.total_seconds() / 3600

    leads_df["priority_score"] = (
        (1 - leads_df["sla_remaining_hr"].clip(lower=0) / leads_df["sla_hours"]) * 0.4
        + leads_df["score"].fillna(0) * 0.4
        + (leads_df["estimated_value"].fillna(0) / 20000).clip(0, 1) * 0.2
    )

    # ---------- URGENCY BANDS ----------
    def urgency_label(row):
        if row.sla_remaining_hr <= 6:
            return "🔴 Critical"
        if row.sla_remaining_hr <= 24:
            return "🟠 High"
        if row.sla_remaining_hr <= 48:
            return "🟡 Medium"
        return "🟢 Normal"

    leads_df["urgency"] = leads_df.apply(urgency_label, axis=1)

    # ---------- TOP PRIORITY QUEUE ----------
    st.markdown("## 🔥 Priority Queue (What needs attention now)")

    priority_df = (
        leads_df.sort_values("priority_score", ascending=False)
        .head(10)
        [[
            "urgency",
            "lead_id",
            "stage",
            "sla_remaining_hr",
            "assigned_to",
            "estimated_value",
            "score",
        ]]
    )

    st.dataframe(
        priority_df.style.format({
            "sla_remaining_hr": "{:.1f}h",
            "estimated_value": "${:,.0f}",
            "score": "{:.2f}",
        }),
        use_container_width=True,
    )

    # ---------- STAGE OVERVIEW ----------
    st.markdown("---")
    st.markdown("## 🧭 Pipeline by Stage")

    stage_summary = (
        leads_df.groupby("stage")
        .agg(
            leads=("lead_id", "count"),
            value=("estimated_value", "sum"),
            avg_score=("score", "mean"),
        )
        .reset_index()
    )

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            px.bar(
                stage_summary,
                x="stage",
                y="leads",
                title="Lead Volume by Stage",
            ),
            use_container_width=True,
        )

    with c2:
        st.plotly_chart(
            px.bar(
                stage_summary,
                x="stage",
                y="value",
                title="Pipeline Value by Stage",
            ),
            use_container_width=True,
        )

    # ---------- STAGE DETAIL TABLE ----------
    st.markdown("### 📋 Detailed Stage Breakdown")

    selected_stage = st.selectbox(
        "View stage details",
        sorted(leads_df["stage"].unique()),
    )

    stage_df = leads_df[leads_df["stage"] == selected_stage].sort_values(
        "priority_score", ascending=False
    )

    st.dataframe(
        stage_df[[
            "urgency",
            "lead_id",
            "assigned_to",
            "sla_remaining_hr",
            "estimated_value",
            "score",
        ]].style.format({
            "sla_remaining_hr": "{:.1f}h",
            "estimated_value": "${:,.0f}",
            "score": "{:.2f}",
        }),
        use_container_width=True,
    )

    # ---------- EXECUTIVE SUMMARY ----------
    st.markdown("---")
    st.markdown("## 🧠 Executive Pipeline Insight")

    critical = (leads_df.sla_remaining_hr <= 6).sum()
    total_value = leads_df.estimated_value.sum()
    avg_score = leads_df.score.mean()

    summary = (
        f"The pipeline currently contains **{len(leads_df)} active leads** with a total "
        f"estimated value of **${total_value:,.0f}**. "
        f"There are **{critical} leads** approaching SLA breach, requiring immediate action. "
        f"Overall conversion confidence remains {'strong' if avg_score >= 0.6 else 'moderate' if avg_score >= 0.4 else 'low'}, "
        f"with an average lead quality score of **{avg_score:.2f}**."
    )

    st.info(summary)


# Analytics page (donut + SLA line + overdue table)
def page_analytics():
    st.markdown("<div class='header'>📈 Analytics & SLA</div>", unsafe_allow_html=True)
    st.markdown("<em>Donut of pipeline stages + SLA overdue chart and table</em>", unsafe_allow_html=True)
    df = leads_df.copy()
    if df.empty:
        st.info("No leads to analyze.")
        return
    # =========================================================
    # TIME RANGE SELECTION
    # =========================================================
    range_key = st.selectbox(
        "Select Time Window",
        [
            "Today",
            "Last 7 Days",
            "Last 30 Days",
            "Last 90 Days",
            "Last 6 Months",
            "Last 12 Months",
            "Custom"
        ],
        index=2
    )
    
    custom_start = custom_end = None
    if range_key == "Custom":
        c1, c2 = st.columns(2)
        with c1:
            custom_start = st.date_input("Start Date")
        with c2:
            custom_end = st.date_input("End Date")
    
    # =========================================================
    # 🧠 BUSINESS INTELLIGENCE ENGINE
    # =========================================================
    intelligence = compute_business_intelligence(
        range_key,
        custom_start,
        custom_end
    )

    # =========================================================
    # 🧠 EXECUTIVE SUMMARY (AUTO-GENERATED)
    # =========================================================
    st.markdown("## 🧠 Executive Summary")
    
    for line in intelligence.get("executive_narrative", []):
        st.info(line)
    st.markdown("---")


    if df.empty:
        st.info("No leads to analyze.")
        return
    
    st.subheader("Job Volume")
    st.metric("Total Jobs", intelligence["volume"]["total_jobs"])
    st.metric("Job Trend vs Previous Period", f"{intelligence['volume']['trend']*100:.1f}%")


    #----------- Donut: pipeline stages-----------------
    stage_counts = df["stage"].value_counts().reindex(PIPELINE_STAGES, fill_value=0)
    pie_df = pd.DataFrame({"stage": stage_counts.index, "count": stage_counts.values})
    fig = px.pie(pie_df, names="stage", values="count", hole=0.45, color="stage")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    # SLA Overdue time series (last 30 days)
    st.subheader("SLA Overdue (last 30 days)")
    today = datetime.utcnow().date()
    days = [today - timedelta(days=i) for i in range(29, -1, -1)]
    ts = []
    for d in days:
        start_dt = datetime.combine(d, datetime.min.time())
        end_dt = datetime.combine(d, datetime.max.time())
        sub = df[(df["created_at"] >= start_dt) & (df["created_at"] <= end_dt)]
        overdue_cnt = 0
        for _, r in sub.iterrows():
            _, overdue = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
            if overdue and r.get("stage") not in ("Won","Lost"):
                overdue_cnt += 1
        ts.append({"date": d, "overdue": overdue_cnt})
    ts_df = pd.DataFrame(ts)
    fig2 = px.line(ts_df, x="date", y="overdue", markers=True, title="SLA Overdue Count (30d)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("---")
    st.subheader("Current Overdue Leads")
    overdue_rows = []
    for _, r in df.iterrows():
        _, overdue = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
        if overdue and r.get("stage") not in ("Won","Lost"):
            overdue_rows.append({"lead_id": r.get("lead_id"), "stage": r.get("stage"), "value": r.get("estimated_value"), "assigned_to": r.get("assigned_to")})
    if overdue_rows:
        st.dataframe(pd.DataFrame(overdue_rows))
    else:
        st.info("No overdue leads currently.")
        
def compute_business_health_score(df, prev_df):
    if df.empty:
        return {"score": 0, "breakdown": {}}

    # ---- Volume Momentum ----
    vol_now = len(df)
    vol_prev = len(prev_df) if not prev_df.empty else vol_now
    volume_score = min(1, vol_now / max(vol_prev, 1))

    # ---- Revenue Quality ----
    rev_now = df["estimated_value"].sum()
    rev_prev = prev_df["estimated_value"].sum() if not prev_df.empty else rev_now
    revenue_score = min(1, rev_now / max(rev_prev, 1))

    # ---- Mix Stability ----
    mix = df.groupby("damage_type")["estimated_value"].sum()
    mix_share = mix / mix.sum()
    mix_score = 1 - mix_share.max()  # higher = more diversified

    # ---- Efficiency ----
    efficiency_now = rev_now / max(vol_now, 1)
    efficiency_prev = (
        rev_prev / max(vol_prev, 1) if not prev_df.empty else efficiency_now
    )
    efficiency_score = min(1, efficiency_now / max(efficiency_prev, 1))

    # ---- Weighted Total ----
    score = round(
        (
            volume_score * 0.25
            + revenue_score * 0.30
            + mix_score * 0.20
            + efficiency_score * 0.25
        ) * 100,
        1,
    )

    return {
        "score": score,
        "breakdown": {
            "Volume Momentum": round(volume_score * 100, 1),
            "Revenue Quality": round(revenue_score * 100, 1),
            "Mix Stability": round(mix_score * 100, 1),
            "Efficiency": round(efficiency_score * 100, 1),
        },
    }

def generate_executive_narrative(data):
    """
    Hardened Executive Narrative Generator with:
    - Confidence scoring
    - Risk flags
    - Narrative health indicator
    - Versioning based on BI data completeness
    Returns a dict with:
      'lines': list of narrative dicts {'text':..., 'confidence':...}
      'risk_flags': list of strings
      'health_score': float (0-100)
      'version': str
    """

    narrative = []
    risk_flags = []
    health_score = 100  # Start at 100 and deduct for missing/incomplete data

    # -----------------------------
    # Safe data extraction
    # -----------------------------
    revenue = data.get("revenue", {})
    volume = data.get("volume", {})
    trends = data.get("trends", {})
    seasonal = data.get("seasonal", {})
    mix = data.get("mix", {})

    # -----------------------------
    # Determine version based on data completeness
    # -----------------------------
    total_keys = ["revenue", "volume", "trends", "seasonal", "mix"]
    available_keys = sum(1 for k in total_keys if data.get(k))
    if available_keys == len(total_keys):
        version = "Full BI"
    elif available_keys >= 3:
        version = "Partial BI"
    else:
        version = "Minimal BI"
        health_score -= 40  # penalty for sparse data

    # -----------------------------
    # Helper function for confidence
    # -----------------------------
    def confidence_for_metric(metric_value, max_score=100):
        """Assign confidence based on data validity"""
        if metric_value is None or metric_value == 0:
            return 40  # low confidence
        elif metric_value < 0:
            return 60
        else:
            return max_score

    # -----------------------------
    # Revenue Narrative
    # -----------------------------
    total_revenue = revenue.get("total", 0)
    revenue_trend = revenue.get("trend", 0)

    rev_conf = confidence_for_metric(total_revenue)
    if total_revenue > 0:
        if revenue_trend > 0:
            narrative.append({
                "text": f"Revenue performance is trending upward, with total revenue reaching ${total_revenue:,.0f}.",
                "confidence": rev_conf
            })
        elif revenue_trend < 0:
            narrative.append({
                "text": f"Revenue shows a slight decline, totaling ${total_revenue:,.0f}, indicating potential market softening.",
                "confidence": rev_conf
            })
            risk_flags.append("Revenue declining")
        else:
            narrative.append({
                "text": f"Revenue remains stable at ${total_revenue:,.0f} for the selected period.",
                "confidence": rev_conf
            })
    else:
        narrative.append({
            "text": "Revenue data is limited for the selected period, suggesting reduced activity or incomplete reporting.",
            "confidence": rev_conf
        })
        health_score -= 20

    # -----------------------------
    # Volume Narrative
    # -----------------------------
    total_jobs = volume.get("total", 0)
    volume_trend = volume.get("trend", 0)
    vol_conf = confidence_for_metric(total_jobs)

    if total_jobs > 0:
        if volume_trend > 0:
            narrative.append({
                "text": f"Job volume increased to {int(total_jobs)} jobs, reflecting growing demand.",
                "confidence": vol_conf
            })
        elif volume_trend < 0:
            narrative.append({
                "text": f"Job volume declined to {int(total_jobs)} jobs, indicating a slowdown in intake.",
                "confidence": vol_conf
            })
            risk_flags.append("Volume declining")
        else:
            narrative.append({
                "text": f"Job volume remained steady at {int(total_jobs)} jobs.",
                "confidence": vol_conf
            })
    else:
        narrative.append({
            "text": "Job volume data is minimal, which may reflect low intake or missing operational records.",
            "confidence": vol_conf
        })
        health_score -= 20

    # -----------------------------
    # Seasonal Insight Narrative
    # -----------------------------
    peak_month = seasonal.get("peak_month")
    low_month = seasonal.get("low_month")
    seasonal_conf = 80 if peak_month or low_month else 40

    if peak_month:
        narrative.append({
            "text": f"Seasonal analysis indicates peak operational activity around {peak_month}, aligning with historical demand patterns.",
            "confidence": 80
        })





def page_business_intelligence():
    # -----------------------------
    # 🧠 EXECUTIVE BUSINESS HEALTH
    # -----------------------------
    st.markdown("## 🧠 Business Health Score")
    
    health = data.get("health")
    
    if health:
        st.metric(
            "Overall Business Health",
            f"{health['score']} / 100",
            help="Composite score of volume, revenue quality, mix stability, and efficiency"
        )
    
        cols = st.columns(4)
        for i, (k, v) in enumerate(health.get("breakdown", {}).items()):
            cols[i].metric(k, f"{v}%")
    else:
        st.info("Business health score not available for this period.")
    
    # -----------------------------
    # 🧠 HEALTH SCORE INTERPRETATION
    # -----------------------------
    if health and "score" in health:
        if health["score"] >= 75:
            st.success("🟢 Business is strong and scaling efficiently.")
        elif health["score"] >= 55:
            st.warning("🟡 Business is stable but showing early pressure.")
        else:
            st.error("🔴 Business health is deteriorating. Immediate review recommended.")

    
    st.markdown("### 🔄 Period-over-Period Change")
    
    comp = data["comparison"]
    
    c1, c2, c3 = st.columns(3)
    
    def fmt(p):
        if p is None:
            return "N/A"
        return f"{p*100:+.1f}%"
    
    c1.metric("Jobs vs Last Period", fmt(comp["jobs_change_pct"]))
    c2.metric("Revenue vs Last Period", fmt(comp["revenue_change_pct"]))
    c3.metric("Avg Revenue / Job", fmt(comp["avg_rev_change_pct"]))
    
    
    st.markdown("### 🌦️ Seasonal Context")
    
    if data.get("seasonal"):
        for s in data["seasonal"]:
            st.warning(s)
    else:
        st.success("🟢 Performance is within expected seasonal range")
    

    # -----------------------------
    # TIME WINDOW CONTROLS
    # -----------------------------
    col1, col2 = st.columns([2, 3])

    with col1:
        range_key = st.selectbox(
            "Time Range",
            ["daily", "weekly", "30d", "90d", "6m", "12m", "custom"],
            index=2
        )

    with col2:
        custom_start, custom_end = None, None
        if range_key == "custom":
            custom_start = st.date_input("Start date")
            custom_end = st.date_input("End date")

    data = compute_business_intelligence(
        range_key,
        custom_start,
        custom_end
    )

    df = data.get("raw_df")

    if df is None or df.empty:
        st.warning("No jobs found for this period.")
        return

    # -----------------------------
    # JOB VOLUME & MIX
    # -----------------------------
    st.markdown("### 📊 Job Volume & Mix")

    v = data.get("volume", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Jobs", v.get("total_jobs", 0))
    c2.metric("Jobs / Day", v.get("jobs_per_day", 0))
    c3.metric("Lead Sources", len(v.get("lead_sources", {})))

    if v.get("job_types"):
        st.plotly_chart(
            px.bar(
                x=list(v["job_types"].keys()),
                y=list(v["job_types"].values()),
                labels={"x": "Job Type", "y": "Count"},
                title="Jobs by Type"
            ),
            use_container_width=True
        )

    # -----------------------------
    # REVENUE INTELLIGENCE
    # -----------------------------
    st.markdown("### 💰 Revenue Intelligence")

    r = data.get("revenue", {})

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${r.get('total_revenue', 0):,.0f}")
    c2.metric("Avg / Job", f"${r.get('avg_revenue_per_job', 0):,.0f}")
    c3.metric(
        "Revenue Risk",
        f"{int(r.get('revenue_concentration', 0) * 100)}% Top Dependency"
    )

    if r.get("revenue_by_job_type"):
        st.plotly_chart(
            px.pie(
                names=list(r["revenue_by_job_type"].keys()),
                values=list(r["revenue_by_job_type"].values()),
                title="Revenue Share by Job Type"
            ),
            use_container_width=True
        )
    # -----------------------------
    # JOB TYPE INTELLIGENCE
    # -----------------------------
    st.markdown("### 🧩 Job Type Intelligence")

    jt = data["job_types"]
    
    if jt:
        st.dataframe(
            jt["table"][
                ["damage_type", "jobs", "job_share", "revenue", "revenue_share", "avg_revenue"]
            ],
            use_container_width=True
        )
    
        c1, c2 = st.columns(2)
    
        with c1:
            st.plotly_chart(
                px.pie(
                    jt["table"],
                    names="damage_type",
                    values="jobs",
                    title="Job Volume Distribution"
                ),
                use_container_width=True
            )
    
        with c2:
            st.plotly_chart(
                px.pie(
                    jt["table"],
                    names="damage_type",
                    values="revenue",
                    title="Revenue Distribution by Job Type"
                ),
                use_container_width=True
            )
    
        if jt["insights"]:
            st.markdown("#### ⚠️ Job Type Signals")
            for i in jt["insights"]:
                st.warning(i)
    else:
        st.info("No job type data available for this period.")


    # -----------------------------
    # EFFICIENCY & LEVERAGE
    # -----------------------------
    st.markdown("### ⚙️ Efficiency & Leverage")

    ineff = data.get("efficiency", {}).get("high_volume_low_value_jobs", [])

    if ineff:
        st.warning(
            "⚠️ High-effort, low-return job types detected: "
            + ", ".join(ineff)
        )
    else:
        st.success("No major efficiency leaks detected in this period.")

    # -----------------------------
    # EXECUTIVE SIGNALS
    # -----------------------------
    st.markdown("### 🧠 Executive Signals")

    signals = data.get("signals", [])

    if not signals:
        st.info("No abnormal signals detected. Business activity is within normal range.")
    else:
        for s in signals:
            st.error(s)

    # -----------------------------
    # RAW DATA
    # -----------------------------
    with st.expander("📄 View Underlying Jobs Data"):
        st.dataframe(df, use_container_width=True)



    STATUS_COLORS = {
    "live": "green",
    "recent": "orange",
    "offline": "red",
}
def page_technician_map_tracking():
    st.markdown("## 🗺️ Technician Live Map")

    df = get_latest_location_pings()

    if df.empty:
        st.warning("No technician GPS data available.")
        return

    # Convert timestamp safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    def classify_status(ts):
        if pd.isna(ts):
            return "offline"
        mins = (datetime.utcnow() - ts).total_seconds() / 60
        if mins <= 10:
            return "active"
        elif mins <= 30:
            return "idle"
        return "offline"

    df["status"] = df["timestamp"].apply(classify_status)

    status_color = {
        "active": "green",
        "idle": "orange",
        "offline": "red"
    }

    # Center map
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles="OpenStreetMap"
    )

    for _, r in df.iterrows():
        folium.CircleMarker(
            location=[r["latitude"], r["longitude"]],
            radius=8,
            color=status_color[r["status"]],
            fill=True,
            fill_opacity=0.85,
            popup=f"""
            <b>Technician:</b> {r['tech_username']}<br>
            <b>Status:</b> {r['status']}<br>
            <b>Last Ping:</b> {r['timestamp']}
            """
        ).add_to(m)

    st_folium(m, height=600, use_container_width=True)


def page_technician_mobile():
    st.markdown("## 📍 Technician Live Location")

    tech_username = st.text_input("Your technician username")

    if not tech_username:
        st.info("Enter your username to begin tracking")
        return

    st.markdown(
        """
        <script>
        navigator.geolocation.getCurrentPosition(
            (pos) => {
                const lat = pos.coords.latitude;
                const lon = pos.coords.longitude;
                const acc = pos.coords.accuracy;

                fetch("/?_save_location=1", {
                    method: "POST",
                    headers: {"Content-Type": "application/json"},
                    body: JSON.stringify({
                        username: "%s",
                        lat: lat,
                        lon: lon,
                        accuracy: acc
                    })
                });
            }
        );
        </script>
        """ % tech_username,
        unsafe_allow_html=True
    )

    st.success("📡 Location sent (refreshes every time page loads)")


# CPA & ROI page
def page_cpa_roi():
    st.markdown("<div class='header'>💰 CPA & ROI</div>", unsafe_allow_html=True)
    st.markdown("<em>Total Marketing Spend vs Conversions and ROI calculations.</em>", unsafe_allow_html=True)
    df = leads_df.copy()
    if df.empty:
        st.info("No leads")
        return
    total_spend = float(df["ad_cost"].sum())
    won_df = df[df["stage"] == "Won"]
    conversions = len(won_df)
    cpa = (total_spend / conversions) if conversions else 0.0
    revenue = float(won_df["estimated_value"].sum())
    roi = revenue - total_spend
    roi_pct = (roi / total_spend * 100) if total_spend else 0.0
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Marketing Spend</div><div class='kpi-number' style='color:{KPI_COLORS[0]}'>${total_spend:,.2f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='kpi-card'><div class='kpi-title'>Conversions (Won)</div><div class='kpi-number' style='color:{KPI_COLORS[1]}'>{conversions}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='kpi-card'><div class='kpi-title'>CPA</div><div class='kpi-number' style='color:{KPI_COLORS[3]}'>${cpa:,.2f}</div></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='kpi-card'><div class='kpi-title'>ROI</div><div class='kpi-number' style='color:{KPI_COLORS[6]}'>${roi:,.2f} ({roi_pct:.1f}%)</div></div>", unsafe_allow_html=True)
    st.markdown("---")
    # chart: spend vs conversions by source
    agg = df.groupby("source").agg(total_spend=("ad_cost","sum"), conversions=("stage", lambda s: (s=="Won").sum())).reset_index()
    if not agg.empty:
        fig = px.bar(agg, x="source", y=["total_spend","conversions"], barmode="group", title="Total Spend vs Conversions by Source")
        st.plotly_chart(fig, use_container_width=True)


# ML internal page
def page_ml_internal():
    st.markdown("<div class='header'>🧠 Internal ML — Lead Scoring</div>", unsafe_allow_html=True)
    st.markdown("<em>Model runs internally and writes score back to leads. No user tuning exposed.</em>", unsafe_allow_html=True)
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
                st.write(traceback.format_exc())
    model, cols = load_internal_model()
    if model:
        st.success("Model available (internal)")
        if st.button("Score all leads and persist scores"):
            df = leads_to_df()
            scored = score_dataframe(df.copy(), model, cols)
            s = get_session()
            try:
                for _, r in scored.iterrows():
                    lead = s.query(Lead).filter(Lead.lead_id == r["lead_id"]).first()
                    if lead:
                        lead.score = float(r["score"])
                        s.add(lead)
                s.commit()
                st.success("Scores persisted to DB")
            except Exception as e:
                s.rollback()
                st.error("Failed to persist scores: " + str(e))
            finally:
                s.close()
        if st.checkbox("Preview top scored leads"):
            df = leads_to_df()
            scored = score_dataframe(df.copy(), model, cols).sort_values("score", ascending=False).head(20)
            st.dataframe(scored[["lead_id","source","stage","estimated_value","ad_cost","score"]])


# ---------- BEGIN BLOCK G: AI RECOMMENDATIONS PAGE ----------
def page_ai_recommendations():
    """AI Recommendations — cleaned, safe, and optimized."""
    import plotly.express as px
    from sqlalchemy import func


    st.markdown("<div class='header'>🤖 AI Recommendations</div>", unsafe_allow_html=True)
    st.markdown("<em>Heuristic recommendations and quick diagnostics for the pipeline.</em>", unsafe_allow_html=True)


    # Load leads defensively
    try:
        df = leads_to_df()
    except Exception as e:
        st.error(f"Failed to load leads: {e}")
        df = pd.DataFrame()


    if df.empty:
        st.info("No leads to analyze.")
        return


    # 1) Top overdue leads
    st.subheader("Top Overdue Leads")
    overdue_list = []
    for _, r in df.iterrows():
        rem_s, overdue_flag = calculate_remaining_sla(r.get("sla_entered_at") or r.get("created_at"), r.get("sla_hours"))
        if overdue_flag and r.get("stage") not in ("Won", "Lost"):
            overdue_list.append({
                "lead_id": r["lead_id"],
                "stage": r.get("stage"),
                "assigned_to": r.get("assigned_to"),
                "value": r.get("estimated_value") or 0.0,
                "overdue_seconds": rem_s
            })
    if overdue_list:
        over_df = pd.DataFrame(overdue_list).sort_values("value", ascending=False)
        # keep columns unique and friendly
        over_df = over_df.rename(columns={"lead_id": "Lead ID", "stage": "Stage", "assigned_to": "Assigned To", "value": "Est. Value", "overdue_seconds": "Overdue Seconds"})
        st.table(over_df[["Lead ID", "Stage", "Assigned To", "Est. Value"]].head(10))
    else:
        st.info("No overdue leads.")


    st.markdown("---")


    # 2) Pipeline Bottlenecks (stage counts)
    st.subheader("Pipeline Bottlenecks")
    try:
        stages = PIPELINE_STAGES
    except Exception:
        stages = ["New", "Contacted", "Inspection Scheduled", "Inspection", "Estimate Sent", "Won", "Lost"]


    stage_counts = df["stage"].value_counts().reindex(stages, fill_value=0)
    stage_df = stage_counts.reset_index()
    stage_df.columns = ["Stage", "Count"]        # ensure unique column names
    st.table(stage_df.head(10))


    # Small horizontal bar chart (plotly)
    try:
        fig = px.bar(stage_df, x="Count", y="Stage", orientation="h", title="Leads by Stage", height=300)
        fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass


    st.markdown("---")
 

    # 3) Technician workload (from assignments)
    st.subheader("Technician Workload (Assigned Inspections)")
    try:
        s = get_session()
        try:
            rows = s.query(InspectionAssignment.technician_username, func.count(InspectionAssignment.id)).group_by(InspectionAssignment.technician_username).all()
            if rows:
                tw = pd.DataFrame(rows, columns=["Technician", "Assigned Inspections"])
                tw = tw.sort_values("Assigned Inspections", ascending=False).reset_index(drop=True)
                st.table(tw)
            else:
                st.info("No assignments yet.")
        finally:
            s.close()
    except Exception as e:
        st.error("Failed to load assignments: " + str(e))


    st.markdown("---")


    # 4) Suggestions (simple heuristics)
    st.subheader("Suggested Actions")
    suggestions = []


    # Scheduled but unassigned
    scheduled = df[(df["inspection_scheduled"] == True)]
    scheduled_unassigned = scheduled[(scheduled["assigned_to"].isnull()) | (scheduled["assigned_to"] == "")]
    for _, r in scheduled_unassigned.iterrows():
        suggestions.append(f"Lead {r['lead_id']} is scheduled for inspection but has no assigned technician — assign ASAP.")


    # High-value not contacted
    high_uncontacted = df[(df["estimated_value"] >= 5000) & (df["contacted"] == False)]
    for _, r in high_uncontacted.iterrows():
        suggestions.append(f"High-value lead {r['lead_id']} (${int(r['estimated_value']):,}) not contacted — prioritize contact within SLA.")


    # Many leads stuck in same stage
    bottleneck = stage_df.sort_values("Count", ascending=False).iloc[0] if not stage_df.empty else None
    if bottleneck is not None and bottleneck["Count"] > max(5, len(df) * 0.2):
        suggestions.append(f"Stage '{bottleneck['Stage']}' has {int(bottleneck['Count'])} leads — check for process blockers in this stage.")


    # Show suggestions
    if suggestions:
        for sgt in suggestions[:15]:
            st.markdown(f"- {sgt}")
    else:
        st.markdown("No immediate suggestions. Pipeline looks healthy.")


    st.markdown("---")


    # 5) Quick export of problematic leads (CSV)
    st.subheader("Export: Problem Leads")
    try:
        problem_df = over_df if (len(over_df) > 0) else pd.DataFrame()
        if not problem_df.empty:
            csv = problem_df.to_csv(index=False)
            st.download_button("Download overdue leads (CSV)", data=csv, file_name="overdue_leads.csv", mime="text/csv")
        else:
            st.info("No overdue leads to export.")
    except Exception:
        pass


    # End of page




# Settings page: user & role management, weights (priority), audit trail
def page_settings():
    st.markdown("<div class='header'>⚙️ Settings & User Management</div>", unsafe_allow_html=True)
    st.markdown("<em>Add team users, set roles for role-based integration later.</em>", unsafe_allow_html=True)
    st.subheader("Users")
    users_df = get_users_df()
    with st.form("add_user_form"):
        uname = st.text_input("Username (unique)")
        fname = st.text_input("Full name")
        role = st.selectbox("Role", ["Admin","Estimator","Adjuster","Tech","Viewer"], index=0)
        if st.form_submit_button("Add / Update User"):
            if not uname:
                st.error("Username required")
            else:
                add_user(uname.strip(), full_name=fname.strip(), role=role)
                st.success("User saved")
                st.rerun()
    if not users_df.empty:
        st.dataframe(users_df)
    st.markdown("---")
    st.markdown("## 👷 Technician Management")
    
    with st.expander("➕ Add Technician", expanded=False):
        col1, col2 = st.columns(2)
    
        with col1:
            tech_username = st.text_input("Username (unique)")
            tech_name = st.text_input("Full Name")
            tech_phone = st.text_input("Phone Number")
    
        with col2:
            tech_role = st.selectbox(
                "Specialization",
                ["Estimator", "Restoration Tech", "Inspector", "Adjuster", "Other"]
            )
            tech_active = st.checkbox("Active", value=True)
    
        if st.button("Save Technician"):
            if not tech_username:
                st.error("Username is required")
            else:
                try:
                    add_technician(
                        tech_username.strip(),
                        full_name=tech_name.strip(),
                        phone=tech_phone.strip(),
                        specialization=tech_role,
                        active=tech_active
                    )
                    st.success("✅ Technician saved")
                    st.experimental_rerun()
                except Exception as e:
                    st.error("Failed to save technician: " + str(e))
    
    st.markdown("### 📋 Existing Technicians")
    
    tech_df = get_technicians_df(active_only=False)
    
    if tech_df.empty:
        st.info("No technicians added yet.")
    else:
        for _, row in tech_df.iterrows():
            col1, col2, col3 = st.columns([3,2,2])
    
            with col1:
                st.write(f"👷 **{row['full_name']}** (`{row['username']}`)")
    
            with col2:
                new_status = st.selectbox(
                    "Status",
                    ["available", "assigned", "enroute", "onsite", "completed"],
                    index=["available","assigned","enroute","onsite","completed"].index(
                        row.get("status","available")
                    ),
                    key=f"status_{row['username']}"
                )
    
            with col3:
                if st.button("Update", key=f"btn_{row['username']}"):
                    update_technician_status(row["username"], new_status)
                    st.success("Status updated")
                    st.rerun()




def page_technician_mobile():
    st.markdown("## 📱 Technician Mobile")

    techs = get_technicians_df(active_only=True)
    if techs.empty:
        st.warning("No active technicians found.")
        return

    tech = st.selectbox("Select Technician", techs["username"].tolist())

    st.markdown("### 🧾 My Tasks")
    tasks = get_tasks_for_user(tech)
    if tasks.empty:
        st.info("No tasks assigned.")
    else:
        for _, t in tasks.iterrows():
            st.checkbox(
                f"{t['title']} (Lead: {t['lead_id']})",
                value=(t["status"] == "done"),
                key=f"task_{t['id']}"
            )

    st.markdown("### 📍 Send Location Ping")
    lat = st.number_input("Latitude", format="%.6f")
    lon = st.number_input("Longitude", format="%.6f")

    if st.button("Send Location Ping"):
        persist_location_ping(tech, lat, lon)
        st.success("Location sent")

    # ---------- BEGIN BLOCK D: SETTINGS UI - TECHNICIANS MANAGEMENT ----------
    st.markdown("---")
    st.subheader("Technicians (Field Users)")
    tech_df = get_technicians_df(active_only=False)
    with st.form("add_technician_form"):
        t_uname = st.text_input("Technician username (unique)")
        t_name = st.text_input("Full name")
        t_phone = st.text_input("Phone")
        t_role_sel = st.selectbox("Specialization", ["Tech", "Estimator", "Adjuster", "Driver"], index=0)
        t_active = st.checkbox("Active", value=True)
        if st.form_submit_button("Add / Update Technician"):
            if not t_uname:
                st.error("Technician username required")
            else:
                try:
                    add_technician(
                    username=tech_username.strip(),
                    full_name=tech_name.strip(),
                    phone=tech_phone.strip(),
                    specialization=tech_role,
                    active=tech_active
                )

                except Exception as e:
                    st.error("Failed to save technician: " + str(e))
    if tech_df is not None and not tech_df.empty:
        st.dataframe(tech_df)
    else:
        st.info("No technicians yet.")


# ---------- END BLOCK D ----------


    st.subheader("Priority weight tuning (internal)")
    wscore = st.slider("Model score weight", 0.0, 1.0, 0.6, 0.05)
    wvalue = st.slider("Estimate value weight", 0.0, 1.0, 0.3, 0.05)
    wsla = st.slider("SLA urgency weight", 0.0, 1.0, 0.1, 0.05)
    baseline = st.number_input("Value baseline (for normalization)", value=5000.0)
    if st.button("Save weights"):
        st.session_state.weights = {"score_w": wscore, "value_w": wvalue, "sla_w": wsla, "value_baseline": baseline}
        st.success("Weights updated (in session)")


    st.markdown("---")
    st.subheader("Audit Trail")
    s = get_session()
    try:
        hist = s.query(LeadHistory).order_by(LeadHistory.timestamp.desc()).limit(200).all()
        if hist:
            hist_df = pd.DataFrame([{"lead_id":h.lead_id,"changed_by":h.changed_by,"field":h.field,"old":h.old_value,"new":h.new_value,"timestamp":h.timestamp} for h in hist])
            st.dataframe(hist_df)
        else:
            st.info("No audit entries yet.")
    finally:
        s.close()


# Exports page
def page_exports():
    st.markdown("<div class='header'>📤 Exports & Imports</div>", unsafe_allow_html=True)
    st.markdown("<em>Export leads, import CSV/XLSX. Imported rows upsert by lead_id.</em>", unsafe_allow_html=True)
    df = leads_to_df(None, None)
    if not df.empty:
        towrite = io.BytesIO()
        df.to_excel(towrite, index=False, engine="openpyxl")
        towrite.seek(0)
        b64 = base64.b64encode(towrite.read()).decode()
        href = f"data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}"
        st.markdown(f'<a href="{href}" download="leads_export.xlsx">Download leads_export.xlsx</a>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload leads (CSV/XLSX) for import/upsert", type=["csv","xlsx"])
    if uploaded:
        try:
            if uploaded.name.lower().endswith(".csv"):
                df_in = pd.read_csv(uploaded)
            else:
                df_in = pd.read_excel(uploaded)
            if "lead_id" not in df_in.columns:
                st.error("File must include a lead_id column")
            else:
                count = 0
                for _, r in df_in.iterrows():
                    try:
                        upsert_lead_record({
                            "lead_id": str(r["lead_id"]),
                            "created_at": pd.to_datetime(r.get("created_at")) if r.get("created_at") is not None else datetime.utcnow(),
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
                            "converted": bool(r.get("converted") or False)
                        }, actor="admin")
                        count += 1
                    except Exception:
                        continue
                st.success(f"Imported/Upserted {count} rows.")
        except Exception as e:
            st.error("Failed to import: " + str(e))


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
                return jsonify({"error":"missing fields (tech_username, latitude, longitude)"}), 400
            pid = persist_location_ping(tech_username=str(tech), latitude=float(lat), longitude=float(lon), lead_id=lead_id, accuracy=accuracy, timestamp=ts_parsed)
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
    import numpy as np
    import pandas as pd
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.markdown("## 🌦️ Seasonal Trends & Weather-Based Damage Insights")
    st.markdown(
        "<em>Analyze historical weather patterns, forecast damage risk, and receive strategic recommendations.</em>",
        unsafe_allow_html=True
    )

    # =========================================================
    # 1. LOCATION SELECTION
    # =========================================================
    countries = get_all_countries()
    country = st.selectbox("Country", [c["name"] for c in countries])
    country_code = next(c["code"] for c in countries if c["name"] == country)

    city_query = st.text_input("City (e.g. Miami, London, Toronto)")
    matches = search_cities(country_code, city_query)

    if not matches:
        st.info("Start typing a city name.")
        return

    labels = [
        f"{m['name']}, {m['admin1']}" if m["admin1"] else m["name"]
        for m in matches
    ]
    selected = st.selectbox("Select city", labels)
    chosen = matches[labels.index(selected)]

    st.success(f"📍 {selected}, {country}")

    # =========================================================
    # 2. CONTROLS
    # =========================================================
    hist_range = st.selectbox(
        "Historical window",
        ["3 months", "6 months", "12 months"],
        index=1
    )

    forecast_range = st.selectbox(
        "Forecast horizon",
        ["3 months", "6 months", "12 months"]
    )

    months = {"3 months": 3, "6 months": 6, "12 months": 12}[hist_range]
    forecast_months = {"3 months": 3, "6 months": 6, "12 months": 12}[forecast_range]

    if not st.button("Generate Insights"):
        return

    # =========================================================
    # 3. DATA FETCH
    # =========================================================
    with st.spinner("Generating insights..."):
        hist_df = fetch_weather(chosen["lat"], chosen["lon"], months)
        forecast_days = forecast_months * 30
        forecast_df = fetch_forecast_weather(chosen["lat"], chosen["lon"], forecast_days)

    # =========================================================
    # 4. SAFETY CHECKS
    # =========================================================
    if hist_df.empty:
        st.error("Historical weather data unavailable for this location.")
        return

    if forecast_df.empty:
        st.warning(
            "⚠️ Forecast limited by API. Longer-range outlook inferred from historical patterns."
        )
        forecast_df = hist_df.copy()

    # =========================================================
    # 5. FEATURE ENGINEERING (SINGLE PASS)
    # =========================================================
    for df_ in [hist_df, forecast_df]:
        df_["humidity_pct"] = np.clip(60 + df_["rainfall_mm"] * 0.3, 30, 100)
        df_["storm_flag"] = (df_["rainfall_mm"] >= 20).astype(int)
        df_["water_damage_prob"] = np.clip(df_["rainfall_mm"] / 120, 0, 1)
        df_["mold_prob"] = np.clip(df_["humidity_pct"] / 100, 0, 1)
        df_["roof_storm_prob"] = df_["storm_flag"]
        df_["freeze_burst_prob"] = np.clip((df_["temperature_c"] < 1).astype(int), 0, 1)

    # =========================================================
    # 6. DEMAND DISTRIBUTION & SEASON SCORE
    # =========================================================
    demand = {
        "Water Damage": float(forecast_df["water_damage_prob"].mean()),
        "Mold Remediation": float(forecast_df["mold_prob"].mean()),
        "Storm / Roof": float(forecast_df["roof_storm_prob"].mean()),
        "Freeze / Pipe Burst": float(forecast_df["freeze_burst_prob"].mean()),
    }
    demand = {k: max(0.0, min(v, 1.0)) for k, v in demand.items()}
    season_score = round(np.mean(list(demand.values())), 2)

    # =========================================================
    # 7. TIME WINDOWS (3 / 6 / 12 MONTHS)
    # =========================================================
    windows = add_time_windows(hist_df)

    for i, (label, wdf) in enumerate(windows.items()):
        st.markdown(f"**Last {label.upper()}**")
        col1, col2 = st.columns(2)
        with col1:
            fig_temp = px.line(
                wdf, x="date", y="temperature_c", title=f"Average Temperature ({label})"
            )
            st.plotly_chart(fig_temp, use_container_width=True, key=f"temp_chart_{i}")
        with col2:
            fig_hum = px.line(
                wdf, x="date", y="humidity_pct", title=f"Average Humidity ({label})"
            )
            st.plotly_chart(fig_hum, use_container_width=True, key=f"hum_chart_{i}")

    # =========================================================
    # 8. SEASONAL TREND ANALYSIS — HISTORY VS FORECAST
    # =========================================================
    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["rainfall_mm"], name="Rainfall (Historical)", mode="lines"
    ))
    trend_fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["rainfall_mm"], name="Rainfall (Forecast)", mode="lines", line=dict(dash="dash")
    ))
    trend_fig.add_trace(go.Scatter(
        x=hist_df["date"], y=hist_df["temperature_c"], name="Temperature (Historical)", mode="lines", yaxis="y2"
    ))
    trend_fig.add_trace(go.Scatter(
        x=forecast_df["date"], y=forecast_df["temperature_c"], name="Temperature (Forecast)", mode="lines", line=dict(dash="dash"), yaxis="y2"
    ))
    trend_fig.update_layout(
        xaxis_title="Date",
        yaxis=dict(title="Rainfall (mm)"),
        yaxis2=dict(title="Temperature (°C)", overlaying="y", side="right"),
        legend=dict(orientation="h"),
        height=450
    )
    st.plotly_chart(trend_fig, use_container_width=True, key="trend_fig")

    # =========================================================
    # 9. DAMAGE RISK TRENDS
    # =========================================================
    risk_fig = go.Figure()
    for col, label_ in [
        ("water_damage_prob", "Water Damage"),
        ("mold_prob", "Mold"),
        ("roof_storm_prob", "Storm / Roof"),
        ("freeze_burst_prob", "Freeze / Pipe Burst"),
    ]:
        risk_fig.add_trace(go.Scatter(
            x=hist_df["date"], y=hist_df[col], name=f"{label_} (History)", mode="lines"
        ))
        risk_fig.add_trace(go.Scatter(
            x=forecast_df["date"], y=forecast_df[col], name=f"{label_} (Forecast)", mode="lines", line=dict(dash="dash")
        ))
    risk_fig.update_layout(yaxis=dict(range=[0,1]), legend=dict(orientation="h"), height=450)
    st.plotly_chart(risk_fig, use_container_width=True, key="risk_fig")

    # =========================================================
    # 10. WEATHER CHARTS — HISTORICAL
    # =========================================================
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            px.line(hist_df, x="date", y="rainfall_mm", title="Historical Rainfall"),
            use_container_width=True,
            key="hist_rainfall"
        )
    with c2:
        st.plotly_chart(
            px.line(hist_df, x="date", y="temperature_c", title="Historical Temperature"),
            use_container_width=True,
            key="hist_temp"
        )

    # =========================================================
    # 11. EXECUTIVE SEASONAL INSIGHTS
    # =========================================================
    st.subheader("🧠 Executive Seasonal Insights")
    insights = generate_seasonal_insights(leads_df, hist_df)
    if not insights:
        st.info("No significant seasonal signals detected for the selected period.")
    else:
        for i, insight in enumerate(insights):
            st.info(insight, icon="💡")

    # =========================================================
    # 12. SUMMARY METRICS & NARRATIVE
    # =========================================================
    summary = {
        "avg_rain_hist": hist_df["rainfall_mm"].mean(),
        "avg_rain_forecast": forecast_df["rainfall_mm"].mean(),
        "avg_temp_hist": hist_df["temperature_c"].mean(),
        "avg_temp_forecast": forecast_df["temperature_c"].mean(),
        "avg_water_risk": forecast_df["water_damage_prob"].mean(),
        "avg_mold_risk": forecast_df["mold_prob"].mean(),
        "avg_storm_risk": forecast_df["roof_storm_prob"].mean(),
        "avg_freeze_risk": forecast_df["freeze_burst_prob"].mean(),
    }

    st.markdown("## 📝 Narrative Analysis")
    notes = []
    if summary["avg_rain_forecast"] > summary["avg_rain_hist"] * 1.15:
        notes.append("🌧️ Rainfall is trending above seasonal norms — water intrusion risk rising.")
    elif summary["avg_rain_forecast"] < summary["avg_rain_hist"] * 0.85:
        notes.append("🌤️ Rainfall below normal — flood exposure reduced.")
    else:
        notes.append("🌦️ Rainfall is seasonally stable.")

    if summary["avg_temp_forecast"] > summary["avg_temp_hist"] + 2:
        notes.append("🔥 Warmer temperatures may elevate mold risk.")
    elif summary["avg_temp_forecast"] < summary["avg_temp_hist"] - 2:
        notes.append("❄️ Colder temperatures increase freeze/pipe burst risk.")
    else:
        notes.append("🌡️ Temperatures remain within seasonal norms.")

    for n in notes:
        st.info(n)

    # =========================================================
    # ✅ 12a. EXECUTIVE NARRATIVE (INSERT HERE)
    # =========================================================
    st.markdown("## 🧠 Executive Summary")
    for line in data["executive_narrative"]:
        st.info(line)

    # =========================================================
    # 13. DAMAGE RISK OUTLOOK & RECOMMENDATIONS
    # =========================================================
    st.markdown("## 🧠 Damage Risk & Recommendations")
    risk_notes = []
    if summary["avg_water_risk"] > 0.55:
        risk_notes.append("💧 High likelihood of water damage events.")
    if summary["avg_mold_risk"] > 0.45:
        risk_notes.append("🦠 Elevated mold growth risk.")
    if summary["avg_storm_risk"] > 0.4:
        risk_notes.append("🌪️ Storm-related roof damage risk.")
    if summary["avg_freeze_risk"] > 0.3:
        risk_notes.append("❄️ Freeze/pipe burst risk present.")
    if not risk_notes:
        risk_notes.append("🟢 Overall weather-driven damage risk is low.")
    for r in risk_notes:
        st.warning(r)

    if season_score >= 0.6:
        st.error("🔥 Peak Season — increase staffing, pre-stage equipment, boost emergency marketing.")
    elif season_score >= 0.4:
        st.warning("⚠️ Elevated Activity — flexible scheduling and monitor leads closely.")
    else:
        st.success("🟢 Low / Normal Season — focus on branding, SEO, and internal optimization.")

    # =========================================================
    # 14. EXPECTED LEADS & STAFFING
    # =========================================================
    BASE_MONTHLY_LEADS = 40
    expected_total_leads = int(BASE_MONTHLY_LEADS * (0.6 + season_score) * forecast_months)
    st.markdown("## 🔢 Expected Lead Volume")
    st.success(f"📈 ~{expected_total_leads} jobs over {forecast_months} months")
    techs = max(1, int(np.ceil(expected_total_leads / (18 * forecast_months))))
    st.metric("Recommended Technicians", techs)



# ---------- BEGIN BLOCK E: PAGE – COMPETITOR INTELLIGENCE ----------
def page_competitor_intelligence():
    st.title("🏆 Competitor Intelligence")

    # ===============================
    # 🚨 COMPETITIVE ALERTS (TOP)
    # ===============================
    st.subheader("🚨 Competitive Alerts")

    s = get_session()
    alerts = (
        s.query(CompetitorAlert)
        .order_by(CompetitorAlert.created_at.desc())
        .limit(5)
        .all()
    )
    s.close()

    if not alerts:
        st.success("No competitive threats detected.")
    else:
        for i, a in enumerate(alerts):
            key = f"alert_{i}"
            if a.severity == "high":
                st.error(a.message, key=key)
            else:
                st.warning(a.message, key=key)

    st.divider()

    # ===============================
    # 🔍 COMPETITOR DISCOVERY
    # ===============================
    with st.expander("➕ Discover Competitors"):
        lat = st.number_input("Latitude", value=39.9612, key="comp_lat")
        lon = st.number_input("Longitude", value=-82.9988, key="comp_lon")
        keyword = st.text_input(
            "Search keyword",
            "water damage restoration",
            key="comp_keyword"
        )

        if st.button("Run Competitor Scan", key="run_comp_scan"):
            ingest_competitors_openstreetmap(lat, lon, keyword)
            st.success("Competitor scan completed.")

    # ===============================
    # 📊 FETCH COMPETITORS
    # ===============================
    s = get_session()
    try:
        competitors = s.query(Competitor).all()
    finally:
        s.close()

    if not competitors:
        st.info("No competitors tracked yet.")
        return

    # ===============================
    # 📊 BUILD COMPETITOR TABLE
    # ===============================
    rows = []
    hq_lat = st.session_state.get("hq_lat")
    hq_lon = st.session_state.get("hq_lon")

    for c in competitors:
        distance = 10
        if hq_lat and hq_lon and c.latitude and c.longitude:
            distance = haversine_km(hq_lat, hq_lon, c.latitude, c.longitude)

        score = calculate_competitor_score(
            c.rating or 0,
            c.total_reviews or 0,
            distance
        )

        rows.append({
            "Name": c.name,
            "Rating": c.rating,
            "Reviews": c.total_reviews,
            "Category": c.primary_category,
            "Distance (km)": round(distance, 2),
            "Strength Score": score,
            "Velocity (7d)": review_velocity(c.id, 7),
            "Velocity (30d)": review_velocity(c.id, 30),
        })

    df = pd.DataFrame(rows).sort_values(
        "Strength Score", ascending=False
    )

    st.subheader("Top Competitors")
    st.dataframe(df, use_container_width=True, key="competitor_table")

    # ===============================
    # 📊 MARKET PRESSURE
    # ===============================
    def market_pressure_score(df):
        if df.empty:
            return 0
        return round(
            df["Velocity (7d)"].mean() * 0.4 +
            df["Strength Score"].mean() * 60,
            1
        )

    pressure = market_pressure_score(df)

    st.metric(
        "Market Pressure Score",
        pressure,
        delta="Rising" if pressure > 60 else "Stable",
        key="market_pressure"
    )

    # ===============================
    # 📉 SEO VISIBILITY GAP
    # ===============================
    st.subheader("📉 SEO Visibility Gap")

    you_reviews = st.number_input("Your total reviews", value=120, key="you_reviews")
    you_rating = st.number_input("Your rating", value=4.6, key="you_rating")

    if not df.empty:
        gap = seo_visibility_gap(you_reviews, you_rating, df)

        if gap["pressure"] == "HIGH":
            st.error(
                f"Competitors average {gap['review_gap']} more reviews and "
                f"{gap['rating_gap']} higher rating. SEO pressure is HIGH.",
                key="seo_gap_high"
            )
        else:
            st.warning(
                "You are competitive, but review velocity must be maintained.",
                key="seo_gap_warn"
            )

    # ===============================
    # 🧠 EXECUTIVE COMPETITIVE SUMMARY
    # ===============================
    st.subheader("🧠 Executive Competitive Summary")

    if not df.empty:
        top = df.iloc[0]
        st.markdown(f"""
**Market Overview**

The local restoration market is currently under **{gap['pressure']} competitive pressure**.

**Key Threat**
- {top['Name']} leads the market with {top['Reviews']} reviews and rapid growth velocity.

**Risk Outlook**
- Continued review acceleration from competitors could reduce inbound lead share.
- Immediate review acquisition and proximity-focused SEO are recommended.

**Recommended Actions**
1. Launch review campaigns immediately
2. Optimize GMB categories and services
3. Increase local landing page coverage
""", unsafe_allow_html=True)
# ---------- END BLOCK E ----------

# ---------- END BLOCK E ----------


# ----------------------
# Router (main)
# ----------------------
if page == "Overview":
    page_overview()
elif page == "Lead Capture":
    page_lead_capture()
elif page == "Pipeline Board":
    page_pipeline_board()
elif page == "Analytics":
    page_analytics()
elif page == "Seasonal Trends":
    page_seasonal_trends()
elif page == "Competitor Intelligence":
    page_competitor_intelligence()
elif page == "CPA & ROI":
    page_cpa_roi()
elif page == "Business Intelligence":
    page_business_intelligence()
elif page == "AI Recommendations":
    page_ai_recommendations()
elif page == "ML (internal)":
    page_ml_internal()
elif page == "Technician Mobile":
    page_technician_mobile()
elif page == "Technician Map Tracking":
    page_technician_map_tracking()
elif page == "Technician Mobile":
    page_technician_mobile()
elif page == "Tasks":
    page_tasks()
elif page == "Settings":
    page_settings()
elif page == "Exports":
    page_exports()
else:
    st.info("Page not implemented yet.")


# Footer
st.markdown("---")
st.markdown("<div class='small-muted'>ReCapture Pro. SQLite persistence. Integrated Field Tracking (upgrade) enabled.</div>", unsafe_allow_html=True)
