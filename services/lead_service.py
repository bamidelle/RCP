from database.supabase_client import supabase
from datetime import datetime

def create_lead(organization_id, name, email, phone, status="new", value=0):
    response = supabase.table("leads").insert({
        "organization_id": organization_id,
        "name": name,
        "email": email,
        "phone": phone,
        "status": status,
        "value": value,
        "created_at": datetime.utcnow().isoformat()
    }).execute()

    return response


def get_leads_by_date(organization_id, start_date=None, end_date=None):
    query = supabase.table("leads").select("*").eq("organization_id", organization_id)

    if start_date:
        query = query.gte("created_at", start_date)

    if end_date:
        query = query.lte("created_at", end_date)

    response = query.order("created_at", desc=True).execute()
    return response.data
