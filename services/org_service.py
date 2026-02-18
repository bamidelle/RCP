from database.supabase_client import supabase

def create_organization(name, plan="starter"):
    response = supabase.table("organizations").insert({
        "name": name,
        "plan": plan
    }).execute()
    return response.data

def get_organizations():
    response = supabase.table("organizations").select("*").execute()
    return response.data
