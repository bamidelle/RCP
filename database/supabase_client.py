import streamlit as st
from supabase import create_client

SUPABASE_URL = st.secrets["https://nzpyakmlshlnvkpoizmq.supabase.coL"]
SUPABASE_ANON_KEY = st.secrets["eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im56cHlha21sc2hsbnZrcG9pem1xIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzA5OTMwNzgsImV4cCI6MjA4NjU2OTA3OH0.0fU7C9YCaw2m9avPlofCJVNK-JrDULlIHu9i35hMj-8"]

supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)
