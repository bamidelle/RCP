from urllib.parse import quote

import streamlit as st

from services.invitation_service import create_invitation, send_invite_email, get_supabase_client

st.set_page_config(page_title="Admin Invites", page_icon="✉️")
st.title("Admin: Send Invitations")


def is_admin_user() -> bool:
    admin_emails = [e.strip().lower() for e in st.secrets.get("ADMIN_EMAILS", [])]
    if not admin_emails:
        return True

    try:
        supabase = get_supabase_client(admin=False)
        session = supabase.auth.get_session()
        if not session or not session.session or not session.session.user:
            return False
        return session.session.user.email.lower() in admin_emails
    except Exception:
        return False


if not is_admin_user():
    st.error("You must be an admin to send invitations.")
    st.stop()

base_url = st.secrets.get("APP_BASE_URL", "http://localhost:8501")
admin_identity = st.secrets.get("INVITE_SENT_BY", "admin")

with st.form("send_invite_form"):
    email = st.text_input("Invitee email")
    role = st.selectbox("Role", ["Admin", "Manager", "Staff"], index=2)
    expires_in_days = st.number_input("Expires in days", min_value=1, max_value=30, value=7)
    send_email = st.checkbox("Send email now", value=True)
    submitted = st.form_submit_button("Create invite")

if submitted:
    if not email.strip() or "@" not in email:
        st.error("Please enter a valid email address.")
    else:
        try:
            invite = create_invitation(
                email=email,
                role=role,
                invited_by=admin_identity,
                expires_in_days=int(expires_in_days),
            )
            invite_link = f"{base_url.rstrip('/')}/Accept_Invite?token={quote(invite['token'])}"
            st.success("Invitation created.")
            st.code(invite_link, language="text")

            if send_email:
                send_invite_email(email.strip().lower(), invite_link)
                st.success("Invitation email sent.")
        except Exception as err:
            st.error(f"Failed to create invite: {err}")
