import streamlit as st

from services.invitation_service import (
    get_active_invitation,
    get_supabase_client,
    mark_invitation_used,
)

st.set_page_config(page_title="Accept Invite", page_icon="✅")
st.title("Accept your invitation")

params = st.query_params
token = params.get("token", "")

if not token:
    st.error("Missing invitation token.")
    st.stop()

invite = get_active_invitation(token)
if not invite:
    st.error("This invitation is invalid, expired, or already used.")
    st.stop()

st.info(f"Invited as role: {invite['role']}")

with st.form("accept_invite_form"):
    email = st.text_input("Email", value=invite["email"], disabled=True)
    full_name = st.text_input("Full name")
    password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm password", type="password")
    submitted = st.form_submit_button("Create account")

if submitted:
    if len(password) < 8:
        st.error("Password must be at least 8 characters.")
    elif password != confirm_password:
        st.error("Passwords do not match.")
    else:
        try:
            supabase = get_supabase_client(admin=False)
            result = supabase.auth.sign_up(
                {
                    "email": invite["email"],
                    "password": password,
                    "options": {
                        "data": {
                            "full_name": full_name,
                            "role": invite["role"],
                        }
                    },
                }
            )
            if not result or not result.user:
                st.error("Account creation failed.")
                st.stop()

            mark_invitation_used(
                token=token,
                accepted_user_id=result.user.id,
                user_email=invite["email"],
            )
            st.success("Account created and invitation accepted. You can now log in.")
        except Exception as err:
            st.error(f"Could not accept invitation: {err}")
