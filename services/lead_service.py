# Lead Service Operations

class LeadService:
    def __init__(self, supabase_client):
        self.client = supabase_client

    def create_lead(self, lead_data):
        """Create a new lead in Supabase."""
        response = self.client.table('leads').insert(lead_data).execute()
        return response

    def get_leads(self, filters=None):
        """Retrieve leads from Supabase, optionally filtered by criteria."""
        query = self.client.table('leads')
        if filters:
            query = query.select().filter(filters)
        else:
            query = query.select()
        response = query.execute()
        return response

    def update_lead(self, lead_id, updated_data):
        """Update an existing lead in Supabase by its ID."""
        response = self.client.table('leads').update(updated_data).eq('id', lead_id).execute()
        return response

    def delete_lead(self, lead_id):
        """Delete a lead from Supabase by its ID."""
        response = self.client.table('leads').delete().eq('id', lead_id).execute()
        return response

    def find_lead(self, lead_id):
        """Find a lead by its ID."""
        response = self.client.table('leads').select().eq('id', lead_id).execute()
        return response

# Additional utility methods can be added as needed.