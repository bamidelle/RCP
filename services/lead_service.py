import supabase from "@supabase/supabase-js";

class LeadService {
    constructor() {
        this.supabase = supabase.createClient(process.env.SUPABASE_URL, process.env.SUPABASE_ANON_KEY);
    }

    async createLead(data) {
        const { error } = await this.supabase
            .from('leads')
            .insert([data]);
        if (error) throw error;
        return 'Lead created successfully';
    }

    async getAllLeads() {
        const { data, error } = await this.supabase
            .from('leads')
            .select('*');
        if (error) throw error;
        return data;
    }

    async getLeadById(id) {
        const { data, error } = await this.supabase
            .from('leads')
            .select('*')
            .eq('id', id)
            .single();
        if (error) throw error;
        return data;
    }

    async updateLead(id, updates) {
        const { error } = await this.supabase
            .from('leads')
            .update(updates)
            .eq('id', id);
        if (error) throw error;
        return 'Lead updated successfully';
    }

    async deleteLead(id) {
        const { error } = await this.supabase
            .from('leads')
            .delete()
            .eq('id', id);
        if (error) throw error;
        return 'Lead deleted successfully';
    }
}

export default new LeadService();
