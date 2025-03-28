from supabase import create_client, Client

# Credenciales de Supabase
SUPABASE_URL = "https://zjdlmsjjzjxeogilzpam.supabase.co"  # Ejemplo: https://xyz.supabase.co
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpqZGxtc2pqemp4ZW9naWx6cGFtIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDMxNDAwNTksImV4cCI6MjA1ODcxNjA1OX0.KISWstPeEL_GYbTKGvACuyHwabACKZCck5Vp5dAI_6E"

# Crear cliente de Supabase
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
