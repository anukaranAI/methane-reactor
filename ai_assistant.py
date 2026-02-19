import google.generativeai as genai
import streamlit as st

class GeminiAssistant:
    def __init__(self, api_key):
        self.model = None
        self.model_available = False
        
        if not api_key:
            st.error("⚠️ AI ERROR: Invalid API Key.")
            return

        try:
            genai.configure(api_key=api_key)
            all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            available_names = [m.replace("models/", "") for m in all_models]
            
            preferred_order = ["gemini-1.5-flash", "gemini-flash-latest", "gemini-1.5-pro", "gemini-pro"]
            target_model = next((m for m in preferred_order if m in available_names), available_names[0] if available_names else None)
            
            if target_model:
                self.model = genai.GenerativeModel(target_model)
                self.model_available = True
            
        except Exception as e:
            st.error(f"❌ AI Config Error: {e}")
            self.model_available = False

    def generate_response(self, user_message, context_str):
        if not self.model_available: return "AI is not available."
        prompt = (
            "You are an expert catalytic reactor engineering assistant for KinetiScale.\n"
            f"Simulation context: {context_str}\n\n"
            f"User request: {user_message}"
        )
        try:
            return self.model.generate_content(prompt).text
        except Exception as e:
            if "429" in str(e): return "⚠️ Quota Limit Exceeded. Please wait 30 seconds."
            return f"Error: {e}"