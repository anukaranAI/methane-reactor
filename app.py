import streamlit as st
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from dataclasses import dataclass
import time
import google.generativeai as genai
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Anukaran AI - Reactor Sim",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# API CONFIGURATION
# ============================================================================
GEMINI_API_KEY = "AIzaSyDTJGfbX8BrxLrNpk6OYQbAVb7_eiIB5Us" 

# ============================================================================
# PHYSICAL CONSTANTS & HELPER FUNCTIONS (Restored Full Physics)
# ============================================================================
R_GAS = 8.314
MW_CH4 = 16.04e-3
MW_H2 = 2.016e-3
MW_C = 12.01e-3
MW_N2 = 28.01e-3

def gas_viscosity(T, y_CH4, y_H2, y_N2):
    # dynamic viscosity changes with Temp
    mu_CH4 = 1.02e-5 * (T / 300) ** 0.87
    mu_H2 = 8.76e-6 * (T / 300) ** 0.68
    mu_N2 = 1.78e-5 * (T / 300) ** 0.67
    return y_CH4 * mu_CH4 + y_H2 * mu_H2 + y_N2 * mu_N2

def gas_density(T, P, y_CH4, y_H2, y_N2):
    # ideal gas law density
    MW_mix = y_CH4 * MW_CH4 + y_H2 * MW_H2 + y_N2 * MW_N2
    return P * MW_mix / (R_GAS * T)

def diffusivity_CH4(T, P):
    # fuller-schettler-giddings correlation
    return 1.87e-5 * (T / 300) ** 1.75 * (101325 / P)

def heat_capacity_mix(T, y_CH4, y_H2, y_N2):
    # shomate equation approximation
    Cp_CH4 = 35.69 + 0.0275 * T
    Cp_H2 = 28.84 + 0.00192 * T
    Cp_N2 = 29.12 + 0.00293 * T
    return y_CH4 * Cp_CH4 + y_H2 * Cp_H2 + y_N2 * Cp_N2

def arrhenius_rate_constant(T, A, Ea, beta):
    return A * T ** beta * np.exp(-Ea / (R_GAS * T))

def effectiveness_factor(phi):
    if phi < 0.1: return 1.0
    elif phi > 100: return 3.0 / phi
    else: return (3.0 / phi) * (1.0 / np.tanh(phi) - 1.0 / phi)

def ergun_pressure_drop(u, rho, mu, d_p, eps):
    term1 = 150 * mu * (1 - eps)**2 / (d_p**2 * eps**3) * u
    term2 = 1.75 * rho * (1 - eps) / (d_p * eps**3) * u**2
    return term1 + term2

# ============================================================================
# AI ASSISTANT LOGIC (Robust)
# ============================================================================
class GeminiAssistant:
    def __init__(self, api_key):
        self.model = None
        self.model_available = False
        
        if not api_key or "PASTE_YOUR" in api_key:
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
            else:
                st.error("❌ No AI models found.")
                self.model_available = False
            
        except Exception as e:
            st.error(f"❌ AI Config Error: {e}")
            self.model_available = False

    def generate_response(self, user_message, context_str):
        if not self.model_available: return "AI is not available."
        prompt = (
            "You are an expert chemical reaction engineering assistant.\n"
            f"Simulation context: {context_str}\n\n"
            f"User request: {user_message}"
        )
        try:
            return self.model.generate_content(prompt).text
        except Exception as e:
            if "429" in str(e): return "⚠️ Quota limit exceeded. Please wait 30s."
            return f"Error: {e}"

# ============================================================================
# REACTOR LOGIC (Full Implementation)
# ============================================================================
@dataclass
class ReactorConfig:
    diameter: float; bed_height: float; particle_diameter: float; catalyst_density: float
    particle_porosity: float; tortuosity: float; bed_porosity: float; catalyst_mass: float
    inlet_temperature: float; inlet_pressure: float; flow_rate: float
    y_CH4_in: float; y_H2_in: float; y_N2_in: float
    pre_exponential: float; activation_energy: float; beta: float; heat_of_reaction: float
    
    @property
    def cross_section_area(self) -> float:
        return np.pi * (self.diameter / 2) ** 2

class MethaneDecompositionReactor:
    def __init__(self, config: ReactorConfig, isothermal: bool = True):
        self.cfg = config
        self.isothermal = isothermal
        
        # Calculate inlet molar concentration and flows
        C_total_in = config.inlet_pressure / (R_GAS * config.inlet_temperature) / 1000 # mol/m3 -> mol/L
        self.F_total_in = config.flow_rate * C_total_in
        self.F_CH4_in = config.y_CH4_in * self.F_total_in
        self.F_H2_in = config.y_H2_in * self.F_total_in
        self.F_N2_in = config.y_N2_in * self.F_total_in
    
    def _ode_system(self, z, y):
        F_CH4, F_H2, T, P = y
        cfg = self.cfg
        
        # Stability clamps
        F_CH4 = max(F_CH4, 1e-30); F_H2 = max(F_H2, 0.0); T = max(T, 300.0); P = max(P, 1000.0)
        
        # Mole fractions
        F_total = F_CH4 + F_H2 + self.F_N2_in
        y_CH4 = F_CH4 / F_total; y_H2 = F_H2 / F_total; y_N2 = self.F_N2_in / F_total
        
        # Physical properties (Using dynamic functions now)
        rho = gas_density(T, P, y_CH4, y_H2, y_N2)
        mu = gas_viscosity(T, y_CH4, y_H2, y_N2)
        
        # Flow velocity
        Q = F_total * 1000 * R_GAS * T / P # m3/s
        u = Q / cfg.cross_section_area
        
        # Kinetics
        C_CH4 = F_CH4 / Q
        D_mol = diffusivity_CH4(T, P)
        D_eff = D_mol * cfg.particle_porosity / cfg.tortuosity
        k = arrhenius_rate_constant(T, cfg.pre_exponential, cfg.activation_energy, cfg.beta)
        
        phi = (cfg.particle_diameter / 6) * np.sqrt(k / D_eff) if D_eff > 0 else 0
        eta = effectiveness_factor(phi)
        r_bed = k * eta * C_CH4 * (1 - cfg.bed_porosity)
        
        # Derivatives
        A = cfg.cross_section_area
        dF_CH4_dz = -1.0 * r_bed * A
        dF_H2_dz = +2.0 * r_bed * A
        
        # Energy Balance
        if self.isothermal:
            dT_dz = 0.0
        else:
            Cp_mix = heat_capacity_mix(T, y_CH4, y_H2, y_N2)
            F_total_mol = F_total * 1000
            Q_rxn = -cfg.heat_of_reaction * r_bed * A * 1000
            dT_dz = Q_rxn / (F_total_mol * Cp_mix + 1e-10)
            
        # Momentum Balance (Ergun)
        dP_dz = -ergun_pressure_drop(u, rho, mu, cfg.particle_diameter, cfg.bed_porosity)
        
        return np.array([dF_CH4_dz, dF_H2_dz, dT_dz, dP_dz])
        
    def solve(self, n_points=200):
        y0 = np.array([self.F_CH4_in, self.F_H2_in, self.cfg.inlet_temperature, self.cfg.inlet_pressure])
        solution = solve_ivp(
            self._ode_system, (0, self.cfg.bed_height), y0, 
            method='RK45', t_eval=np.linspace(0, self.cfg.bed_height, n_points),
            rtol=1e-8, atol=1e-12
        )
        
        # Unpack results
        z = solution.t
        F_CH4 = np.maximum(solution.y[0], 0)
        F_H2 = np.maximum(solution.y[1], 0)
        T = solution.y[2]
        P = solution.y[3]
        
        F_total = F_CH4 + F_H2 + self.F_N2_in
        
        return {
            'z': z,
            'F_CH4': F_CH4, 'F_H2': F_H2,
            'T': T, 'P': P,
            'X_CH4': np.clip((self.F_CH4_in - F_CH4) / self.F_CH4_in, 0, 1),
            'y_CH4': F_CH4/F_total,
            'y_H2': F_H2/F_total,
            'y_N2': self.F_N2_in/F_total,
            'V_dot_H2_Nm3_h': F_H2 * 1000 * R_GAS * 273.15 / 101325 * 3600
        }

# ============================================================================
# UI LAYOUT
# ============================================================================

# --- State Management ---
if 'simulation_data' not in st.session_state: st.session_state.simulation_data = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'config_data' not in st.session_state: st.session_state.config_data = None

# --- Helper: Handle AI ---
def handle_ai_request(prompt_text):
    st.session_state.chat_history.append({"role": "user", "content": prompt_text})
    
    context_str = "No simulation run yet."
    if st.session_state.simulation_data:
        r = st.session_state.simulation_data
        cfg = st.session_state.config_data
        context_str = (
            f"Reactor Config: L={cfg.bed_height}m, D={cfg.diameter}m. "
            f"Inlet: T={cfg.inlet_temperature-273.15:.0f}C, P={cfg.inlet_pressure/1e5:.1f}bar. "
            f"Key Results: Conversion={r['X_CH4'][-1]*100:.2f}%, "
            f"H2 Production={r['V_dot_H2_Nm3_h'][-1]:.4f} Nm3/h. "
            f"Isothermal Mode: {st.session_state.get('iso_check', True)}."
        )

    ai = GeminiAssistant(GEMINI_API_KEY)
    response = ai.generate_response(prompt_text, context_str)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# --- Header ---
st.title("🔬 ANUKARAN AI")
st.markdown("### Methane Decomposition Reactor Simulator")
st.markdown("---")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Input Parameters")
    
    with st.expander("Geometry", expanded=True):
        d_reac = st.number_input("Reactor Diameter (cm)", 1.0, 100.0, 5.0)
        h_bed = st.number_input("Bed Height (cm)", 1.0, 500.0, 20.0)
    
    with st.expander("Conditions", expanded=True):
        t_in = st.number_input("Inlet Temp (°C)", 300.0, 1500.0, 900.0)
        p_in = st.number_input("Inlet Pressure (bar)", 0.1, 100.0, 1.0)
        flow = st.number_input("Flow Rate (mL/min)", 10.0, 10000.0, 100.0)
    
    with st.expander("Advanced"):
        y_ch4 = st.number_input("CH4 Mole Frac", 0.0, 1.0, 0.2)
        st.session_state['iso_check'] = st.checkbox("Isothermal Mode", value=True)

    if st.button("▶️ Run Simulation", type="primary"):
        # Full Config Object
        total_y = y_ch4 + 0.8 # assuming balance is N2
        config = ReactorConfig(
            diameter=d_reac/100, bed_height=h_bed/100, particle_diameter=500e-6,
            catalyst_density=2000, particle_porosity=0.5, tortuosity=3.0, bed_porosity=0.4,
            catalyst_mass=0.05, inlet_temperature=t_in+273.15, inlet_pressure=p_in*1e5,
            flow_rate=flow/60/1e6, y_CH4_in=y_ch4/total_y, y_H2_in=0.0, y_N2_in=(0.8)/total_y,
            pre_exponential=1e6, activation_energy=100e3, beta=0.0, heat_of_reaction=74.87e6
        )
        
        reactor = MethaneDecompositionReactor(config, st.session_state['iso_check'])
        with st.spinner("Simulating physics..."):
            st.session_state.simulation_data = reactor.solve()
            st.session_state.config_data = config
        st.success("Simulation Complete")

# --- Main Layout ---
col_results, col_chat = st.columns([1.8, 1.2])

with col_results:
    st.subheader("📊 Simulation Results")
    if st.session_state.simulation_data:
        r = st.session_state.simulation_data
        
        m1, m2, m3 = st.columns(3)
        m1.metric("Conversion", f"{r['X_CH4'][-1]*100:.2f}%")
        m2.metric("H2 Yield", f"{r['V_dot_H2_Nm3_h'][-1]:.4f} Nm³/h")
        m3.metric("Outlet Temp", f"{r['T'][-1]-273.15:.1f} °C")
        
        tab1, tab2, tab3 = st.tabs(["Conversion", "Flow Rates", "Conditions"])
        
        with tab1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(r['z']*100, r['X_CH4']*100, color='#1a73e8', lw=2)
            ax.set_ylabel('CH4 Conversion (%)'); ax.set_xlabel('Bed Length (cm)'); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        with tab2:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(r['z']*100, r['F_CH4']*1000, label='CH4', color='red')
            ax.plot(r['z']*100, r['F_H2']*1000, label='H2', color='green')
            ax.set_ylabel('Flow (mmol/s)'); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        with tab3:
            fig, ax1 = plt.subplots(figsize=(6, 3))
            ax1.plot(r['z']*100, r['T']-273.15, color='orange', label='Temp')
            ax1.set_ylabel('Temp (°C)', color='orange')
            ax2 = ax1.twinx()
            ax2.plot(r['z']*100, r['P']/1e5, color='blue', label='Pressure')
            ax2.set_ylabel('Pressure (bar)', color='blue')
            st.pyplot(fig)
            
    else:
        st.info("👈 Configure parameters and click Run to start.")

# --- Chat Panel (With Action Buttons) ---
with col_chat:
    st.subheader("🤖 AI Assistant")
    
    # Action Buttons
    with st.container():
        c1, c2, c3 = st.columns(3)
        if c1.button("📊 Analyze", use_container_width=True):
            handle_ai_request("Analyze the simulation results. Point out limiting factors.")
        if c2.button("🔧 Optimize", use_container_width=True):
            handle_ai_request("Suggest optimizations for Temperature and Pressure to improve H2 yield.")
        if c3.button("📚 Explain", use_container_width=True):
            handle_ai_request("Explain the reaction kinetics and transport phenomena occurring here.")
            
    st.markdown("---")
    
    # Chat History
    chat_container = st.container(height=400)
    for msg in st.session_state.chat_history:
        chat_container.chat_message(msg["role"]).write(msg["content"])
        
    # Input
    if prompt := st.chat_input("Ask a question..."):
        handle_ai_request(prompt)
        st.rerun()

    # Save
    if st.session_state.chat_history:
        txt = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
        st.download_button("💾 Save Chat Log", txt, "chat_log.txt")
