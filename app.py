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
# PHYSICAL CONSTANTS & HELPER FUNCTIONS
# ============================================================================
R_GAS = 8.314
MW_CH4 = 16.04e-3
MW_H2 = 2.016e-3
MW_C = 12.01e-3
MW_N2 = 28.01e-3

def gas_viscosity(T, y_CH4, y_H2, y_N2):
    mu_CH4 = 1.02e-5 * (T / 300) ** 0.87
    mu_H2 = 8.76e-6 * (T / 300) ** 0.68
    mu_N2 = 1.78e-5 * (T / 300) ** 0.67
    return y_CH4 * mu_CH4 + y_H2 * mu_H2 + y_N2 * mu_N2

def gas_density(T, P, y_CH4, y_H2, y_N2):
    MW_mix = y_CH4 * MW_CH4 + y_H2 * MW_H2 + y_N2 * MW_N2
    return P * MW_mix / (R_GAS * T)

def diffusivity_CH4(T, P):
    return 1.87e-5 * (T / 300) ** 1.75 * (101325 / P)

def heat_capacity_mix(T, y_CH4, y_H2, y_N2):
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
# AI ASSISTANT LOGIC
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
            if "429" in str(e): return "⚠️ Quota Limit Exceeded. Please wait 30s."
            return f"Error: {e}"

# ============================================================================
# REACTOR LOGIC
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
        C_total_in = config.inlet_pressure / (R_GAS * config.inlet_temperature) / 1000
        self.F_total_in = config.flow_rate * C_total_in
        self.F_CH4_in = config.y_CH4_in * self.F_total_in
        self.F_H2_in = config.y_H2_in * self.F_total_in
        self.F_N2_in = config.y_N2_in * self.F_total_in
    
    def _ode_system(self, z, y):
        F_CH4, F_H2, T, P = y
        cfg = self.cfg
        
        # Clamps to prevent negative/zero crashes
        F_CH4 = max(F_CH4, 1e-30); F_H2 = max(F_H2, 0.0); T = max(T, 300.0); P = max(P, 1000.0)
        
        F_total = F_CH4 + F_H2 + self.F_N2_in
        y_CH4 = F_CH4 / F_total; y_H2 = F_H2 / F_total; y_N2 = self.F_N2_in / F_total
        
        rho = gas_density(T, P, y_CH4, y_H2, y_N2)
        mu = gas_viscosity(T, y_CH4, y_H2, y_N2)
        Q = F_total * 1000 * R_GAS * T / P
        u = Q / cfg.cross_section_area
        
        C_CH4 = F_CH4 / Q
        D_mol = diffusivity_CH4(T, P)
        D_eff = D_mol * cfg.particle_porosity / cfg.tortuosity
        k = arrhenius_rate_constant(T, cfg.pre_exponential, cfg.activation_energy, cfg.beta)
        
        phi = (cfg.particle_diameter / 6) * np.sqrt(k / D_eff) if D_eff > 0 else 0
        eta = effectiveness_factor(phi)
        r_bed = k * eta * C_CH4 * (1 - cfg.bed_porosity)
        
        A = cfg.cross_section_area
        dF_CH4_dz = -1.0 * r_bed * A
        dF_H2_dz = +2.0 * r_bed * A
        
        if self.isothermal:
            dT_dz = 0.0
        else:
            Cp_mix = heat_capacity_mix(T, y_CH4, y_H2, y_N2)
            F_total_mol = F_total * 1000
            Q_rxn = -cfg.heat_of_reaction * r_bed * A * 1000
            dT_dz = Q_rxn / (F_total_mol * Cp_mix + 1e-10)
            
        dP_dz = -ergun_pressure_drop(u, rho, mu, cfg.particle_diameter, cfg.bed_porosity)
        return np.array([dF_CH4_dz, dF_H2_dz, dT_dz, dP_dz])
        
    def solve(self, n_points=200):
        y0 = np.array([self.F_CH4_in, self.F_H2_in, self.cfg.inlet_temperature, self.cfg.inlet_pressure])
        solution = solve_ivp(
            self._ode_system, (0, self.cfg.bed_height), y0, 
            method='RK45', t_eval=np.linspace(0, self.cfg.bed_height, n_points),
            rtol=1e-8, atol=1e-12
        )
        z = solution.t
        F_CH4 = np.maximum(solution.y[0], 0)
        F_H2 = np.maximum(solution.y[1], 0)
        T = solution.y[2]
        P = solution.y[3]
        F_total = F_CH4 + F_H2 + self.F_N2_in
        
        return {
            'z': z, 'F_CH4': F_CH4, 'F_H2': F_H2, 'T': T, 'P': P,
            'y_CH4': F_CH4/F_total, 'y_H2': F_H2/F_total, 'y_N2': self.F_N2_in/F_total,
            'X_CH4': np.clip((self.F_CH4_in - F_CH4) / self.F_CH4_in, 0, 1),
            'm_dot_C_kg_s': (self.F_CH4_in - F_CH4) * MW_C * 1000,
            'm_dot_H2_kg_s': F_H2 * MW_H2 * 1000,
            'V_dot_H2_Nm3_h': F_H2 * 1000 * R_GAS * 273.15 / 101325 * 3600
        }

# ============================================================================
# MAIN APP & CALLBACKS
# ============================================================================

# --- Initialize State ---
if 'simulation_data' not in st.session_state: st.session_state.simulation_data = None
if 'chat_history' not in st.session_state: st.session_state.chat_history = []
if 'config_data' not in st.session_state: st.session_state.config_data = None

# --- CORE SIMULATION FUNCTION (Called by Button) ---
def run_simulation():
    """Reads current sidebar state and runs simulation immediately"""
    try:
        # 1. Gather inputs directly from session state
        total_y = st.session_state.y_ch4 + st.session_state.y_h2 + st.session_state.y_n2
        if total_y == 0: total_y = 1.0
        
        # 2. Build Config
        config = ReactorConfig(
            diameter=st.session_state.d_reac/100, 
            bed_height=st.session_state.h_bed/100, 
            particle_diameter=st.session_state.d_part*1e-6,
            catalyst_density=st.session_state.rho_cat, 
            particle_porosity=st.session_state.eps_part, 
            tortuosity=st.session_state.tau,
            bed_porosity=st.session_state.eps_bed, 
            catalyst_mass=st.session_state.mass_cat/1000,
            inlet_temperature=st.session_state.t_in+273.15, 
            inlet_pressure=st.session_state.p_in*1e5, 
            flow_rate=st.session_state.flow/60/1e6,
            y_CH4_in=st.session_state.y_ch4/total_y, 
            y_H2_in=st.session_state.y_h2/total_y, 
            y_N2_in=st.session_state.y_n2/total_y,
            pre_exponential=st.session_state.pre_exp, 
            activation_energy=st.session_state.act_e*1000, 
            beta=st.session_state.beta,
            heat_of_reaction=st.session_state.dh*1e6
        )
        
        # 3. Solve
        reactor = MethaneDecompositionReactor(config, isothermal=st.session_state.iso_check)
        st.session_state.simulation_data = reactor.solve()
        st.session_state.config_data = config
        
    except Exception as e:
        st.error(f"Simulation Failed: {e}")

# --- AI Helper ---
def handle_ai_request(prompt_text):
    st.session_state.chat_history.append({"role": "user", "content": prompt_text})
    
    context_str = "No simulation run yet."
    if st.session_state.simulation_data:
        r = st.session_state.simulation_data
        cfg = st.session_state.config_data
        context_str = (
            f"Inlet: T={cfg.inlet_temperature-273.15:.0f}C, P={cfg.inlet_pressure/1e5:.1f}bar. "
            f"Results: Conversion={r['X_CH4'][-1]*100:.2f}%, "
            f"H2 Yield={r['V_dot_H2_Nm3_h'][-1]:.4f} Nm3/h."
        )

    ai = GeminiAssistant(GEMINI_API_KEY)
    response = ai.generate_response(prompt_text, context_str)
    st.session_state.chat_history.append({"role": "assistant", "content": response})

# --- HEADER ---
st.image("https://raw.githubusercontent.com/anukaranAI/methane-reactor/main/AnukaranNew7.png", width=500)
st.markdown("### Methane Decomposition Reactor Simulator | Full Physics Engine")
st.markdown("---")

# --- SIDEBAR (With Keys) ---
with st.sidebar:
    st.header("⚙️ Input Parameters")
    
    st.markdown("#### 📐 Geometry")
    st.number_input("Reactor Diameter (cm)", 0.1, 500.0, 5.0, key="d_reac")
    st.number_input("Bed Height (cm)", 0.1, 1000.0, 20.0, key="h_bed")
    
    st.markdown("#### 🧪 Catalyst")
    st.number_input("Particle Diameter (μm)", 1.0, 10000.0, 500.0, key="d_part")
    st.number_input("Catalyst Density (kg/m³)", 100.0, 10000.0, 2000.0, key="rho_cat")
    st.number_input("Particle Porosity", 0.0, 1.0, 0.5, key="eps_part")
    st.number_input("Tortuosity", 1.0, 10.0, 3.0, key="tau")
    st.number_input("Bed Porosity", 0.0, 1.0, 0.4, key="eps_bed")
    st.number_input("Catalyst Mass (g)", 0.1, 100000.0, 50.0, key="mass_cat")
    
    st.markdown("#### 🌡️ Conditions")
    st.number_input("Inlet Temp (°C)", 25.0, 2000.0, 900.0, key="t_in")
    st.number_input("Inlet Pressure (bar)", 0.1, 200.0, 1.0, key="p_in")
    st.number_input("Flow Rate (mL/min)", 0.1, 100000.0, 100.0, key="flow")
    
    st.markdown("#### 🧬 Composition")
    st.number_input("CH₄ Mole Fraction", 0.0, 1.0, 0.20, key="y_ch4")
    st.number_input("H₂ Mole Fraction", 0.0, 1.0, 0.00, key="y_h2")
    st.number_input("N₂ Mole Fraction", 0.0, 1.0, 0.80, key="y_n2")
    
    st.markdown("#### ⚡ Kinetics")
    st.number_input("Pre-exp (A) [1/s]", 1.0, 1e15, 1.0e6, format="%e", key="pre_exp")
    st.number_input("Activation Energy (kJ/mol)", 1.0, 1000.0, 100.0, key="act_e")
    st.number_input("Temp Exponent (β)", -10.0, 10.0, 0.0, key="beta")
    st.number_input("Heat of Rxn (kJ/mol)", -1000.0, 1000.0, 74.87, key="dh")
    
    st.markdown("#### 🔧 Options")
    st.checkbox("Isothermal Simulation", value=True, key="iso_check")

    # THE FIX: USING ON_CLICK CALLBACK
    st.button("▶️ Run Simulation", type="primary", use_container_width=True, on_click=run_simulation)

# --- MAIN RESULTS ---
col_results, col_chat = st.columns([1.8, 1.2])

with col_results:
    st.subheader("📊 Simulation Results")
    
    if st.session_state.simulation_data:
        r = st.session_state.simulation_data
        cfg = st.session_state.config_data
        
        # Summary
        m1, m2, m3 = st.columns(3)
        m1.metric("Conversion", f"{r['X_CH4'][-1]*100:.2f} %")
        m2.metric("H2 Generation", f"{r['V_dot_H2_Nm3_h'][-1]:.4f} Nm³/h")
        m3.metric("Outlet Temp", f"{r['T'][-1]-273.15:.1f} °C")
        
        # Tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Conversion", "Flow", "Composition", "Temp", "Pressure", "Yield"])
        
        z_cm = r['z'] * 100
        
        def plot_graph(x, y, ylabel, title, color):
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(x, y, color=color, lw=2)
            ax.set_xlabel("Axial Position [cm]"); ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3); ax.set_title(title)
            st.pyplot(fig)
            plt.close(fig) # CLEANUP MEMORY

        with tab1: plot_graph(z_cm, r['X_CH4']*100, "CH4 Conversion [%]", "Conversion Profile", "blue")
        with tab2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(z_cm, r['F_CH4']*1000, 'r-', label='CH4')
            ax.plot(z_cm, r['F_H2']*1000, 'g-', label='H2')
            ax.set_ylabel("Flow [mmol/s]"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig); plt.close(fig)
        with tab3:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(z_cm, r['y_CH4'], 'r-', label='CH4')
            ax.plot(z_cm, r['y_H2'], 'g-', label='H2')
            ax.plot(z_cm, r['y_N2'], 'b--', label='N2')
            ax.set_ylabel("Mole Fraction"); ax.legend(); ax.grid(True, alpha=0.3)
            st.pyplot(fig); plt.close(fig)
        with tab4: plot_graph(z_cm, r['T']-273.15, "Temp [C]", "Temperature Profile", "orange")
        with tab5: plot_graph(z_cm, r['P']/1e5, "Pressure [bar]", "Pressure Profile", "purple")
        with tab6: plot_graph(z_cm, r['V_dot_H2_Nm3_h'], "H2 [Nm3/h]", "Cumulative Production", "green")
        
        # Download
        df = pd.DataFrame({'z': z_cm, 'X': r['X_CH4'], 'T': r['T']-273.15})
        st.download_button("💾 Download CSV", df.to_csv(index=False), "results.csv", "text/csv")
        
    else:
        st.info("👈 Configure parameters and click Run to start.")

# --- CHAT ---
with col_chat:
    st.subheader("🤖 AI Assistant")
    with st.container():
        c1, c2, c3 = st.columns(3)
        if c1.button("📊 Analyze", use_container_width=True): handle_ai_request("Analyze the results.")
        if c2.button("🔧 Optimize", use_container_width=True): handle_ai_request("How to optimize H2 yield?")
        if c3.button("📚 Explain", use_container_width=True): handle_ai_request("Explain the physics.")
            
    st.markdown("---")
    chat_box = st.container(height=500)
    for msg in st.session_state.chat_history:
        chat_box.chat_message(msg["role"]).write(msg["content"])
        
    if prompt := st.chat_input("Ask a question..."):
        handle_ai_request(prompt)
        st.rerun()
