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

# !!! SECURITY WARNING !!!
# On EC2, it is better to set this as an Environment Variable.
# For now, paste your key here.
GEMINI_API_KEY = "AIzaSyDTJGfbX8BrxLrNpk6OYQbAVb7_eiIB5Us" 

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================
R_GAS = 8.314
MW_CH4 = 16.04e-3
MW_H2 = 2.016e-3
MW_C = 12.01e-3
MW_N2 = 28.01e-3

# ============================================================================
# AI ASSISTANT LOGIC (Stable Version)
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
            
            # Simple fallback logic for Streamlit speed
            # We prefer the 1.5 Flash model for speed/cost
            target_model = "gemini-1.5-flash"
            
            self.model = genai.GenerativeModel(target_model)
            self.model_available = True
            
        except Exception as e:
            st.error(f"❌ AI Config Error: {e}")
            self.model_available = False

    def generate_response(self, user_message, context_str):
        if not self.model_available:
            return "AI is not available. Check API Key."

        prompt = (
            "You are an expert chemical reaction engineering assistant.\n"
            f"Simulation context: {context_str}\n\n"
            f"User question: {user_message}"
        )

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            if "429" in str(e):
                return "⚠️ Quota Limit Exceeded. Please wait a moment."
            return f"Error: {e}"

# ============================================================================
# REACTOR LOGIC (Original Classes)
# ============================================================================
@dataclass
class ReactorConfig:
    diameter: float
    bed_height: float
    particle_diameter: float
    catalyst_density: float
    particle_porosity: float
    tortuosity: float
    bed_porosity: float
    catalyst_mass: float
    inlet_temperature: float
    inlet_pressure: float
    flow_rate: float
    y_CH4_in: float
    y_H2_in: float
    y_N2_in: float
    pre_exponential: float
    activation_energy: float
    beta: float
    heat_of_reaction: float
    
    @property
    def cross_section_area(self) -> float:
        return np.pi * (self.diameter / 2) ** 2

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

class MethaneDecompositionReactor:
    def __init__(self, config: ReactorConfig, isothermal: bool = True):
        self.cfg = config
        self.isothermal = isothermal
        C_total_in = config.inlet_pressure / (R_GAS * config.inlet_temperature)
        self.C_total_in = C_total_in / 1000
        self.F_total_in = config.flow_rate * self.C_total_in
        self.F_CH4_in = config.y_CH4_in * self.F_total_in
        self.F_H2_in = config.y_H2_in * self.F_total_in
        self.F_N2_in = config.y_N2_in * self.F_total_in
        self.u_in = config.flow_rate / config.cross_section_area
    
    def _ode_system(self, z, y):
        F_CH4, F_H2, T, P = y
        cfg = self.cfg
        F_CH4 = max(F_CH4, 1e-30)
        F_H2 = max(F_H2, 0.0)
        T = max(T, 300.0)
        P = max(P, 1000.0)
        F_total = F_CH4 + F_H2 + self.F_N2_in
        y_CH4 = F_CH4 / F_total
        y_H2 = F_H2 / F_total
        y_N2 = self.F_N2_in / F_total
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
        dT_dz = 0.0 if self.isothermal else (-cfg.heat_of_reaction * r_bed * A * 1000) / ((F_total * 1000) * heat_capacity_mix(T, y_CH4, y_H2, y_N2) + 1e-10)
        dP_dz = -ergun_pressure_drop(u, rho, mu, cfg.particle_diameter, cfg.bed_porosity)
        return np.array([dF_CH4_dz, dF_H2_dz, dT_dz, dP_dz])
    
    def solve(self, n_points=200):
        y0 = np.array([self.F_CH4_in, self.F_H2_in, self.cfg.inlet_temperature, self.cfg.inlet_pressure])
        solution = solve_ivp(self._ode_system, (0, self.cfg.bed_height), y0, method='RK45', t_eval=np.linspace(0, self.cfg.bed_height, n_points), rtol=1e-8, atol=1e-12)
        z = solution.t
        F_CH4, F_H2, T, P = np.maximum(solution.y[0], 0), np.maximum(solution.y[1], 0), solution.y[2], solution.y[3]
        F_total = F_CH4 + F_H2 + self.F_N2_in
        V_dot_H2 = F_H2 * 1000 * R_GAS * 273.15 / 101325 * 3600
        return {'z': z, 'F_CH4': F_CH4, 'F_H2': F_H2, 'T': T, 'P': P, 'V_dot_H2_Nm3_h': V_dot_H2, 'X_CH4': np.clip((self.F_CH4_in - F_CH4) / self.F_CH4_in, 0, 1), 'y_CH4': F_CH4/F_total, 'y_H2': F_H2/F_total, 'y_N2': self.F_N2_in/F_total, 'm_dot_C_kg_s': (self.F_CH4_in - F_CH4) * MW_C * 1000, 'm_dot_H2_kg_s': F_H2 * MW_H2 * 1000}

# ============================================================================
# UI LAYOUT
# ============================================================================

# --- Initialize Session State ---
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'config_data' not in st.session_state:
    st.session_state.config_data = None

# --- Header ---
st.title("🔬 ANUKARAN AI")
st.markdown("### Methane Decomposition Reactor Simulator | CH₄(g) → C(s) + 2H₂(g)")
st.markdown("---")

# --- Sidebar Inputs ---
with st.sidebar:
    st.header("⚙️ Input Parameters")
    
    st.subheader("📐 Geometry")
    d_reac = st.number_input("Reactor Diameter (cm)", 1.0, 100.0, 5.0)
    h_bed = st.number_input("Bed Height (cm)", 1.0, 500.0, 20.0)
    
    st.subheader("🧪 Catalyst")
    d_part = st.number_input("Particle Diameter (μm)", 10.0, 5000.0, 500.0)
    rho_cat = st.number_input("Density (kg/m³)", 500.0, 5000.0, 2000.0)
    eps_part = st.number_input("Particle Porosity", 0.0, 1.0, 0.5)
    tau = st.number_input("Tortuosity", 1.0, 10.0, 3.0)
    eps_bed = st.number_input("Bed Porosity", 0.1, 0.9, 0.4)
    mass_cat = st.number_input("Catalyst Mass (g)", 1.0, 10000.0, 50.0)
    
    st.subheader("🌡️ Conditions")
    t_in = st.number_input("Inlet Temp (°C)", 300.0, 1500.0, 900.0)
    p_in = st.number_input("Inlet Pressure (bar)", 0.1, 100.0, 1.0)
    flow = st.number_input("Flow Rate (mL/min)", 10.0, 10000.0, 100.0)
    
    st.subheader("🧬 Composition")
    y_ch4 = st.number_input("CH4 Mole Fraction", 0.0, 1.0, 0.2)
    y_h2 = st.number_input("H2 Mole Fraction", 0.0, 1.0, 0.0)
    y_n2 = st.number_input("N2 Mole Fraction", 0.0, 1.0, 0.8)
    
    st.subheader("⚡ Kinetics")
    pre_exp = st.number_input("Pre-exp A (1/s)", 1.0, 1e10, 1.0e6, format="%e")
    act_e = st.number_input("Activation Energy (kJ/mol)", 10.0, 500.0, 100.0)
    beta = st.number_input("Beta", -5.0, 5.0, 0.0)
    dh = st.number_input("Heat of Rxn (kJ/mol)", 0.0, 200.0, 74.87)
    
    st.subheader("🔧 Options")
    isothermal = st.checkbox("Isothermal Mode", value=True)
    
    if st.button("▶️ Run Simulation", type="primary"):
        # Create Config
        total = y_ch4 + y_h2 + y_n2
        config = ReactorConfig(
            diameter=d_reac/100, bed_height=h_bed/100, particle_diameter=d_part*1e-6,
            catalyst_density=rho_cat, particle_porosity=eps_part, tortuosity=tau,
            bed_porosity=eps_bed, catalyst_mass=mass_cat/1000,
            inlet_temperature=t_in+273.15, inlet_pressure=p_in*1e5, flow_rate=flow/60/1e6,
            y_CH4_in=y_ch4/total, y_H2_in=y_h2/total, y_N2_in=y_n2/total,
            pre_exponential=pre_exp, activation_energy=act_e*1000, beta=beta,
            heat_of_reaction=dh*1e6
        )
        
        # Run Solve
        reactor = MethaneDecompositionReactor(config, isothermal)
        with st.spinner("Solving ODEs..."):
            results = reactor.solve()
        
        st.session_state.simulation_data = results
        st.session_state.config_data = config
        st.success("Simulation Complete!")

# --- Main Layout ---
col_results, col_chat = st.columns([2, 1])

with col_results:
    st.subheader("📊 Simulation Results")
    
    if st.session_state.simulation_data:
        r = st.session_state.simulation_data
        
        # Metrics Row
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Conversion", f"{r['X_CH4'][-1]*100:.2f}%")
        m2.metric("H2 Production", f"{r['V_dot_H2_Nm3_h'][-1]:.4f} Nm³/h")
        m3.metric("Outlet Temp", f"{r['T'][-1]-273.15:.1f} °C")
        m4.metric("Pressure Drop", f"{(p_in - r['P'][-1]/1e5)*1000:.2f} mbar")
        
        # Plots
        tab1, tab2, tab3, tab4 = st.tabs(["Conversion", "Flow Rates", "Composition", "Conditions"])
        
        z_cm = r['z'] * 100
        
        with tab1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(z_cm, r['X_CH4']*100, color='#1a73e8', linewidth=2)
            ax.set_ylabel('CH4 Conversion (%)')
            ax.set_xlabel('Bed Length (cm)')
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        with tab2:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(z_cm, r['F_CH4']*1000, label='CH4', color='red')
            ax.plot(z_cm, r['F_H2']*1000, label='H2', color='green')
            ax.set_ylabel('Molar Flow (mmol/s)')
            ax.set_xlabel('Bed Length (cm)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        with tab3:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(z_cm, r['y_CH4']*100, label='CH4', linestyle='--')
            ax.plot(z_cm, r['y_H2']*100, label='H2')
            ax.plot(z_cm, r['y_N2']*100, label='N2')
            ax.set_ylabel('Mole Fraction (%)')
            ax.set_xlabel('Bed Length (cm)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        with tab4:
            fig, ax1 = plt.subplots(figsize=(6, 3))
            ax1.plot(z_cm, r['T']-273.15, color='orange', label='Temp')
            ax1.set_ylabel('Temp (°C)', color='orange')
            ax2 = ax1.twinx()
            ax2.plot(z_cm, r['P']/1e5, color='blue', label='Pressure')
            ax2.set_ylabel('Pressure (bar)', color='blue')
            st.pyplot(fig)
            
        # Download Data
        df_export = pd.DataFrame({
            'z_cm': z_cm, 'X_CH4': r['X_CH4'], 'T_C': r['T']-273.15, 'P_bar': r['P']/1e5
        })
        st.download_button("💾 Download CSV", df_export.to_csv(index=False), "results.csv", "text/csv")
        
    else:
        st.info("👈 Set parameters and click 'Run Simulation' to see results.")

# --- Chat Interface ---
with col_chat:
    st.subheader("🤖 AI Assistant")
    
    # Init AI
    ai = GeminiAssistant(GEMINI_API_KEY)
    
    # Display Chat History
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
    # Chat Input
    if prompt := st.chat_input("Ask about the simulation..."):
        # Display user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Context String
        context_str = "No simulation run yet."
        if st.session_state.simulation_data:
            r = st.session_state.simulation_data
            cfg = st.session_state.config_data
            context_str = (
                f"Inlet T={cfg.inlet_temperature-273.15:.0f}C, P={cfg.inlet_pressure/1e5:.1f}bar. "
                f"Result: CH4 Conversion={r['X_CH4'][-1]*100:.2f}%, "
                f"H2 Produced={r['V_dot_H2_Nm3_h'][-1]:.4f} Nm3/h."
            )
            
        # Generate Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = ai.generate_response(prompt, context_str)
                st.markdown(response)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
