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
# !!! SECURITY WARNING: Do not commit this key to GitHub !!!
GEMINI_API_KEY = "AIzaSyDTJGfbX8BrxLrNpk6OYQbAVb7_eiIB5Us" 

# ============================================================================
# PHYSICAL CONSTANTS & HELPER FUNCTIONS (From Desktop App)
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
# AI ASSISTANT LOGIC (Robust Auto-Detect + Action Buttons)
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
            # Auto-detect best available model
            all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
            available_names = [m.replace("models/", "") for m in all_models]
            
            preferred_order = ["gemini-1.5-flash", "gemini-flash-latest", "gemini-1.5-pro", "gemini-pro"]
            target_model = next((m for m in preferred_order if m in available_names), available_names[0] if available_names else None)
            
            if target_model:
                self.model = genai.GenerativeModel(target_model)
                self.model_available = True
            else:
                st.error("❌ No text generation models found for this API key.")
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
            if "429" in str(e): return "⚠️ Quota Limit Exceeded. Please wait 30 seconds."
            return f"Error: {e}"

# ============================================================================
# REACTOR LOGIC (Matches Desktop App)
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
        
        # Stability Clamps
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
        F_C = self.F_CH4_in - F_CH4
        
        return {
            'z': z, 'F_CH4': F_CH4, 'F_H2': F_H2, 'T': T, 'P': P,
            'y_CH4': F_CH4/F_total, 'y_H2': F_H2/F_total, 'y_N2': self.F_N2_in/F_total,
            'X_CH4': np.clip((self.F_CH4_in - F_CH4) / self.F_CH4_in, 0, 1),
            'm_dot_C_kg_s': F_C * MW_C * 1000,
            'm_dot_H2_kg_s': F_H2 * MW_H2 * 1000,
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

# --- Header (LOGO ADDED HERE) ---
st.image("https://raw.githubusercontent.com/anukaranAI/methane-reactor/main/AnukaranNew7.png", width=500)
st.markdown("### Methane Decomposition Reactor Simulator | Full Physics Engine")
st.markdown("---")

# --- Sidebar Inputs (MATCHING DESKTOP APP EXACTLY) ---
with st.sidebar:
    st.header("⚙️ Input Parameters")
    
    # 1. Reactor Geometry
    st.markdown("#### 📐 Reactor Geometry")
    d_reac = st.number_input("Reactor Diameter (cm)", 0.1, 500.0, 5.0)
    h_bed = st.number_input("Bed Height (cm)", 0.1, 1000.0, 20.0)
    # This was in the desktop GUI list but typically not used in calculation class, included for completeness
    h_total = st.number_input("Total Reactor Height (cm)", 0.1, 1000.0, 30.0)
    
    # 2. Catalyst Properties
    st.markdown("#### 🧪 Catalyst Properties")
    d_part = st.number_input("Particle Diameter (μm)", 1.0, 10000.0, 500.0)
    rho_cat = st.number_input("Catalyst Density (kg/m³)", 100.0, 10000.0, 2000.0)
    eps_part = st.number_input("Particle Porosity", 0.0, 1.0, 0.5)
    tau = st.number_input("Tortuosity", 1.0, 10.0, 3.0)
    eps_bed = st.number_input("Bed Porosity", 0.0, 1.0, 0.4)
    mass_cat = st.number_input("Catalyst Mass (g)", 0.1, 100000.0, 50.0)
    
    # 3. Operating Conditions
    st.markdown("#### 🌡️ Operating Conditions")
    t_in = st.number_input("Inlet Temperature (°C)", 25.0, 2000.0, 900.0)
    p_in = st.number_input("Inlet Pressure (bar)", 0.1, 200.0, 1.0)
    flow = st.number_input("Flow Rate (mL/min)", 0.1, 100000.0, 100.0)
    
    # 4. Inlet Composition
    st.markdown("#### 🧬 Inlet Composition")
    y_ch4 = st.number_input("CH₄ Mole Fraction", 0.0, 1.0, 0.20)
    y_h2 = st.number_input("H₂ Mole Fraction", 0.0, 1.0, 0.00)
    y_n2 = st.number_input("N₂ Mole Fraction", 0.0, 1.0, 0.80)
    
    # 5. Kinetic Parameters
    st.markdown("#### ⚡ Kinetic Parameters")
    pre_exp = st.number_input("Pre-exponential (A) [1/s]", 1.0, 1e15, 1.0e6, format="%e")
    act_e = st.number_input("Activation Energy (kJ/mol)", 1.0, 1000.0, 100.0)
    beta = st.number_input("Temperature Exponent (β)", -10.0, 10.0, 0.0)
    dh = st.number_input("Heat of Reaction (kJ/mol)", -1000.0, 1000.0, 74.87)
    
    # 6. Options
    st.markdown("#### 🔧 Options")
    st.session_state['iso_check'] = st.checkbox("Isothermal Simulation", value=True)

    # Run Button
    if st.button("▶️ Run Simulation", type="primary", use_container_width=True):
        # Normalize composition
        total = y_ch4 + y_h2 + y_n2
        if total == 0: total = 1.0
        
        config = ReactorConfig(
            diameter=d_reac/100, bed_height=h_bed/100, particle_diameter=d_part*1e-6,
            catalyst_density=rho_cat, particle_porosity=eps_part, tortuosity=tau,
            bed_porosity=eps_bed, catalyst_mass=mass_cat/1000,
            inlet_temperature=t_in+273.15, inlet_pressure=p_in*1e5, flow_rate=flow/60/1e6,
            y_CH4_in=y_ch4/total, y_H2_in=y_h2/total, y_N2_in=y_n2/total,
            pre_exponential=pre_exp, activation_energy=act_e*1000, beta=beta,
            heat_of_reaction=dh*1e6
        )
        
        reactor = MethaneDecompositionReactor(config, st.session_state['iso_check'])
        with st.spinner("Simulating..."):
            st.session_state.simulation_data = reactor.solve()
            st.session_state.config_data = config
        st.success("Simulation Complete")

# --- Main Layout ---
col_results, col_chat = st.columns([1.8, 1.2])

with col_results:
    st.subheader("📊 Simulation Results")
    if st.session_state.simulation_data:
        r = st.session_state.simulation_data
        cfg = st.session_state.config_data
        
        # Summary Display (Matches Desktop Metrics)
        m1, m2, m3 = st.columns(3)
        m1.metric("CH₄ Conversion", f"{r['X_CH4'][-1]*100:.2f} %")
        m2.metric("H₂ Generation", f"{r['V_dot_H2_Nm3_h'][-1]:.4f} Nm³/h")
        m3.metric("H₂ Mass Flow", f"{r['m_dot_H2_kg_s'][-1]*3600:.4f} kg/h")
        
        m4, m5, m6 = st.columns(3)
        m4.metric("Carbon Production", f"{r['m_dot_C_kg_s'][-1]*3600:.4f} kg/h")
        m5.metric("Outlet Temperature", f"{r['T'][-1]-273.15:.1f} °C")
        m6.metric("Pressure Drop", f"{(cfg.inlet_pressure - r['P'][-1])/1000:.2f} kPa")
        
        # Tabs (Matches Desktop Tabs)
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Conversion", "Flow Rates", "Composition", "Temperature", "Pressure", "H₂ Production"
        ])
        
        z_cm = r['z'] * 100
        
        with tab1:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(z_cm, r['X_CH4']*100, color='blue', lw=2)
            ax.set_ylabel('CH₄ Conversion [%]'); ax.set_xlabel('Axial Position [cm]'); ax.grid(True, alpha=0.3)
            ax.set_title("Methane Conversion Profile")
            st.pyplot(fig)
            
        with tab2:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(z_cm, r['F_CH4']*1000, label='CH₄', color='red', lw=2)
            ax.plot(z_cm, r['F_H2']*1000, label='H₂', color='green', lw=2)
            ax.set_ylabel('Molar Flow Rate [mmol/s]'); ax.set_xlabel('Axial Position [cm]'); 
            ax.legend(); ax.grid(True, alpha=0.3); ax.set_title("Species Flow Rates")
            st.pyplot(fig)
            
        with tab3:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(z_cm, r['y_CH4']*100, label='CH₄', linestyle='-', color='red', lw=2)
            ax.plot(z_cm, r['y_H2']*100, label='H₂', linestyle='-', color='green', lw=2)
            ax.plot(z_cm, r['y_N2']*100, label='N₂', linestyle='--', color='blue', lw=2)
            ax.set_ylabel('Mole Fraction [%]'); ax.set_xlabel('Axial Position [cm]'); 
            ax.legend(); ax.grid(True, alpha=0.3); ax.set_title("Gas Composition")
            st.pyplot(fig)
            
        with tab4:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(z_cm, r['T']-273.15, color='red', lw=2)
            ax.set_ylabel('Temperature [°C]'); ax.set_xlabel('Axial Position [cm]'); 
            ax.grid(True, alpha=0.3); ax.set_title("Temperature Profile")
            st.pyplot(fig)

        with tab5:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(z_cm, r['P']/1e5, color='blue', lw=2)
            ax.set_ylabel('Pressure [bar]'); ax.set_xlabel('Axial Position [cm]'); 
            ax.grid(True, alpha=0.3); ax.set_title("Pressure Profile")
            st.pyplot(fig)

        with tab6:
            fig, ax = plt.subplots(figsize=(6, 3.5))
            ax.plot(z_cm, r['V_dot_H2_Nm3_h'], color='green', lw=2)
            ax.set_ylabel('H₂ Generation [Nm³/h]'); ax.set_xlabel('Axial Position [cm]'); 
            ax.grid(True, alpha=0.3); ax.set_title("Cumulative H₂ Production")
            st.pyplot(fig)
            
        # Download Button
        df_export = pd.DataFrame({
            'z_cm': z_cm, 'X_CH4': r['X_CH4'], 'T_C': r['T']-273.15, 'P_bar': r['P']/1e5, 
            'H2_Nm3_h': r['V_dot_H2_Nm3_h']
        })
        st.download_button("💾 Download CSV Results", df_export.to_csv(index=False), "results.csv", "text/csv")

    else:
        st.info("👈 Set parameters in the sidebar and click 'Run Simulation'.")

# --- Chat Panel (With Action Buttons) ---
with col_chat:
    st.subheader("🤖 AI Assistant")
    
    # Action Buttons (New Feature)
    with st.container():
        c1, c2, c3 = st.columns(3)
        if c1.button("📊 Analyze", use_container_width=True):
            handle_ai_request("Analyze the current simulation results. What are the limiting factors?")
        if c2.button("🔧 Optimize", use_container_width=True):
            handle_ai_request("Based on these results, how can I optimize Inlet Temp, Pressure, or Flow Rate to maximize H2 production?")
        if c3.button("📚 Explain", use_container_width=True):
            handle_ai_request("Explain the physical and chemical phenomena happening in this reactor in simple terms.")
            
    st.markdown("---")
    
    # Chat History
    chat_container = st.container(height=500)
    for msg in st.session_state.chat_history:
        chat_container.chat_message(msg["role"]).write(msg["content"])
        
    # Input
    if prompt := st.chat_input("Ask a question about your simulation..."):
        handle_ai_request(prompt)
        st.rerun()

    # Save
    if st.session_state.chat_history:
        txt = "\n".join([f"{m['role']}: {m['content']}" for m in st.session_state.chat_history])
        st.download_button("💾 Save Chat Log", txt, "chat_log.txt")
