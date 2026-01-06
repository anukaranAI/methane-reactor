"""
╔══════════════════════════════════════════════════════════════════╗
║                         ANUKARAN AI                              ║
║          Methane Decomposition Reactor Simulator                 ║
║                                                                  ║
║   Lab Scale Calibration → Industrial Scale-Up Optimization       ║
║              With AI Assistant & Transient Analysis              ║
║                                                                  ║
║   Reaction: CH₄ → C + 2H₂                                       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- LOCAL IMPORTS ---
from ai_assistant import GeminiAssistant

from core.experimental_data import (
    EXPERIMENTAL_DATA, TOS_MINUTES,
    get_experimental_data, get_available_temperatures,
)
from core.transient_model import (
    ReactorConfig, SteadyStateReactor, TransientReactor,
    DeactivationModel, DeactivationParams,
)
from core.scaleup import (
    IndustrialScaleUp, ScaleUpResult, get_assumptions_text,
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Anukaran AI - Reactor Scale-Up",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# DEFAULT VALUES
# ============================================================================
DEFAULTS = {
    # Lab Scale
    'lab_flow_rate': 75.0,          # mL/min
    'lab_diameter': 1.8,            # cm
    'lab_bed_height': 2.59,         # cm
    'lab_catalyst_mass': 2.2,       # g
    'lab_particle_size': 10.0,      # μm - fine powder for lab
    'lab_particle_density': 2821.56, # kg/m³
    
    # Industrial Scale
    'ind_flow_rate': 1960.0,        # LPM
    'ind_catalyst_mass': 57.0,      # kg
    'ind_particle_size': 3000.0,    # μm = 3mm - Industrial uses larger particles!
    'ind_particle_density': 2821.56, # kg/m³
    'ind_LD_min': 1.0,
    'ind_LD_max': 3.0,
    'ind_target_H2': 30.0,          # %
    'ind_max_dP': 50.0,             # kPa
    
    # Assumed Parameters
    'bed_porosity': 0.4,
    'particle_porosity': 0.5,
    'tortuosity': 3.0,
    'inlet_pressure': 1.0,          # bar
    'temperature': 800.0,           # °C
    'inlet_CH4': 100.0,             # %
    
    # Kinetics (Initial guesses)
    'pre_exponential': 5.0e4,       # 1/s
    'activation_energy': 150.0,     # kJ/mol
    'deactivation_kd': 0.008,       # 1/min
}

# ============================================================================
# SESSION STATE
# ============================================================================
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

other_defaults = {
    'calibration_done': False,
    'lab_results': None,
    'lab_transient_results': None,
    'industrial_results': None,
    'industrial_transient_results': None,
    'chat_history': [],
    'auto_insight': '',
    'calibrated_A_value': 5.0e4,
    'calibrated_Ea_value': 150.0,
    'calibrated_kd_value': 0.008,
}

for key, value in other_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# ============================================================================
# API KEY
# ============================================================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = ""

# ============================================================================
# HELPER: Calculate Carbon Deposition
# ============================================================================

def calculate_carbon_rate(conversion, flow_rate_mol_s, catalyst_mass_kg):
    """
    Calculate carbon deposition rate.
    CH₄ → C + 2H₂
    1 mol CH₄ produces 1 mol C (12 g/mol)
    Returns: carbon deposition rate in g/h per kg catalyst
    """
    mol_CH4_converted = conversion * flow_rate_mol_s
    carbon_rate_g_s = mol_CH4_converted * 12.0
    carbon_rate_g_h_kg = (carbon_rate_g_s * 3600) / catalyst_mass_kg
    return carbon_rate_g_h_kg


def calculate_cumulative_carbon(time_min, conversion_array, flow_rate_mol_s, catalyst_mass_kg):
    """Calculate cumulative carbon deposited over time"""
    mol_CH4_converted = conversion_array * flow_rate_mol_s
    carbon_mol_s = mol_CH4_converted
    dt_s = (time_min[1] - time_min[0]) * 60 if len(time_min) > 1 else 60
    cumulative_carbon_mol = np.cumsum(carbon_mol_s) * dt_s
    cumulative_carbon_g = cumulative_carbon_mol * 12.0
    cumulative_carbon_g_per_kg = cumulative_carbon_g / catalyst_mass_kg
    return cumulative_carbon_g_per_kg


# ============================================================================
# HELPER: Create Reactor Configs
# ============================================================================

def create_lab_reactor_config(A: float = None, Ea: float = None):
    """Create ReactorConfig for lab scale"""
    if A is None:
        A = st.session_state.get('calibrated_A_value', st.session_state.pre_exponential)
    if Ea is None:
        Ea = st.session_state.get('calibrated_Ea_value', st.session_state.activation_energy) * 1000
    
    return ReactorConfig(
        diameter_m=st.session_state.lab_diameter / 100,
        bed_height_m=st.session_state.lab_bed_height / 100,
        particle_diameter_m=st.session_state.lab_particle_size * 1e-6,
        particle_density=st.session_state.lab_particle_density,
        particle_porosity=st.session_state.particle_porosity,
        tortuosity=st.session_state.tortuosity,
        bed_porosity=st.session_state.bed_porosity,
        catalyst_mass_kg=st.session_state.lab_catalyst_mass / 1000,
        temperature_K=st.session_state.temperature + 273.15,
        inlet_pressure_Pa=st.session_state.inlet_pressure * 1e5,
        flow_rate_m3_s=st.session_state.lab_flow_rate / 60 / 1e6,
        y_CH4_in=st.session_state.inlet_CH4 / 100,
        y_H2_in=0.0,
        pre_exponential=A,
        activation_energy=Ea,
    )


def create_industrial_reactor_config(diameter_m: float, height_m: float, A: float = None, Ea: float = None):
    """
    Create ReactorConfig for industrial scale.
    
    Note: Industrial reactors use larger particles (1-5 mm) compared to lab (10 μm)
    to maintain reasonable pressure drop at high flow rates.
    """
    if A is None:
        A = st.session_state.get('calibrated_A_value', st.session_state.pre_exponential)
    if Ea is None:
        Ea = st.session_state.get('calibrated_Ea_value', st.session_state.activation_energy) * 1000
    
    # Convert units properly
    particle_diameter_m = st.session_state.ind_particle_size * 1e-6  # μm to m
    flow_rate_m3_s = st.session_state.ind_flow_rate / 60 / 1000      # LPM to m³/s
    
    return ReactorConfig(
        diameter_m=diameter_m,
        bed_height_m=height_m,
        particle_diameter_m=particle_diameter_m,
        particle_density=st.session_state.ind_particle_density,
        particle_porosity=st.session_state.particle_porosity,
        tortuosity=st.session_state.tortuosity,
        bed_porosity=st.session_state.bed_porosity,
        catalyst_mass_kg=st.session_state.ind_catalyst_mass,
        temperature_K=st.session_state.temperature + 273.15,
        inlet_pressure_Pa=st.session_state.inlet_pressure * 1e5,
        flow_rate_m3_s=flow_rate_m3_s,
        y_CH4_in=st.session_state.inlet_CH4 / 100,
        y_H2_in=0.0,
        pre_exponential=A,
        activation_energy=Ea,
    )


def get_molar_flow_rate(flow_rate_mL_min, pressure_Pa, temperature_K):
    """Calculate molar flow rate from volumetric flow"""
    R = 8.314
    flow_rate_m3_s = flow_rate_mL_min / 60 / 1e6
    C_total = pressure_Pa / (R * temperature_K)
    return flow_rate_m3_s * C_total


# ============================================================================
# AI FUNCTIONS
# ============================================================================

def get_ai_response(prompt: str, context: str = "") -> str:
    if not GEMINI_API_KEY:
        return "⚠️ AI not configured. Add GEMINI_API_KEY to secrets."
    try:
        ai = GeminiAssistant(GEMINI_API_KEY)
        full_prompt = f"""You are a chemical engineer AI for methane decomposition reactor (CH₄ → C + 2H₂).
Context: {context}
Question: {prompt}
Give concise, helpful response with bullet points."""
        return ai.generate_response(full_prompt, context)
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


def generate_auto_insight(action: str, results: dict = None) -> str:
    if action == "experimental_data":
        exp_data = get_experimental_data(800)
        return f"""**📊 Experimental Data @ {st.session_state.temperature}°C**
- Initial H₂: **{exp_data['H2_percent'][0]:.2f}%** | Final: **{exp_data['H2_percent'][-1]:.2f}%**
- CH₄ increases from {exp_data['CH4_percent'][0]:.1f}% to {exp_data['CH4_percent'][-1]:.1f}% (less conversion)
- 💡 Carbon deposition causes catalyst deactivation"""

    elif action == "lab_calibration" and results:
        return f"""**🔬 Lab Calibration @ {st.session_state.temperature}°C**
- Model H₂: **{results.get('H2_t0', 0):.2f}%** vs Exp: **{results.get('exp_H2_t0', 24.13):.2f}%**
- RMSE: **{results.get('rmse', 0):.2f}%**
- 💡 {'Good fit!' if results.get('rmse', 100) < 5 else 'Adjust A or Ea'}"""

    elif action == "transient" and results:
        return f"""**⏱️ Transient Analysis**
- Initial H₂: **{results.get('H2_initial', 0):.2f}%**
- Carbon deposited: **{results.get('carbon_total', 0):.1f} g/kg catalyst**
- Time to 50% activity: **{results.get('half_life', 0):.0f} min**"""

    elif action == "industrial" and results:
        return f"""**🏭 Industrial Optimization**
- Optimal: D={results.get('diameter', 0):.1f}cm, L={results.get('height', 0):.1f}cm
- H₂: **{results.get('H2', 0):.2f}%** | ΔP: {results.get('dP', 0):.1f} kPa
- Particle: {st.session_state.ind_particle_size} μm ({st.session_state.ind_particle_size/1000:.1f} mm)
- 💡 {'✅ Target met!' if results.get('H2', 0) >= st.session_state.ind_target_H2 else '⚠️ Below target'}"""

    return "👋 Welcome! Run simulations to get insights."


def get_context_string() -> str:
    parts = [
        f"Reaction: CH₄ → C + 2H₂",
        f"Temperature: {st.session_state.temperature}°C, Target H₂: {st.session_state.ind_target_H2}%",
        f"Lab: D={st.session_state.lab_diameter}cm, L={st.session_state.lab_bed_height}cm, Flow={st.session_state.lab_flow_rate}mL/min, Particle={st.session_state.lab_particle_size}μm",
        f"Industrial: Flow={st.session_state.ind_flow_rate}LPM, Particle={st.session_state.ind_particle_size}μm ({st.session_state.ind_particle_size/1000:.1f}mm)",
    ]
    if st.session_state.calibration_done:
        parts.append(f"Calibrated: A={st.session_state.calibrated_A_value:.2e}, Ea={st.session_state.calibrated_Ea_value:.0f}kJ/mol")
    return "\n".join(parts)


def handle_chat(user_input: str):
    if user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        response = get_ai_response(user_input, get_context_string())
        st.session_state.chat_history.append({"role": "assistant", "content": response})


# ============================================================================
# PLOTTING FUNCTIONS
# ============================================================================

def plot_experiment_data(exp_data, target_H2):
    """Plot experimental data: H2, CH4 vs time"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(exp_data['TOS_min'], exp_data['H2_percent'], 'bo-', markersize=8, linewidth=2, label='H₂ (Experimental)')
    axes[0].axhline(y=target_H2, color='green', linestyle='--', linewidth=2, label=f'Target ({target_H2}%)')
    axes[0].fill_between(exp_data['TOS_min'], exp_data['H2_percent'], alpha=0.3)
    axes[0].set_xlabel('Time on Stream [min]')
    axes[0].set_ylabel('H₂ Composition [%]')
    axes[0].set_title('H₂ Production')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, max(45, exp_data['H2_percent'].max() + 5)])
    
    axes[1].plot(exp_data['TOS_min'], exp_data['CH4_percent'], 'rs-', markersize=8, linewidth=2, label='CH₄ (Experimental)')
    axes[1].fill_between(exp_data['TOS_min'], exp_data['CH4_percent'], alpha=0.3, color='red')
    axes[1].set_xlabel('Time on Stream [min]')
    axes[1].set_ylabel('CH₄ Composition [%]')
    axes[1].set_title('CH₄ at Outlet (Unreacted)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_transient_results(results, target_H2, catalyst_mass_kg, flow_mol_s, title_prefix=""):
    """Plot transient results: H2, CH4, Activity, Carbon"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    time = results['time_min']
    H2 = results['H2_percent']
    CH4 = results['CH4_percent']
    activity = results['activity']
    conversion = results['conversion'] / 100
    
    cumulative_carbon = calculate_cumulative_carbon(time, conversion, flow_mol_s, catalyst_mass_kg)
    
    # H2 vs Time
    axes[0, 0].plot(time, H2, 'b-', linewidth=2, label='H₂')
    axes[0, 0].axhline(y=target_H2, color='green', linestyle='--', linewidth=2, label=f'Target ({target_H2}%)')
    axes[0, 0].fill_between(time, target_H2, H2, where=H2>=target_H2, alpha=0.3, color='green')
    axes[0, 0].fill_between(time, H2, target_H2, where=H2<target_H2, alpha=0.3, color='red')
    axes[0, 0].set_xlabel('Time [min]')
    axes[0, 0].set_ylabel('H₂ [%]')
    axes[0, 0].set_title(f'{title_prefix}H₂ Production')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # CH4 vs Time
    axes[0, 1].plot(time, CH4, 'r-', linewidth=2, label='CH₄')
    axes[0, 1].set_xlabel('Time [min]')
    axes[0, 1].set_ylabel('CH₄ [%]')
    axes[0, 1].set_title(f'{title_prefix}CH₄ at Outlet')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Activity vs Time
    axes[1, 0].plot(time, activity, 'g-', linewidth=2, label='Activity')
    axes[1, 0].axhline(y=0.5, color='orange', linestyle='--', label='50% Activity')
    axes[1, 0].set_xlabel('Time [min]')
    axes[1, 0].set_ylabel('Catalyst Activity [-]')
    axes[1, 0].set_title(f'{title_prefix}Catalyst Deactivation')
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Carbon Deposition vs Time
    axes[1, 1].plot(time, cumulative_carbon, 'm-', linewidth=2, label='Cumulative Carbon')
    axes[1, 1].set_xlabel('Time [min]')
    axes[1, 1].set_ylabel('Carbon [g/kg catalyst]')
    axes[1, 1].set_title(f'{title_prefix}Carbon Deposition')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, cumulative_carbon[-1] if len(cumulative_carbon) > 0 else 0


def plot_calibration_comparison(results, exp_data, target_H2):
    """Plot calibration: Model vs Experimental for H2 and CH4"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(results['time_min'], results['H2_percent'], 'b-', linewidth=2, label='Model')
    axes[0].scatter(exp_data['TOS_min'], exp_data['H2_percent'], c='red', s=100, zorder=5, label='Experimental')
    axes[0].axhline(y=target_H2, color='green', linestyle='--', linewidth=2, label=f'Target ({target_H2}%)')
    axes[0].set_xlabel('Time on Stream [min]')
    axes[0].set_ylabel('H₂ [%]')
    axes[0].set_title('H₂ Composition: Model vs Experimental')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, max(50, results['H2_percent'].max() + 5)])
    
    axes[1].plot(results['time_min'], results['CH4_percent'], 'b-', linewidth=2, label='Model')
    axes[1].scatter(exp_data['TOS_min'], exp_data['CH4_percent'], c='red', s=100, zorder=5, label='Experimental')
    axes[1].set_xlabel('Time on Stream [min]')
    axes[1].set_ylabel('CH₄ [%]')
    axes[1].set_title('CH₄ Composition: Model vs Experimental')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_parametric_study(parametric, optimal, target_H2, max_dP):
    """Plot parametric study: H2, Conversion, Pressure Drop vs L/D"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    LD = [p['LD_ratio'] for p in parametric]
    H2 = [p['H2_percent'] for p in parametric]
    conv = [p['conversion_percent'] for p in parametric]
    dP = [p['pressure_drop_kPa'] for p in parametric]
    
    axes[0].plot(LD, H2, 'b-o', markersize=6, linewidth=2)
    axes[0].axhline(y=target_H2, color='green', linestyle='--', linewidth=2, label=f'Target ({target_H2}%)')
    axes[0].axvline(x=optimal.optimal_LD_ratio, color='red', linestyle=':', linewidth=2, label=f'Optimal L/D={optimal.optimal_LD_ratio:.2f}')
    axes[0].set_xlabel('L/D Ratio')
    axes[0].set_ylabel('H₂ [%]')
    axes[0].set_title('H₂ vs L/D')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(LD, conv, 'g-o', markersize=6, linewidth=2)
    axes[1].axvline(x=optimal.optimal_LD_ratio, color='red', linestyle=':', linewidth=2, label='Optimal')
    axes[1].set_xlabel('L/D Ratio')
    axes[1].set_ylabel('Conversion [%]')
    axes[1].set_title('CH₄ Conversion vs L/D')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(LD, dP, 'r-o', markersize=6, linewidth=2)
    axes[2].axhline(y=max_dP, color='orange', linestyle='--', linewidth=2, label=f'Max ({max_dP} kPa)')
    axes[2].axvline(x=optimal.optimal_LD_ratio, color='red', linestyle=':', linewidth=2, label='Optimal')
    axes[2].set_xlabel('L/D Ratio')
    axes[2].set_ylabel('Pressure Drop [kPa]')
    axes[2].set_title('ΔP vs L/D')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


# ============================================================================
# HEADER
# ============================================================================
col_h1, col_h2 = st.columns([1, 8])
with col_h1:
    st.image("https://raw.githubusercontent.com/anukaranAI/methane-reactor/main/AnukaranNew7.png", width=80)
with col_h2:
    st.markdown("## Methane Decomposition Reactor Scale-Up")
    st.caption(f"**CH₄ → C + 2H₂** | {st.session_state.temperature}°C | Target H₂: {st.session_state.ind_target_H2}%")

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        for key, value in DEFAULTS.items():
            st.session_state[key] = value
        st.session_state.calibration_done = False
        st.session_state.lab_results = None
        st.session_state.industrial_results = None
        st.session_state.lab_transient_results = None
        st.session_state.industrial_transient_results = None
        st.rerun()
    
    st.markdown("---")
    
    with st.expander("🌡️ Operating Conditions", expanded=True):
        st.number_input("Temperature (°C)", 600.0, 1000.0, 
                       value=st.session_state.temperature, step=10.0, key="temperature")
        st.number_input("Inlet Pressure (bar)", 0.5, 10.0,
                       value=st.session_state.inlet_pressure, step=0.1, key="inlet_pressure")
        st.number_input("Inlet CH₄ (%)", 10.0, 100.0,
                       value=st.session_state.inlet_CH4, step=5.0, key="inlet_CH4")
    
    with st.expander("🔬 Lab Scale", expanded=False):
        st.number_input("Flow Rate (mL/min)", 10.0, 500.0,
                       value=st.session_state.lab_flow_rate, step=5.0, key="lab_flow_rate")
        st.number_input("Diameter (cm)", 0.5, 10.0,
                       value=st.session_state.lab_diameter, step=0.1, key="lab_diameter")
        st.number_input("Bed Height (cm)", 0.5, 20.0,
                       value=st.session_state.lab_bed_height, step=0.1, key="lab_bed_height")
        st.number_input("Catalyst Mass (g)", 0.1, 100.0,
                       value=st.session_state.lab_catalyst_mass, step=0.1, key="lab_catalyst_mass")
        st.number_input("Particle Size (μm)", 1.0, 1000.0,
                       value=st.session_state.lab_particle_size, step=1.0, key="lab_particle_size",
                       help="Lab scale uses fine powder (typically 10-100 μm)")
        st.number_input("Particle Density (kg/m³)", 1000.0, 5000.0,
                       value=st.session_state.lab_particle_density, step=10.0, key="lab_particle_density")
    
    with st.expander("🏭 Industrial Scale", expanded=False):
        st.number_input("Flow Rate (LPM)", 100.0, 10000.0,
                       value=st.session_state.ind_flow_rate, step=100.0, key="ind_flow_rate")
        st.number_input("Catalyst Mass (kg)", 1.0, 500.0,
                       value=st.session_state.ind_catalyst_mass, step=1.0, key="ind_catalyst_mass")
        st.number_input("Particle Size (μm)", 100.0, 10000.0,
                       value=st.session_state.ind_particle_size, step=100.0, key="ind_particle_size",
                       help="Industrial reactors use larger particles (1-5 mm = 1000-5000 μm) for lower ΔP")
        st.number_input("Particle Density (kg/m³)", 1000.0, 5000.0,
                       value=st.session_state.ind_particle_density, step=10.0, key="ind_particle_density")
        st.number_input("Target H₂ (%)", 10.0, 50.0,
                       value=st.session_state.ind_target_H2, step=1.0, key="ind_target_H2")
        c1, c2 = st.columns(2)
        c1.number_input("L/D Min", 0.5, 5.0,
                       value=st.session_state.ind_LD_min, step=0.1, key="ind_LD_min")
        c2.number_input("L/D Max", 1.0, 10.0,
                       value=st.session_state.ind_LD_max, step=0.1, key="ind_LD_max")
        st.number_input("Max ΔP (kPa)", 10.0, 200.0,
                       value=st.session_state.ind_max_dP, step=5.0, key="ind_max_dP")
        
        # Show particle size comparison
        st.caption(f"📊 Lab: {st.session_state.lab_particle_size} μm | Industrial: {st.session_state.ind_particle_size} μm ({st.session_state.ind_particle_size/1000:.1f} mm)")
    
    with st.expander("📐 Assumptions", expanded=False):
        st.number_input("Bed Porosity (ε)", 0.2, 0.6,
                       value=st.session_state.bed_porosity, step=0.01, key="bed_porosity")
        st.number_input("Particle Porosity (εp)", 0.2, 0.8,
                       value=st.session_state.particle_porosity, step=0.01, key="particle_porosity")
        st.number_input("Tortuosity (τ)", 1.0, 6.0,
                       value=st.session_state.tortuosity, step=0.1, key="tortuosity")
    
    st.markdown("---")
    if st.session_state.calibration_done:
        st.success("✅ Calibrated")
        st.caption(f"A={st.session_state.calibrated_A_value:.2e}")
        st.caption(f"Ea={st.session_state.calibrated_Ea_value:.0f} kJ/mol")
        st.caption(f"kd={st.session_state.calibrated_kd_value:.4f} 1/min")

# ============================================================================
# MAIN TABS
# ============================================================================
tabs = st.tabs([
    "📊 Experimental Data",
    "🔬 Lab Calibration",
    "⏱️ Transient Analysis",
    "🏭 Industrial Scale-Up",
    "📈 Comparison",
    "📋 Settings"
])

# ============================================================================
# TAB 1: EXPERIMENTAL DATA
# ============================================================================
with tabs[0]:
    st.header(f"📊 Experimental Data @ {st.session_state.temperature}°C")
    
    st.session_state.auto_insight = generate_auto_insight("experimental_data")
    
    available_temps = get_available_temperatures()
    closest_temp = min(available_temps, key=lambda x: abs(x - st.session_state.temperature))
    
    if st.session_state.temperature not in available_temps:
        st.warning(f"⚠️ No data for {st.session_state.temperature}°C. Showing: {closest_temp}°C")
    
    exp_data = get_experimental_data(closest_temp)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Raw Data")
        df = pd.DataFrame({
            'TOS (min)': exp_data['TOS_min'],
            'H₂ (%)': exp_data['H2_percent'],
            'CH₄ (%)': exp_data['CH4_percent'],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("### Reaction Stoichiometry")
        st.latex(r"CH_4 \rightarrow C + 2H_2")
        st.markdown("""
        - **1 mol CH₄** → **1 mol C** + **2 mol H₂**
        - Carbon deposits on catalyst surface
        - Causes deactivation over time
        """)
    
    with col2:
        st.markdown("### H₂ and CH₄ vs Time")
        fig = plot_experiment_data(exp_data, st.session_state.ind_target_H2)
        st.pyplot(fig)
        plt.close(fig)
    
    st.markdown("### Key Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Initial H₂", f"{exp_data['H2_percent'][0]:.2f}%")
    m2.metric("Final H₂", f"{exp_data['H2_percent'][-1]:.2f}%")
    m3.metric("Initial CH₄", f"{exp_data['CH4_percent'][0]:.1f}%")
    m4.metric("Final CH₄", f"{exp_data['CH4_percent'][-1]:.1f}%")

# ============================================================================
# TAB 2: LAB CALIBRATION
# ============================================================================
with tabs[1]:
    st.header("🔬 Lab Scale Kinetic Calibration")
    
    st.markdown(f"**Lab Setup:** T={st.session_state.temperature}°C, Flow={st.session_state.lab_flow_rate} mL/min, "
                f"D={st.session_state.lab_diameter} cm, L={st.session_state.lab_bed_height} cm, "
                f"Particle={st.session_state.lab_particle_size} μm")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("### Kinetic Parameters")
        
        A_local = st.slider("Pre-exponential (log₁₀ A)", 3.0, 8.0,
                           value=np.log10(st.session_state.get('calibrated_A_value', 5e4)),
                           step=0.1, key="A_calib")
        A_value = 10 ** A_local
        st.code(f"A = {A_value:.2e} 1/s")
        
        Ea_local = st.slider("Activation Energy [kJ/mol]", 80.0, 300.0,
                            value=float(st.session_state.get('calibrated_Ea_value', 150)),
                            step=5.0, key="Ea_calib")
        
        kd_local = st.slider("Deactivation kd [1/min]", 0.001, 0.1,
                            value=float(st.session_state.get('calibrated_kd_value', 0.008)),
                            step=0.001, format="%.4f", key="kd_calib")
        
        run_btn = st.button("▶️ Run Calibration", type="primary", use_container_width=True)
    
    with col2:
        if run_btn:
            with st.spinner("Running simulation..."):
                config = create_lab_reactor_config(A=A_value, Ea=Ea_local*1000)
                deact = DeactivationModel('first_order', DeactivationParams(k_d=kd_local))
                reactor = TransientReactor(config, deact)
                results = reactor.solve(t_final_min=210, dt_min=1.0)
                
                st.session_state.lab_results = results
                st.session_state.calibration_done = True
                st.session_state.calibrated_A_value = A_value
                st.session_state.calibrated_Ea_value = Ea_local
                st.session_state.calibrated_kd_value = kd_local
                
                closest_temp = min(get_available_temperatures(), 
                                   key=lambda x: abs(x - st.session_state.temperature))
                exp_data = get_experimental_data(closest_temp)
                model_H2 = np.interp(exp_data['TOS_min'], results['time_min'], results['H2_percent'])
                rmse = np.sqrt(np.mean((model_H2 - exp_data['H2_percent'])**2))
                
                st.session_state.auto_insight = generate_auto_insight("lab_calibration", {
                    'H2_t0': results['H2_percent'][0],
                    'exp_H2_t0': exp_data['H2_percent'][0],
                    'rmse': rmse
                })
            st.success("✅ Complete!")
            st.rerun()
        
        st.markdown("### Model vs Experimental")
        
        if st.session_state.lab_results is not None:
            results = st.session_state.lab_results
            closest_temp = min(get_available_temperatures(),
                               key=lambda x: abs(x - st.session_state.temperature))
            exp_data = get_experimental_data(closest_temp)
            
            model_H2 = np.interp(exp_data['TOS_min'], results['time_min'], results['H2_percent'])
            model_CH4 = np.interp(exp_data['TOS_min'], results['time_min'], results['CH4_percent'])
            rmse_H2 = np.sqrt(np.mean((model_H2 - exp_data['H2_percent'])**2))
            rmse_CH4 = np.sqrt(np.mean((model_CH4 - exp_data['CH4_percent'])**2))
            
            mc1, mc2, mc3, mc4 = st.columns(4)
            mc1.metric("Model H₂(0)", f"{results['H2_percent'][0]:.2f}%")
            mc2.metric("Exp H₂(0)", f"{exp_data['H2_percent'][0]:.2f}%")
            mc3.metric("RMSE H₂", f"{rmse_H2:.2f}%")
            mc4.metric("RMSE CH₄", f"{rmse_CH4:.2f}%")
            
            fig = plot_calibration_comparison(results, exp_data, st.session_state.ind_target_H2)
            st.pyplot(fig)
            plt.close(fig)
            
            with st.expander("📋 Detailed Comparison Table"):
                comp_df = pd.DataFrame({
                    'TOS (min)': exp_data['TOS_min'],
                    'Exp H₂': exp_data['H2_percent'],
                    'Model H₂': model_H2,
                    'Exp CH₄': exp_data['CH4_percent'],
                    'Model CH₄': model_CH4,
                })
                st.dataframe(comp_df.style.format({
                    'Exp H₂': '{:.2f}', 'Model H₂': '{:.2f}',
                    'Exp CH₄': '{:.2f}', 'Model CH₄': '{:.2f}'
                }), use_container_width=True, hide_index=True)
        else:
            st.info("👈 Adjust parameters and click **Run Calibration**")

# ============================================================================
# TAB 3: TRANSIENT ANALYSIS
# ============================================================================
with tabs[2]:
    st.header("⏱️ Transient Analysis")
    
    if not st.session_state.calibration_done:
        st.warning("⚠️ Calibrate kinetics in **Lab Calibration** tab first!")
    else:
        subtab1, subtab2 = st.tabs(["🔬 Lab Transient", "🏭 Industrial Transient"])
        
        # LAB TRANSIENT
        with subtab1:
            st.markdown("### Lab Scale Time-Dependent Analysis")
            st.caption(f"Particle size: {st.session_state.lab_particle_size} μm")
            
            col1, col2 = st.columns([1, 3])
            
            with col1:
                t_final = st.slider("Simulation Time [min]", 60, 500, 210, key="t_lab_trans")
                
                if st.button("▶️ Run Lab Transient", type="primary", use_container_width=True):
                    with st.spinner("Running..."):
                        config = create_lab_reactor_config()
                        deact = DeactivationModel('first_order',
                            DeactivationParams(k_d=st.session_state.calibrated_kd_value))
                        reactor = TransientReactor(config, deact)
                        results = reactor.solve(t_final_min=t_final, dt_min=1.0)
                        st.session_state.lab_transient_results = results
                    st.success("✅ Complete!")
                    st.rerun()
            
            with col2:
                if st.session_state.lab_transient_results:
                    results = st.session_state.lab_transient_results
                    
                    flow_mol_s = get_molar_flow_rate(
                        st.session_state.lab_flow_rate,
                        st.session_state.inlet_pressure * 1e5,
                        st.session_state.temperature + 273.15
                    ) * (st.session_state.inlet_CH4 / 100)
                    
                    fig, total_carbon = plot_transient_results(
                        results,
                        st.session_state.ind_target_H2,
                        st.session_state.lab_catalyst_mass / 1000,
                        flow_mol_s,
                        "Lab: "
                    )
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    half_idx = np.argmin(np.abs(results['activity'] - 0.5))
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Initial H₂", f"{results['H2_percent'][0]:.2f}%")
                    m2.metric("Final H₂", f"{results['H2_percent'][-1]:.2f}%")
                    m3.metric("Half-life", f"{results['time_min'][half_idx]:.0f} min")
                    m4.metric("Total Carbon", f"{total_carbon:.1f} g/kg")
        
        # INDUSTRIAL TRANSIENT
        with subtab2:
            st.markdown("### Industrial Scale Time-Dependent Analysis")
            st.caption(f"Particle size: {st.session_state.ind_particle_size} μm ({st.session_state.ind_particle_size/1000:.1f} mm)")
            
            if st.session_state.industrial_results is None:
                st.warning("⚠️ Run **Industrial Scale-Up** optimization first!")
            else:
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    opt = st.session_state.industrial_results['optimal']
                    st.markdown(f"""**Optimized Reactor:**
- D: {opt.optimal_diameter_m*100:.1f} cm
- L: {opt.optimal_height_m*100:.1f} cm
- L/D: {opt.optimal_LD_ratio:.2f}
- Particle: {st.session_state.ind_particle_size} μm
""")
                    
                    t_final_ind = st.slider("Time [min]", 60, 1000, 300, key="t_ind_trans")
                    
                    if st.button("▶️ Run Industrial Transient", type="primary", use_container_width=True):
                        with st.spinner("Running..."):
                            config = create_industrial_reactor_config(
                                opt.optimal_diameter_m, opt.optimal_height_m
                            )
                            deact = DeactivationModel('first_order',
                                DeactivationParams(k_d=st.session_state.calibrated_kd_value))
                            reactor = TransientReactor(config, deact)
                            results = reactor.solve(t_final_min=t_final_ind, dt_min=1.0)
                            st.session_state.industrial_transient_results = results
                        st.success("✅ Complete!")
                        st.rerun()
                
                with col2:
                    if st.session_state.industrial_transient_results:
                        results = st.session_state.industrial_transient_results
                        
                        flow_mol_s = get_molar_flow_rate(
                            st.session_state.ind_flow_rate * 1000,
                            st.session_state.inlet_pressure * 1e5,
                            st.session_state.temperature + 273.15
                        ) * (st.session_state.inlet_CH4 / 100)
                        
                        fig, total_carbon = plot_transient_results(
                            results,
                            st.session_state.ind_target_H2,
                            st.session_state.ind_catalyst_mass,
                            flow_mol_s,
                            "Industrial: "
                        )
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        below_target = results['H2_percent'] < st.session_state.ind_target_H2
                        if np.any(below_target):
                            regen_time = results['time_min'][np.argmax(below_target)]
                            st.warning(f"⚠️ Regeneration needed at **{regen_time:.0f} min**")
                        else:
                            st.success("✅ H₂ above target for entire simulation")
                        
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Initial H₂", f"{results['H2_percent'][0]:.2f}%")
                        m2.metric("Final H₂", f"{results['H2_percent'][-1]:.2f}%")
                        m3.metric("Total Carbon", f"{total_carbon:.1f} g/kg")

# ============================================================================
# TAB 4: INDUSTRIAL SCALE-UP
# ============================================================================
with tabs[3]:
    st.header("🏭 Industrial Scale-Up Optimization")
    
    if not st.session_state.calibration_done:
        st.warning("⚠️ Calibrate kinetics first!")
    else:
        st.success(f"✅ Calibrated: A={st.session_state.calibrated_A_value:.2e}, "
                   f"Ea={st.session_state.calibrated_Ea_value:.0f} kJ/mol")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Industrial Specs")
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Flow Rate | {st.session_state.ind_flow_rate} LPM |
| Catalyst | {st.session_state.ind_catalyst_mass} kg |
| Particle Size | {st.session_state.ind_particle_size} μm ({st.session_state.ind_particle_size/1000:.1f} mm) |
| Target H₂ | {st.session_state.ind_target_H2}% |
| L/D Range | {st.session_state.ind_LD_min}-{st.session_state.ind_LD_max} |
| Max ΔP | {st.session_state.ind_max_dP} kPa |
""")
            
            # Warning if particle size is too small
            if st.session_state.ind_particle_size < 500:
                st.warning(f"⚠️ Particle size {st.session_state.ind_particle_size} μm may cause high ΔP. Consider using 1000-5000 μm for industrial scale.")
            
            if st.button("🚀 Optimize Reactor", type="primary", use_container_width=True):
                with st.spinner("Optimizing..."):
                    V_bed = st.session_state.ind_catalyst_mass / (
                        st.session_state.ind_particle_density * (1 - st.session_state.bed_porosity)
                    )
                    
                    LD_range = np.linspace(st.session_state.ind_LD_min, 
                                           st.session_state.ind_LD_max, 21)
                    parametric = []
                    
                    for LD in LD_range:
                        D = (4 * V_bed / (np.pi * LD)) ** (1/3)
                        L = LD * D
                        
                        try:
                            config = create_industrial_reactor_config(D, L)
                            reactor = SteadyStateReactor(config)
                            results = reactor.solve(activity=1.0)
                            
                            parametric.append({
                                'LD_ratio': LD,
                                'diameter_cm': D * 100,
                                'height_cm': L * 100,
                                'H2_percent': results['outlet_H2_percent'],
                                'conversion_percent': results['outlet_conversion'],
                                'pressure_drop_kPa': results['pressure_drop_kPa'],
                            })
                        except Exception as e:
                            st.warning(f"Error at L/D={LD:.2f}: {str(e)}")
                            continue
                    
                    if not parametric:
                        st.error("❌ No valid configurations found. Check particle size and flow rate.")
                    else:
                        # Find optimal
                        valid = [p for p in parametric 
                                if p['H2_percent'] >= st.session_state.ind_target_H2 
                                and p['pressure_drop_kPa'] <= st.session_state.ind_max_dP]
                        
                        if valid:
                            best = min(valid, key=lambda x: x['pressure_drop_kPa'])
                        else:
                            best = min(parametric, key=lambda x: abs(x['H2_percent'] - st.session_state.ind_target_H2))
                        
                        optimal = ScaleUpResult(
                            optimal_diameter_m=best['diameter_cm'] / 100,
                            optimal_height_m=best['height_cm'] / 100,
                            optimal_LD_ratio=best['LD_ratio'],
                            predicted_H2_percent=best['H2_percent'],
                            predicted_conversion=best['conversion_percent'],
                            pressure_drop_kPa=best['pressure_drop_kPa'],
                            meets_target=best['H2_percent'] >= st.session_state.ind_target_H2,
                            meets_pressure_constraint=best['pressure_drop_kPa'] <= st.session_state.ind_max_dP,
                        )
                        
                        st.session_state.industrial_results = {
                            'parametric': parametric,
                            'optimal': optimal,
                        }
                        
                        st.session_state.auto_insight = generate_auto_insight("industrial", {
                            'diameter': optimal.optimal_diameter_m * 100,
                            'height': optimal.optimal_height_m * 100,
                            'H2': optimal.predicted_H2_percent,
                            'dP': optimal.pressure_drop_kPa,
                        })
                    
                st.success("✅ Complete!")
                st.rerun()
        
        with col2:
            if st.session_state.industrial_results:
                opt = st.session_state.industrial_results['optimal']
                parametric = st.session_state.industrial_results['parametric']
                
                st.markdown("### 🏆 Optimal Design")
                
                o1, o2, o3 = st.columns(3)
                o1.metric("Diameter", f"{opt.optimal_diameter_m*100:.1f} cm")
                o2.metric("Height", f"{opt.optimal_height_m*100:.1f} cm")
                o3.metric("L/D Ratio", f"{opt.optimal_LD_ratio:.2f}")
                
                o4, o5, o6 = st.columns(3)
                o4.metric("H₂ Outlet", f"{opt.predicted_H2_percent:.2f}%",
                         delta=f"{opt.predicted_H2_percent - st.session_state.ind_target_H2:+.1f}%")
                o5.metric("Conversion", f"{opt.predicted_conversion:.2f}%")
                o6.metric("ΔP", f"{opt.pressure_drop_kPa:.2f} kPa")
                
                if opt.meets_target and opt.meets_pressure_constraint:
                    st.success("✅ All constraints satisfied!")
                else:
                    st.warning("⚠️ Some constraints not met")
                
                fig = plot_parametric_study(parametric, opt, 
                                           st.session_state.ind_target_H2,
                                           st.session_state.ind_max_dP)
                st.pyplot(fig)
                plt.close(fig)
                
                with st.expander("📋 All Geometries"):
                    st.dataframe(pd.DataFrame(parametric).style.format({
                        'LD_ratio': '{:.2f}',
                        'diameter_cm': '{:.1f}',
                        'height_cm': '{:.1f}',
                        'H2_percent': '{:.2f}',
                        'conversion_percent': '{:.2f}',
                        'pressure_drop_kPa': '{:.2f}'
                    }), use_container_width=True, hide_index=True)

# ============================================================================
# TAB 5: COMPARISON
# ============================================================================
with tabs[4]:
    st.header("📈 Lab vs Industrial Comparison")
    
    if not st.session_state.calibration_done or st.session_state.industrial_results is None:
        st.warning("⚠️ Complete Lab Calibration and Industrial Optimization first!")
    else:
        opt = st.session_state.industrial_results['optimal']
        
        # Run simulations
        lab_config = create_lab_reactor_config()
        lab_reactor = SteadyStateReactor(lab_config)
        lab_results = lab_reactor.solve(activity=1.0)
        
        ind_config = create_industrial_reactor_config(opt.optimal_diameter_m, opt.optimal_height_m)
        ind_reactor = SteadyStateReactor(ind_config)
        ind_results = ind_reactor.solve(activity=1.0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔬 Lab Scale")
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Diameter | {st.session_state.lab_diameter:.1f} cm |
| Height | {st.session_state.lab_bed_height:.2f} cm |
| Flow | {st.session_state.lab_flow_rate} mL/min |
| Catalyst | {st.session_state.lab_catalyst_mass} g |
| Particle | {st.session_state.lab_particle_size} μm |
| **H₂** | **{lab_results['outlet_H2_percent']:.2f}%** |
| **CH₄** | **{lab_results['outlet_CH4_percent']:.2f}%** |
| **Conv** | **{lab_results['outlet_conversion']:.2f}%** |
| **ΔP** | **{lab_results['pressure_drop_kPa']:.3f} kPa** |
""")
        
        with col2:
            st.markdown("### 🏭 Industrial")
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Diameter | {opt.optimal_diameter_m*100:.1f} cm |
| Height | {opt.optimal_height_m*100:.1f} cm |
| L/D | {opt.optimal_LD_ratio:.2f} |
| Flow | {st.session_state.ind_flow_rate} LPM |
| Catalyst | {st.session_state.ind_catalyst_mass} kg |
| Particle | {st.session_state.ind_particle_size} μm |
| **H₂** | **{ind_results['outlet_H2_percent']:.2f}%** |
| **CH₄** | **{ind_results['outlet_CH4_percent']:.2f}%** |
| **Conv** | **{ind_results['outlet_conversion']:.2f}%** |
| **ΔP** | **{ind_results['pressure_drop_kPa']:.2f} kPa** |
""")
        
        with col3:
            st.markdown("### 📊 Scale Factors")
            flow_scale = (st.session_state.ind_flow_rate * 1000) / st.session_state.lab_flow_rate
            cat_scale = (st.session_state.ind_catalyst_mass * 1000) / st.session_state.lab_catalyst_mass
            particle_scale = st.session_state.ind_particle_size / st.session_state.lab_particle_size
            st.markdown(f"""
| Factor | Value |
|--------|-------|
| Flow | **{flow_scale:.0f}x** |
| Catalyst | **{cat_scale:.0f}x** |
| Particle | **{particle_scale:.0f}x** |
| Diameter | {(opt.optimal_diameter_m*100)/st.session_state.lab_diameter:.1f}x |
| Height | {(opt.optimal_height_m*100)/st.session_state.lab_bed_height:.1f}x |
""")
        
        st.markdown("### Performance Comparison")
        
        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        
        x = np.arange(2)
        
        # H2 comparison
        axes[0].bar(x, [lab_results['outlet_H2_percent'], ind_results['outlet_H2_percent']], 
                   color=['steelblue', 'darkorange'])
        axes[0].axhline(y=st.session_state.ind_target_H2, color='green', linestyle='--', label='Target')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(['Lab', 'Industrial'])
        axes[0].set_ylabel('H₂ [%]')
        axes[0].set_title('H₂ Production')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # CH4 comparison
        axes[1].bar(x, [lab_results['outlet_CH4_percent'], ind_results['outlet_CH4_percent']], 
                   color=['steelblue', 'darkorange'])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['Lab', 'Industrial'])
        axes[1].set_ylabel('CH₄ [%]')
        axes[1].set_title('Unreacted CH₄')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Conversion comparison
        axes[2].bar(x, [lab_results['outlet_conversion'], ind_results['outlet_conversion']], 
                   color=['steelblue', 'darkorange'])
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(['Lab', 'Industrial'])
        axes[2].set_ylabel('Conversion [%]')
        axes[2].set_title('CH₄ Conversion')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # Pressure drop comparison
        axes[3].bar(x, [lab_results['pressure_drop_kPa'], ind_results['pressure_drop_kPa']], 
                   color=['steelblue', 'darkorange'])
        axes[3].axhline(y=st.session_state.ind_max_dP, color='red', linestyle='--', label='Max ΔP')
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(['Lab', 'Industrial'])
        axes[3].set_ylabel('ΔP [kPa]')
        axes[3].set_title('Pressure Drop')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ============================================================================
# TAB 6: SETTINGS
# ============================================================================
with tabs[5]:
    st.header("📋 Current Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Operating Conditions")
        st.markdown(f"""
| Parameter | Value | Status |
|-----------|-------|--------|
| Temperature | {st.session_state.temperature} °C | Given |
| Pressure | {st.session_state.inlet_pressure} bar | Assumed |
| Inlet CH₄ | {st.session_state.inlet_CH4}% | Given |
""")
        
        st.markdown("### Lab Scale")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Flow Rate | {st.session_state.lab_flow_rate} mL/min |
| Diameter | {st.session_state.lab_diameter} cm |
| Height | {st.session_state.lab_bed_height} cm |
| Catalyst | {st.session_state.lab_catalyst_mass} g |
| **Particle Size** | **{st.session_state.lab_particle_size} μm** |
| Particle Density | {st.session_state.lab_particle_density} kg/m³ |
""")
        
        st.markdown("### Industrial Scale")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Flow Rate | {st.session_state.ind_flow_rate} LPM |
| Catalyst | {st.session_state.ind_catalyst_mass} kg |
| **Particle Size** | **{st.session_state.ind_particle_size} μm ({st.session_state.ind_particle_size/1000:.1f} mm)** |
| Particle Density | {st.session_state.ind_particle_density} kg/m³ |
| Target H₂ | {st.session_state.ind_target_H2}% |
| L/D Range | {st.session_state.ind_LD_min}-{st.session_state.ind_LD_max} |
| Max ΔP | {st.session_state.ind_max_dP} kPa |
""")
    
    with col2:
        st.markdown("### Assumptions")
        st.markdown(f"""
| Parameter | Value | Typical |
|-----------|-------|---------|
| Bed Porosity | {st.session_state.bed_porosity} | 0.35-0.45 |
| Particle Porosity | {st.session_state.particle_porosity} | 0.3-0.6 |
| Tortuosity | {st.session_state.tortuosity} | 2-4 |
""")
        
        st.markdown("### Kinetics")
        status = "Calibrated" if st.session_state.calibration_done else "Default"
        st.markdown(f"""
| Parameter | Value | Status |
|-----------|-------|--------|
| Pre-exponential (A) | {st.session_state.get('calibrated_A_value', 5e4):.2e} 1/s | {status} |
| Activation Energy | {st.session_state.get('calibrated_Ea_value', 150):.0f} kJ/mol | {status} |
| Deactivation (kd) | {st.session_state.get('calibrated_kd_value', 0.008):.4f} 1/min | {status} |
""")
        
        st.markdown("### Scale-Up Note")
        st.info(f"""
**Particle Size Difference:**
- Lab: {st.session_state.lab_particle_size} μm (fine powder)
- Industrial: {st.session_state.ind_particle_size} μm ({st.session_state.ind_particle_size/1000:.1f} mm)
- Scale factor: {st.session_state.ind_particle_size/st.session_state.lab_particle_size:.0f}x

Industrial reactors use larger particles to maintain reasonable pressure drop at high flow rates. This affects effectiveness factor (η) but is necessary for practical operation.
""")
    
    # Export
    st.markdown("### 📥 Export")
    settings = {
        'Parameter': ['Temperature (°C)', 'Pressure (bar)', 'Lab Flow (mL/min)', 'Lab D (cm)', 'Lab L (cm)', 
                     'Lab Particle (μm)', 'Industrial Flow (LPM)', 'Industrial Catalyst (kg)', 
                     'Industrial Particle (μm)', 'A (1/s)', 'Ea (kJ/mol)', 'kd (1/min)'],
        'Value': [st.session_state.temperature, st.session_state.inlet_pressure,
                 st.session_state.lab_flow_rate, st.session_state.lab_diameter,
                 st.session_state.lab_bed_height, st.session_state.lab_particle_size,
                 st.session_state.ind_flow_rate, st.session_state.ind_catalyst_mass,
                 st.session_state.ind_particle_size,
                 st.session_state.get('calibrated_A_value', 5e4),
                 st.session_state.get('calibrated_Ea_value', 150), 
                 st.session_state.get('calibrated_kd_value', 0.008)]
    }
    st.download_button("📥 Download Settings", pd.DataFrame(settings).to_csv(index=False),
                      "reactor_settings.csv", "text/csv", use_container_width=True)

# ============================================================================
# AI ASSISTANT - BOTTOM
# ============================================================================
st.markdown("---")
st.markdown("## 🤖 AI Assistant")

st.info(st.session_state.auto_insight if st.session_state.auto_insight else "👋 Run simulations to get insights.")

qa1, qa2, qa3, qa4 = st.columns(4)
with qa1:
    if st.button("📊 Explain", use_container_width=True):
        response = get_ai_response("Explain the results", get_context_string())
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
with qa2:
    if st.button("💡 Tips", use_container_width=True):
        response = get_ai_response("How to improve H2 and reduce carbon deposition?", get_context_string())
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
with qa3:
    if st.button("📐 Equations", use_container_width=True):
        st.session_state.chat_history.append({"role": "assistant", "content": """**Reaction:** CH₄ → C + 2H₂

**Rate:** r = k·a·η·C_CH₄·(1-ε)

**Arrhenius:** k = A·exp(-Ea/RT)

**Deactivation:** da/dt = -kd·a

**Ergun ΔP:** -dP/dz = 150μ(1-ε)²u/(dp²ε³) + 1.75ρ(1-ε)u²/(dpε³)

**Carbon:** 1 mol CH₄ → 1 mol C (12 g)

**Note:** Larger particles (↑dp) reduce ΔP but also reduce η"""})
        st.rerun()
with qa4:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

if st.session_state.chat_history:
    st.markdown("### Chat")
    for msg in st.session_state.chat_history[-5:]:
        if msg["role"] == "user":
            st.markdown(f"**👤 You:** {msg['content']}")
        else:
            st.markdown(f"**🤖 AI:** {msg['content']}")
        st.markdown("---")

col_in, col_btn = st.columns([5, 1])
with col_in:
    user_input = st.text_input("Ask...", key="chat_in", label_visibility="collapsed", placeholder="Type question...")
with col_btn:
    if st.button("Send", type="primary", use_container_width=True):
        if user_input:
            handle_chat(user_input)
            st.rerun()

st.markdown("---")
st.caption(f"Anukaran AI © 2024 | CH₄ → C + 2H₂ | {st.session_state.temperature}°C | Target: {st.session_state.ind_target_H2}% | Lab: {st.session_state.lab_particle_size}μm | Industrial: {st.session_state.ind_particle_size}μm")
