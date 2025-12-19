"""
╔══════════════════════════════════════════════════════════════════╗
║                         ANUKARAN AI                              ║
║          Methane Decomposition Reactor Simulator                 ║
║                                                                  ║
║   Lab Scale Calibration → Industrial Scale-Up Optimization       ║
║              With AI Assistant & Transient Analysis              ║
║                     Temperature: 800°C                           ║
║                     Target H2: 30%                               ║
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
    IndustrialScaleUp, get_assumptions_text,
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
# DEFAULT VALUES (Can be changed by user)
# ============================================================================
DEFAULTS = {
    # Lab Scale - Given
    'lab_flow_rate': 75.0,          # mL/min
    'lab_diameter': 1.8,            # cm
    'lab_bed_height': 2.59,         # cm
    'lab_catalyst_mass': 2.2,       # g
    'lab_particle_size': 10.0,      # μm
    'lab_particle_density': 2821.56, # kg/m³
    
    # Industrial Scale - Given
    'ind_flow_rate': 1960.0,        # LPM
    'ind_catalyst_mass': 57.0,      # kg
    'ind_particle_size': 10.0,      # μm
    'ind_particle_density': 2821.56, # kg/m³
    'ind_LD_min': 1.0,
    'ind_LD_max': 3.0,
    'ind_target_H2': 30.0,          # %
    'ind_max_dP': 50.0,             # kPa
    
    # Assumed Parameters (Editable)
    'bed_porosity': 0.4,
    'particle_porosity': 0.5,
    'tortuosity': 3.0,
    'inlet_pressure': 1.0,          # bar
    'temperature': 800.0,           # °C
    'inlet_CH4': 100.0,             # %
    
    # Kinetics (Initial guesses, will be calibrated)
    'pre_exponential': 5.0e4,       # 1/s
    'activation_energy': 150.0,     # kJ/mol
    'deactivation_kd': 0.008,       # 1/min
}

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
# Initialize all parameters with defaults
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Other session state
other_defaults = {
    'calibration_done': False,
    'lab_results': None,
    'lab_transient_results': None,
    'industrial_results': None,
    'industrial_transient_results': None,
    'chat_history': [],
    'auto_insight': '',
    'calibrated_A_value': 5.0e4,      # Add this
    'calibrated_Ea_value': 150.0,      # Add this
    'calibrated_kd_value': 0.008,      # Add this
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
# DYNAMIC CONFIG CLASSES
# ============================================================================

class DynamicLabConfig:
    """Lab config that reads from session state"""
    @property
    def flow_rate_mL_min(self):
        return st.session_state.lab_flow_rate
    
    @property
    def reactor_diameter_cm(self):
        return st.session_state.lab_diameter
    
    @property
    def bed_height_cm(self):
        return st.session_state.lab_bed_height
    
    @property
    def catalyst_mass_g(self):
        return st.session_state.lab_catalyst_mass
    
    @property
    def particle_size_um(self):
        return st.session_state.lab_particle_size
    
    @property
    def particle_density_kg_m3(self):
        return st.session_state.lab_particle_density
    
    @property
    def bed_porosity(self):
        return st.session_state.bed_porosity
    
    @property
    def particle_porosity(self):
        return st.session_state.particle_porosity
    
    @property
    def tortuosity(self):
        return st.session_state.tortuosity
    
    @property
    def inlet_pressure_bar(self):
        return st.session_state.inlet_pressure
    
    @property
    def temperature_C(self):
        return st.session_state.temperature
    
    @property
    def inlet_CH4_percent(self):
        return st.session_state.inlet_CH4


class DynamicIndustrialConfig:
    """Industrial config that reads from session state"""
    @property
    def flow_rate_LPM(self):
        return st.session_state.ind_flow_rate
    
    @property
    def catalyst_mass_kg(self):
        return st.session_state.ind_catalyst_mass
    
    @property
    def particle_size_um(self):
        return st.session_state.ind_particle_size
    
    @property
    def particle_density_kg_m3(self):
        return st.session_state.ind_particle_density
    
    @property
    def LD_ratio_min(self):
        return st.session_state.ind_LD_min
    
    @property
    def LD_ratio_max(self):
        return st.session_state.ind_LD_max
    
    @property
    def target_H2_percent(self):
        return st.session_state.ind_target_H2
    
    @property
    def max_pressure_drop_kPa(self):
        return st.session_state.ind_max_dP
    
    @property
    def bed_porosity(self):
        return st.session_state.bed_porosity
    
    @property
    def particle_porosity(self):
        return st.session_state.particle_porosity
    
    @property
    def tortuosity(self):
        return st.session_state.tortuosity
    
    @property
    def inlet_pressure_bar(self):
        return st.session_state.inlet_pressure
    
    @property
    def temperature_C(self):
        return st.session_state.temperature


# Create dynamic config instances
LAB_CONFIG = DynamicLabConfig()
INDUSTRIAL_CONFIG = DynamicIndustrialConfig()

# ============================================================================
# HELPER: Create ReactorConfig from dynamic settings
# ============================================================================

def create_lab_reactor_config(A: float = None, Ea: float = None):
    """Create ReactorConfig for lab scale using current session state values"""
    # Use calibrated values if available, otherwise use defaults
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
    """Create ReactorConfig for industrial scale"""
    if A is None:
        A = st.session_state.get('calibrated_A_value', st.session_state.pre_exponential)
    if Ea is None:
        Ea = st.session_state.get('calibrated_Ea_value', st.session_state.activation_energy) * 1000
    
    return ReactorConfig(
        diameter_m=diameter_m,
        bed_height_m=height_m,
        particle_diameter_m=st.session_state.ind_particle_size * 1e-6,
        particle_density=st.session_state.ind_particle_density,
        particle_porosity=st.session_state.particle_porosity,
        tortuosity=st.session_state.tortuosity,
        bed_porosity=st.session_state.bed_porosity,
        catalyst_mass_kg=st.session_state.ind_catalyst_mass,
        temperature_K=st.session_state.temperature + 273.15,
        inlet_pressure_Pa=st.session_state.inlet_pressure * 1e5,
        flow_rate_m3_s=st.session_state.ind_flow_rate / 60 / 1000,
        y_CH4_in=st.session_state.inlet_CH4 / 100,
        y_H2_in=0.0,
        pre_exponential=A,
        activation_energy=Ea,
    )

# ============================================================================
# AI ASSISTANT FUNCTIONS
# ============================================================================

def get_ai_response(prompt: str, context: str = "") -> str:
    """Get response from AI assistant"""
    if not GEMINI_API_KEY:
        return "⚠️ AI Assistant not configured. Add GEMINI_API_KEY to Streamlit secrets."
    
    try:
        ai = GeminiAssistant(GEMINI_API_KEY)
        full_prompt = f"""You are an expert chemical engineer AI assistant for a methane decomposition reactor simulator.

Context about current simulation:
{context}

User question: {prompt}

Provide a helpful, concise response. Use bullet points for clarity."""
        
        response = ai.generate_response(full_prompt, context)
        return response
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"


def generate_auto_insight(action: str, results: dict = None) -> str:
    """Generate automatic insight based on user action"""
    
    if action == "experimental_data":
        exp_data = get_experimental_data(800)
        return f"""**📊 Experimental Data Loaded ({st.session_state.temperature}°C)**
- Initial H₂: **{exp_data['H2_percent'][0]:.2f}%**
- Final H₂: **{exp_data['H2_percent'][-1]:.2f}%**
- Decay: **{((exp_data['H2_percent'][0] - exp_data['H2_percent'][-1])/exp_data['H2_percent'][0]*100):.1f}%**

💡 Catalyst deactivation observed due to carbon deposition."""

    elif action == "lab_calibration" and results:
        return f"""**🔬 Lab Calibration Results**
- Model H₂ at t=0: **{results.get('H2_t0', 0):.2f}%**
- Experimental H₂: **{results.get('exp_H2_t0', 24.13):.2f}%**
- RMSE: **{results.get('rmse', 0):.2f}%**

💡 {'Good fit! Ready for scale-up.' if results.get('rmse', 100) < 5 else 'Adjust A or Ea for better fit.'}"""

    elif action == "industrial_optimization" and results:
        return f"""**🏭 Industrial Optimization Complete**
- Diameter: **{results.get('diameter', 0):.1f} cm**
- Height: **{results.get('height', 0):.1f} cm**
- L/D: **{results.get('LD', 0):.2f}**
- H₂: **{results.get('H2', 0):.2f}%**

💡 {'✅ Target achieved!' if results.get('H2', 0) >= st.session_state.ind_target_H2 else '⚠️ Below target.'}"""

    return "👋 Welcome! Configure parameters in sidebar and run simulations."


def get_context_string() -> str:
    """Build context string for AI"""
    parts = [
        f"Temperature: {st.session_state.temperature}°C",
        f"Target H2: {st.session_state.ind_target_H2}%",
        f"Lab: D={st.session_state.lab_diameter}cm, L={st.session_state.lab_bed_height}cm, Flow={st.session_state.lab_flow_rate}mL/min",
        f"Industrial: Flow={st.session_state.ind_flow_rate}LPM, Catalyst={st.session_state.ind_catalyst_mass}kg",
        f"Assumptions: ε_bed={st.session_state.bed_porosity}, ε_p={st.session_state.particle_porosity}, τ={st.session_state.tortuosity}",
    ]
    
    if st.session_state.calibration_done:
        parts.append(f"Calibrated: A={st.session_state.pre_exponential:.2e}, Ea={st.session_state.activation_energy:.0f}kJ/mol, kd={st.session_state.deactivation_kd:.4f}")
    
    return "\n".join(parts)


def handle_chat(user_input: str):
    """Handle chat input"""
    if user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        context = get_context_string()
        response = get_ai_response(user_input, context)
        st.session_state.chat_history.append({"role": "assistant", "content": response})


# ============================================================================
# HEADER
# ============================================================================
col_h1, col_h2 = st.columns([1, 8])
with col_h1:
    st.image("https://raw.githubusercontent.com/anukaranAI/methane-reactor/main/AnukaranNew7.png", width=80)
with col_h2:
    st.markdown("## Methane Decomposition Reactor Scale-Up")
    st.caption(f"CH₄ → C + 2H₂ | **{st.session_state.temperature}°C** | Target H₂: **{st.session_state.ind_target_H2}%** | Lab → Industrial")

st.markdown("---")

# ============================================================================
# SIDEBAR - EDITABLE PARAMETERS
# ============================================================================
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Reset button
    if st.button("🔄 Reset to Defaults", use_container_width=True):
        for key, value in DEFAULTS.items():
            st.session_state[key] = value
        st.session_state.calibration_done = False
        st.session_state.lab_results = None
        st.session_state.industrial_results = None
        st.rerun()
    
    st.markdown("---")
    
    # Operating Conditions
    with st.expander("🌡️ Operating Conditions", expanded=True):
        st.number_input(
            "Temperature (°C)",
            min_value=600.0, max_value=1000.0,
            value=st.session_state.temperature,
            step=10.0,
            key="temperature",
            help="Reactor operating temperature"
        )
        
        st.number_input(
            "Inlet Pressure (bar)",
            min_value=0.5, max_value=10.0,
            value=st.session_state.inlet_pressure,
            step=0.1,
            key="inlet_pressure",
            help="Reactor inlet pressure"
        )
        
        st.number_input(
            "Inlet CH₄ (%)",
            min_value=10.0, max_value=100.0,
            value=st.session_state.inlet_CH4,
            step=5.0,
            key="inlet_CH4",
            help="CH4 concentration in feed"
        )
    
    # Lab Scale Parameters
    with st.expander("🔬 Lab Scale (Given)", expanded=False):
        st.number_input(
            "Flow Rate (mL/min)",
            min_value=10.0, max_value=500.0,
            value=st.session_state.lab_flow_rate,
            step=5.0,
            key="lab_flow_rate"
        )
        
        st.number_input(
            "Reactor Diameter (cm)",
            min_value=0.5, max_value=10.0,
            value=st.session_state.lab_diameter,
            step=0.1,
            key="lab_diameter"
        )
        
        st.number_input(
            "Bed Height (cm)",
            min_value=0.5, max_value=20.0,
            value=st.session_state.lab_bed_height,
            step=0.1,
            key="lab_bed_height"
        )
        
        st.number_input(
            "Catalyst Mass (g)",
            min_value=0.1, max_value=100.0,
            value=st.session_state.lab_catalyst_mass,
            step=0.1,
            key="lab_catalyst_mass"
        )
        
        st.number_input(
            "Particle Size (μm)",
            min_value=1.0, max_value=1000.0,
            value=st.session_state.lab_particle_size,
            step=1.0,
            key="lab_particle_size"
        )
        
        st.number_input(
            "Particle Density (kg/m³)",
            min_value=1000.0, max_value=5000.0,
            value=st.session_state.lab_particle_density,
            step=10.0,
            key="lab_particle_density"
        )
    
    # Industrial Scale Parameters
    with st.expander("🏭 Industrial Scale (Given)", expanded=False):
        st.number_input(
            "Flow Rate (LPM)",
            min_value=100.0, max_value=10000.0,
            value=st.session_state.ind_flow_rate,
            step=100.0,
            key="ind_flow_rate"
        )
        
        st.number_input(
            "Catalyst Mass (kg)",
            min_value=1.0, max_value=500.0,
            value=st.session_state.ind_catalyst_mass,
            step=1.0,
            key="ind_catalyst_mass"
        )
        
        st.number_input(
            "Target H₂ (%)",
            min_value=10.0, max_value=50.0,
            value=st.session_state.ind_target_H2,
            step=1.0,
            key="ind_target_H2"
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.number_input(
                "L/D Min",
                min_value=0.5, max_value=5.0,
                value=st.session_state.ind_LD_min,
                step=0.1,
                key="ind_LD_min"
            )
        with col2:
            st.number_input(
                "L/D Max",
                min_value=1.0, max_value=10.0,
                value=st.session_state.ind_LD_max,
                step=0.1,
                key="ind_LD_max"
            )
        
        st.number_input(
            "Max Pressure Drop (kPa)",
            min_value=10.0, max_value=200.0,
            value=st.session_state.ind_max_dP,
            step=5.0,
            key="ind_max_dP"
        )
    
    # Assumed Parameters
    with st.expander("📐 Assumed Parameters", expanded=False):
        st.number_input(
            "Bed Porosity (ε)",
            min_value=0.2, max_value=0.6,
            value=st.session_state.bed_porosity,
            step=0.01,
            key="bed_porosity",
            help="Void fraction in packed bed (typical: 0.35-0.45)"
        )
        
        st.number_input(
            "Particle Porosity (εp)",
            min_value=0.2, max_value=0.8,
            value=st.session_state.particle_porosity,
            step=0.01,
            key="particle_porosity",
            help="Internal porosity of catalyst particles"
        )
        
        st.number_input(
            "Tortuosity (τ)",
            min_value=1.0, max_value=6.0,
            value=st.session_state.tortuosity,
            step=0.1,
            key="tortuosity",
            help="Pore path length factor (typical: 2-4)"
        )
    
    # Kinetic Parameters
    with st.expander("⚡ Kinetic Parameters", expanded=False):
        st.markdown("*Adjust during calibration*")
        
        A_exp = st.slider(
            "Pre-exponential (log₁₀ A)",
            min_value=3.0, max_value=8.0,
            value=np.log10(st.session_state.pre_exponential),
            step=0.1,
            key="A_slider_sidebar"
        )
        st.session_state.pre_exponential = 10 ** A_exp
        st.caption(f"A = {st.session_state.pre_exponential:.2e} 1/s")
        
        st.number_input(
            "Activation Energy (kJ/mol)",
            min_value=80.0, max_value=300.0,
            value=st.session_state.activation_energy,
            step=5.0,
            key="activation_energy"
        )
        
        st.number_input(
            "Deactivation kd (1/min)",
            min_value=0.001, max_value=0.1,
            value=st.session_state.deactivation_kd,
            step=0.001,
            format="%.4f",
            key="deactivation_kd"
        )
    
    # Calibration status
    st.markdown("---")
    if st.session_state.calibration_done:
        st.success("✅ Kinetics Calibrated")
    else:
        st.info("⏳ Calibration pending")

# ============================================================================
# MAIN TABS
# ============================================================================
tabs = st.tabs([
    "📊 Experimental Data",
    "🔬 Lab Calibration",
    "⏱️ Transient Analysis",
    "🏭 Industrial Scale-Up",
    "📈 Comparison",
    "📋 Current Settings"
])

# ============================================================================
# TAB 1: EXPERIMENTAL DATA
# ============================================================================
with tabs[0]:
    st.header(f"📊 Experimental Data @ {st.session_state.temperature}°C")
    
    st.session_state.auto_insight = generate_auto_insight("experimental_data")
    
    # Check if we have data for the selected temperature
    available_temps = get_available_temperatures()
    closest_temp = min(available_temps, key=lambda x: abs(x - st.session_state.temperature))
    
    if st.session_state.temperature not in available_temps:
        st.warning(f"⚠️ No experimental data for {st.session_state.temperature}°C. Showing closest: {closest_temp}°C")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown(f"### Raw Data ({closest_temp}°C)")
        exp_data = get_experimental_data(closest_temp)
        df = pd.DataFrame({
            'TOS (min)': exp_data['TOS_min'],
            'H₂ (%)': exp_data['H2_percent'],
            'CH₄ (%)': exp_data['CH4_percent'],
        })
        st.dataframe(df, use_container_width=True, hide_index=True)
        
        st.markdown("### Key Metrics")
        m1, m2 = st.columns(2)
        m1.metric("Initial H₂", f"{exp_data['H2_percent'][0]:.2f}%")
        m2.metric("Final H₂", f"{exp_data['H2_percent'][-1]:.2f}%")
        
        decay = (exp_data['H2_percent'][0] - exp_data['H2_percent'][-1]) / exp_data['H2_percent'][0] * 100
        m3, m4 = st.columns(2)
        m3.metric("Decay", f"{decay:.1f}%")
        m4.metric("Target", f"{st.session_state.ind_target_H2}%")
    
    with col2:
        st.markdown("### H₂ vs Time Plot")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(exp_data['TOS_min'], exp_data['H2_percent'], 'o-', 
                color='#1f77b4', markersize=10, linewidth=2, label='Experimental H₂')
        ax.axhline(y=st.session_state.ind_target_H2, color='green', linestyle='--', 
                   linewidth=2, label=f'Target ({st.session_state.ind_target_H2}%)')
        ax.fill_between(exp_data['TOS_min'], exp_data['H2_percent'], alpha=0.3)
        
        ax.set_xlabel('Time on Stream [min]', fontsize=12)
        ax.set_ylabel('H₂ Composition [%]', fontsize=12)
        ax.set_title(f'Catalyst Deactivation @ {closest_temp}°C', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, max(45, exp_data['H2_percent'].max() + 5)])
        
        st.pyplot(fig)
        plt.close(fig)

# ============================================================================
# TAB 2: LAB CALIBRATION
# ============================================================================
# ============================================================================
# TAB 2: LAB CALIBRATION
# ============================================================================
with tabs[1]:
    st.header("🔬 Lab Scale Kinetic Calibration")
    
    st.markdown(f"""
    **Current Settings:** T={st.session_state.temperature}°C, Flow={st.session_state.lab_flow_rate} mL/min, 
    D={st.session_state.lab_diameter} cm, L={st.session_state.lab_bed_height} cm
    """)
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### Kinetic Parameters")
        st.markdown("*Adjust values for calibration:*")
        
        # Use temporary keys for local sliders (not bound to session state)
        A_local = st.slider(
            "Pre-exponential (log₁₀ A)",
            min_value=3.0, max_value=8.0,
            value=np.log10(st.session_state.pre_exponential),
            step=0.1,
            key="A_calibration_slider"  # Different key
        )
        A_value = 10 ** A_local
        st.code(f"A = {A_value:.2e} 1/s")
        
        Ea_local = st.slider(
            "Activation Energy [kJ/mol]",
            min_value=80.0, max_value=300.0,
            value=float(st.session_state.activation_energy),
            step=5.0,
            key="Ea_calibration_slider"  # Different key
        )
        
        kd_local = st.slider(
            "Deactivation kd [1/min]",
            min_value=0.001, max_value=0.1,
            value=float(st.session_state.deactivation_kd),
            step=0.001,
            format="%.4f",
            key="kd_calibration_slider"  # Different key
        )
        
        run_calibration = st.button("▶️ Run Lab Simulation", type="primary", use_container_width=True)
        
    with col2:
        st.markdown("### Model vs Experimental")
        
        # Run simulation if button clicked
        if run_calibration:
            with st.spinner("Running simulation..."):
                # Create config and run
                config = create_lab_reactor_config(A=A_value, Ea=Ea_local*1000)
                deact = DeactivationModel('first_order', DeactivationParams(k_d=kd_local))
                reactor = TransientReactor(config, deact)
                results = reactor.solve(t_final_min=210, dt_min=1.0)
                
                st.session_state.lab_results = results
                st.session_state.calibration_done = True
                
                # Store calibrated values in separate keys (not widget-bound)
                st.session_state['calibrated_A_value'] = A_value
                st.session_state['calibrated_Ea_value'] = Ea_local
                st.session_state['calibrated_kd_value'] = kd_local
                
                # Get closest experimental data
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
            
            st.success("✅ Calibration complete!")
            st.rerun()
        
        # Display results
        if st.session_state.lab_results is not None:
            results = st.session_state.lab_results
            closest_temp = min(get_available_temperatures(), 
                               key=lambda x: abs(x - st.session_state.temperature))
            exp_data = get_experimental_data(closest_temp)
            
            model_H2 = np.interp(exp_data['TOS_min'], results['time_min'], results['H2_percent'])
            rmse = np.sqrt(np.mean((model_H2 - exp_data['H2_percent'])**2))
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Model H₂(t=0)", f"{results['H2_percent'][0]:.2f}%")
            mc2.metric("Exp H₂(t=0)", f"{exp_data['H2_percent'][0]:.2f}%")
            mc3.metric("RMSE", f"{rmse:.2f}%", delta="Good" if rmse < 5 else "Adjust")
            
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(results['time_min'], results['H2_percent'], 'b-', linewidth=2, label='Model')
            ax.scatter(exp_data['TOS_min'], exp_data['H2_percent'], c='red', s=100, zorder=5, label='Experimental')
            ax.axhline(y=st.session_state.ind_target_H2, color='green', linestyle='--', linewidth=2, label='Target')
            ax.set_xlabel('Time on Stream [min]', fontsize=12)
            ax.set_ylabel('H₂ [%]', fontsize=12)
            ax.set_title('Lab Calibration Results', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, max(50, results['H2_percent'].max() + 5)])
            st.pyplot(fig)
            plt.close(fig)
            
            with st.expander("📋 Detailed Comparison"):
                comp_df = pd.DataFrame({
                    'TOS (min)': exp_data['TOS_min'],
                    'Exp H₂ (%)': exp_data['H2_percent'],
                    'Model H₂ (%)': model_H2,
                    'Error (%)': model_H2 - exp_data['H2_percent'],
                })
                st.dataframe(comp_df.style.format({
                    'Exp H₂ (%)': '{:.2f}',
                    'Model H₂ (%)': '{:.2f}',
                    'Error (%)': '{:+.2f}'
                }), use_container_width=True, hide_index=True)
        else:
            st.info("👈 Adjust parameters and click **Run Lab Simulation**")
            
# ============================================================================
# TAB 3: TRANSIENT ANALYSIS
# ============================================================================
with tabs[2]:
    st.header("⏱️ Transient Analysis")
    
    if not st.session_state.calibration_done:
        st.warning("⚠️ Please calibrate kinetics in **Lab Calibration** tab first!")
    else:
        subtab1, subtab2 = st.tabs(["🔬 Lab Transient", "🏭 Industrial Transient"])
        
        with subtab1:
            st.markdown("### Lab Scale Time-Dependent Analysis")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                t_final_lab = st.slider("Simulation Time [min]", 60, 500, 210, key="t_lab")
                
                if st.button("▶️ Run Lab Transient", type="primary", use_container_width=True):
                    with st.spinner("Running..."):
                        config = create_lab_reactor_config()
                        deact = DeactivationModel('first_order', 
                            DeactivationParams(k_d=st.session_state.deactivation_kd))
                        reactor = TransientReactor(config, deact)
                        results = reactor.solve(t_final_min=t_final_lab, dt_min=1.0)
                        st.session_state.lab_transient_results = results
                    st.success("✅ Complete!")
                    st.rerun()
            
            with col2:
                if st.session_state.lab_transient_results:
                    results = st.session_state.lab_transient_results
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    axes[0].plot(results['time_min'], results['H2_percent'], 'b-', linewidth=2)
                    axes[0].axhline(y=st.session_state.ind_target_H2, color='green', linestyle='--', label='Target')
                    axes[0].set_xlabel('Time [min]')
                    axes[0].set_ylabel('H₂ [%]')
                    axes[0].set_title('H₂ Production')
                    axes[0].legend()
                    axes[0].grid(True, alpha=0.3)
                    
                    axes[1].plot(results['time_min'], results['activity'], 'r-', linewidth=2)
                    axes[1].axhline(y=0.5, color='orange', linestyle='--', label='50%')
                    axes[1].set_xlabel('Time [min]')
                    axes[1].set_ylabel('Activity')
                    axes[1].set_title('Catalyst Deactivation')
                    axes[1].set_ylim([0, 1.1])
                    axes[1].legend()
                    axes[1].grid(True, alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)
                    
                    # Metrics
                    half_idx = np.argmin(np.abs(results['activity'] - 0.5))
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Initial H₂", f"{results['H2_percent'][0]:.2f}%")
                    m2.metric("Final H₂", f"{results['H2_percent'][-1]:.2f}%")
                    m3.metric("Half-life", f"{results['time_min'][half_idx]:.0f} min")
        
        with subtab2:
            st.markdown("### Industrial Scale Time-Dependent Analysis")
            
            if st.session_state.industrial_results is None:
                st.warning("⚠️ Run **Industrial Scale-Up** first!")
            else:
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    opt = st.session_state.industrial_results['optimal']
                    st.markdown(f"""**Optimized Reactor:**
- D: {opt.optimal_diameter_m*100:.1f} cm
- L: {opt.optimal_height_m*100:.1f} cm
- L/D: {opt.optimal_LD_ratio:.2f}
""")
                    
                    t_final_ind = st.slider("Time [min]", 60, 1000, 300, key="t_ind")
                    
                    if st.button("▶️ Run Industrial Transient", type="primary", use_container_width=True):
                        with st.spinner("Running..."):
                            D = opt.optimal_diameter_m
                            L = opt.optimal_height_m
                            config = create_industrial_reactor_config(D, L)
                            deact = DeactivationModel('first_order',
                                DeactivationParams(k_d=st.session_state.deactivation_kd))
                            reactor = TransientReactor(config, deact)
                            results = reactor.solve(t_final_min=t_final_ind, dt_min=1.0)
                            st.session_state.industrial_transient_results = results
                        st.success("✅ Complete!")
                        st.rerun()
                
                with col2:
                    if st.session_state.industrial_transient_results:
                        results = st.session_state.industrial_transient_results
                        
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                        
                        axes[0].plot(results['time_min'], results['H2_percent'], 'b-', linewidth=2)
                        axes[0].axhline(y=st.session_state.ind_target_H2, color='green', linestyle='--', label='Target')
                        axes[0].fill_between(results['time_min'], st.session_state.ind_target_H2, results['H2_percent'],
                                            where=results['H2_percent']>=st.session_state.ind_target_H2, alpha=0.3, color='green')
                        axes[0].fill_between(results['time_min'], results['H2_percent'], st.session_state.ind_target_H2,
                                            where=results['H2_percent']<st.session_state.ind_target_H2, alpha=0.3, color='red')
                        axes[0].set_xlabel('Time [min]')
                        axes[0].set_ylabel('H₂ [%]')
                        axes[0].set_title('Industrial H₂')
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)
                        
                        axes[1].plot(results['time_min'], results['activity'], 'r-', linewidth=2)
                        axes[1].set_xlabel('Time [min]')
                        axes[1].set_ylabel('Activity')
                        axes[1].set_title('Catalyst Activity')
                        axes[1].set_ylim([0, 1.1])
                        axes[1].grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        st.pyplot(fig)
                        plt.close(fig)
                        
                        # Time to target
                        below_target = results['H2_percent'] < st.session_state.ind_target_H2
                        if np.any(below_target):
                            regen_time = results['time_min'][np.argmax(below_target)]
                            st.warning(f"⚠️ Regeneration needed at **{regen_time:.0f} min**")
                        else:
                            st.success(f"✅ H₂ above target for entire simulation")

# ============================================================================
# TAB 4: INDUSTRIAL SCALE-UP
# ============================================================================
with tabs[3]:
    st.header("🏭 Industrial Scale-Up Optimization")
    
    if not st.session_state.calibration_done:
        st.warning("⚠️ Please calibrate kinetics first!")
    else:
        st.success(f"✅ Calibrated: A={st.session_state.pre_exponential:.2e}, Ea={st.session_state.activation_energy:.0f} kJ/mol")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Industrial Specs")
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Flow Rate | {st.session_state.ind_flow_rate} LPM |
| Catalyst | {st.session_state.ind_catalyst_mass} kg |
| Particle | {st.session_state.ind_particle_size} μm |
| L/D Range | {st.session_state.ind_LD_min}-{st.session_state.ind_LD_max} |
| Target H₂ | {st.session_state.ind_target_H2}% |
| Max ΔP | {st.session_state.ind_max_dP} kPa |
""")
            
            if st.button("🚀 Optimize Reactor", type="primary", use_container_width=True):
                with st.spinner("Optimizing..."):
                    # Calculate bed volume
                    V_bed = st.session_state.ind_catalyst_mass / (st.session_state.ind_particle_density * (1 - st.session_state.bed_porosity))
                    
                    # Parametric study
                    LD_range = np.linspace(st.session_state.ind_LD_min, st.session_state.ind_LD_max, 21)
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
                                'meets_H2': results['outlet_H2_percent'] >= st.session_state.ind_target_H2,
                                'meets_dP': results['pressure_drop_kPa'] <= st.session_state.ind_max_dP,
                            })
                        except:
                            continue
                    
                    # Find optimal
                    valid = [p for p in parametric if p['meets_H2'] and p['meets_dP']]
                    if valid:
                        best = min(valid, key=lambda x: x['pressure_drop_kPa'])
                    else:
                        best = min(parametric, key=lambda x: abs(x['H2_percent'] - st.session_state.ind_target_H2))
                    
                    # Create result object
                    from core.scaleup import ScaleUpResult
                    optimal = ScaleUpResult(
                        optimal_diameter_m=best['diameter_cm'] / 100,
                        optimal_height_m=best['height_cm'] / 100,
                        optimal_LD_ratio=best['LD_ratio'],
                        predicted_H2_percent=best['H2_percent'],
                        predicted_conversion=best['conversion_percent'],
                        pressure_drop_kPa=best['pressure_drop_kPa'],
                        meets_target=best['meets_H2'],
                        meets_pressure_constraint=best['meets_dP'],
                    )
                    
                    st.session_state.industrial_results = {
                        'parametric': parametric,
                        'optimal': optimal,
                    }
                    
                    st.session_state.auto_insight = generate_auto_insight("industrial_optimization", {
                        'diameter': optimal.optimal_diameter_m * 100,
                        'height': optimal.optimal_height_m * 100,
                        'LD': optimal.optimal_LD_ratio,
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
                o3.metric("L/D", f"{opt.optimal_LD_ratio:.2f}")
                
                o4, o5, o6 = st.columns(3)
                o4.metric("H₂", f"{opt.predicted_H2_percent:.2f}%",
                         delta=f"{opt.predicted_H2_percent - st.session_state.ind_target_H2:+.1f}%")
                o5.metric("Conv", f"{opt.predicted_conversion:.2f}%")
                o6.metric("ΔP", f"{opt.pressure_drop_kPa:.2f} kPa")
                
                if opt.meets_target and opt.meets_pressure_constraint:
                    st.success("✅ All constraints satisfied!")
                else:
                    st.warning("⚠️ Some constraints not met")
                
                # Plot
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                
                LD = [p['LD_ratio'] for p in parametric]
                H2 = [p['H2_percent'] for p in parametric]
                conv = [p['conversion_percent'] for p in parametric]
                dP = [p['pressure_drop_kPa'] for p in parametric]
                
                axes[0].plot(LD, H2, 'b-o', markersize=5)
                axes[0].axhline(y=st.session_state.ind_target_H2, color='g', linestyle='--')
                axes[0].axvline(x=opt.optimal_LD_ratio, color='r', linestyle=':')
                axes[0].set_xlabel('L/D')
                axes[0].set_ylabel('H₂ [%]')
                axes[0].set_title('H₂ vs L/D')
                axes[0].grid(True, alpha=0.3)
                
                axes[1].plot(LD, conv, 'g-o', markersize=5)
                axes[1].axvline(x=opt.optimal_LD_ratio, color='r', linestyle=':')
                axes[1].set_xlabel('L/D')
                axes[1].set_ylabel('Conversion [%]')
                axes[1].set_title('Conversion vs L/D')
                axes[1].grid(True, alpha=0.3)
                
                axes[2].plot(LD, dP, 'r-o', markersize=5)
                axes[2].axhline(y=st.session_state.ind_max_dP, color='orange', linestyle='--')
                axes[2].axvline(x=opt.optimal_LD_ratio, color='r', linestyle=':')
                axes[2].set_xlabel('L/D')
                axes[2].set_ylabel('ΔP [kPa]')
                axes[2].set_title('Pressure Drop vs L/D')
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                with st.expander("📋 All Geometries"):
                    st.dataframe(pd.DataFrame(parametric), use_container_width=True, hide_index=True)

# ============================================================================
# TAB 5: COMPARISON
# ============================================================================
with tabs[4]:
    st.header("📈 Lab vs Industrial Comparison")
    
    if not st.session_state.calibration_done or st.session_state.industrial_results is None:
        st.warning("⚠️ Complete Lab Calibration and Industrial Optimization first!")
    else:
        opt = st.session_state.industrial_results['optimal']
        
        # Run both simulations for comparison
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
| **H₂** | **{lab_results['outlet_H2_percent']:.2f}%** |
| **Conv** | **{lab_results['outlet_conversion']:.2f}%** |
""")
        
        with col2:
            st.markdown("### 🏭 Industrial Scale")
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Diameter | {opt.optimal_diameter_m*100:.1f} cm |
| Height | {opt.optimal_height_m*100:.1f} cm |
| L/D | {opt.optimal_LD_ratio:.2f} |
| Flow | {st.session_state.ind_flow_rate} LPM |
| Catalyst | {st.session_state.ind_catalyst_mass} kg |
| **H₂** | **{ind_results['outlet_H2_percent']:.2f}%** |
| **Conv** | **{ind_results['outlet_conversion']:.2f}%** |
""")
        
        with col3:
            st.markdown("### 📊 Scale Factors")
            flow_scale = (st.session_state.ind_flow_rate * 1000) / st.session_state.lab_flow_rate
            cat_scale = (st.session_state.ind_catalyst_mass * 1000) / st.session_state.lab_catalyst_mass
            st.markdown(f"""
| Factor | Value |
|--------|-------|
| Flow | **{flow_scale:.0f}x** |
| Catalyst | **{cat_scale:.0f}x** |
| Diameter | {(opt.optimal_diameter_m*100)/st.session_state.lab_diameter:.1f}x |
| Height | {(opt.optimal_height_m*100)/st.session_state.lab_bed_height:.1f}x |
""")
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width/2, [lab_results['outlet_H2_percent'], lab_results['outlet_conversion']], 
               width, label='Lab', color='steelblue')
        ax.bar(x + width/2, [ind_results['outlet_H2_percent'], ind_results['outlet_conversion']], 
               width, label='Industrial', color='darkorange')
        ax.axhline(y=st.session_state.ind_target_H2, color='green', linestyle='--', label='Target H₂')
        ax.set_ylabel('Percentage')
        ax.set_xticks(x)
        ax.set_xticklabels(['H₂ (%)', 'Conversion (%)'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close(fig)

# ============================================================================
# TAB 6: CURRENT SETTINGS
# ============================================================================
with tabs[5]:
    st.header("📋 Current Settings & Assumptions")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🌡️ Operating Conditions")
        st.markdown(f"""
| Parameter | Value | Status |
|-----------|-------|--------|
| Temperature | {st.session_state.temperature} °C | Given |
| Inlet Pressure | {st.session_state.inlet_pressure} bar | Assumed |
| Inlet CH₄ | {st.session_state.inlet_CH4}% | Given |
""")
        
        st.markdown("### 🔬 Lab Scale Parameters")
        st.markdown(f"""
| Parameter | Value | Status |
|-----------|-------|--------|
| Flow Rate | {st.session_state.lab_flow_rate} mL/min | Given |
| Diameter | {st.session_state.lab_diameter} cm | Given |
| Bed Height | {st.session_state.lab_bed_height} cm | Given |
| Catalyst Mass | {st.session_state.lab_catalyst_mass} g | Given |
| Particle Size | {st.session_state.lab_particle_size} μm | Given |
| Particle Density | {st.session_state.lab_particle_density} kg/m³ | Given |
""")
        
        st.markdown("### 🏭 Industrial Scale Parameters")
        st.markdown(f"""
| Parameter | Value | Status |
|-----------|-------|--------|
| Flow Rate | {st.session_state.ind_flow_rate} LPM | Given |
| Catalyst Mass | {st.session_state.ind_catalyst_mass} kg | Given |
| Particle Size | {st.session_state.ind_particle_size} μm | Given |
| Target H₂ | {st.session_state.ind_target_H2}% | Given |
| L/D Range | {st.session_state.ind_LD_min}-{st.session_state.ind_LD_max} | Given |
| Max ΔP | {st.session_state.ind_max_dP} kPa | Assumed |
""")
    
    with col2:
        st.markdown("### 📐 Assumed Parameters")
        st.info("💡 These can be changed in the sidebar")
        st.markdown(f"""
| Parameter | Value | Typical Range | Notes |
|-----------|-------|---------------|-------|
| Bed Porosity (ε) | {st.session_state.bed_porosity} | 0.35-0.45 | Void fraction |
| Particle Porosity (εp) | {st.session_state.particle_porosity} | 0.3-0.6 | Internal voids |
| Tortuosity (τ) | {st.session_state.tortuosity} | 2-4 | Path length factor |
""")
        
        st.markdown("### ⚡ Kinetic Parameters")
        if st.session_state.calibration_done:
            st.success("✅ Calibrated from experimental data")
        else:
            st.warning("⏳ Using initial guesses - calibrate in Lab tab")
        
        st.markdown(f"""
| Parameter | Value | Status |
|-----------|-------|--------|
| Pre-exponential (A) | {st.session_state.pre_exponential:.2e} 1/s | {'Calibrated' if st.session_state.calibration_done else 'Initial'} |
| Activation Energy (Ea) | {st.session_state.activation_energy} kJ/mol | {'Calibrated' if st.session_state.calibration_done else 'Initial'} |
| Deactivation (kd) | {st.session_state.deactivation_kd} 1/min | {'Calibrated' if st.session_state.calibration_done else 'Initial'} |
""")
        
        st.markdown("### 📥 Export Settings")
        settings_df = pd.DataFrame({
            'Parameter': [
                'Temperature (°C)', 'Inlet Pressure (bar)', 'Inlet CH4 (%)',
                'Lab Flow (mL/min)', 'Lab Diameter (cm)', 'Lab Height (cm)', 'Lab Catalyst (g)',
                'Ind Flow (LPM)', 'Ind Catalyst (kg)', 'Target H2 (%)',
                'Bed Porosity', 'Particle Porosity', 'Tortuosity',
                'Pre-exponential (1/s)', 'Activation Energy (kJ/mol)', 'Deactivation kd (1/min)'
            ],
            'Value': [
                st.session_state.temperature, st.session_state.inlet_pressure, st.session_state.inlet_CH4,
                st.session_state.lab_flow_rate, st.session_state.lab_diameter, st.session_state.lab_bed_height, st.session_state.lab_catalyst_mass,
                st.session_state.ind_flow_rate, st.session_state.ind_catalyst_mass, st.session_state.ind_target_H2,
                st.session_state.bed_porosity, st.session_state.particle_porosity, st.session_state.tortuosity,
                st.session_state.pre_exponential, st.session_state.activation_energy, st.session_state.deactivation_kd
            ]
        })
        
        st.download_button(
            "📥 Download All Settings (CSV)",
            settings_df.to_csv(index=False),
            "reactor_settings.csv",
            "text/csv",
            use_container_width=True
        )

# ============================================================================
# AI ASSISTANT - BOTTOM PANEL
# ============================================================================
st.markdown("---")
st.markdown("## 🤖 AI Assistant")

# Insight box
st.info(st.session_state.auto_insight if st.session_state.auto_insight else "👋 Welcome! Configure parameters in sidebar and run simulations to get insights.")

# Quick action buttons
qa1, qa2, qa3, qa4 = st.columns(4)

with qa1:
    if st.button("📊 Explain Results", use_container_width=True):
        context = get_context_string()
        response = get_ai_response("Explain the current simulation results simply.", context)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

with qa2:
    if st.button("💡 Optimization Tips", use_container_width=True):
        context = get_context_string()
        response = get_ai_response("How can I improve H2 production?", context)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

with qa3:
    if st.button("📐 Key Equations", use_container_width=True):
        st.session_state.chat_history.append({"role": "assistant", "content": """**Key Equations:**

**Reaction:** CH₄ → C + 2H₂

**Rate:** r = k · a · η · C_CH₄ · (1-ε)

**Arrhenius:** k = A · exp(-Ea/RT)

**Deactivation:** da/dt = -kd · a

**Ergun:** -dP/dz = 150μ(1-ε)²u/(dp²ε³) + 1.75ρ(1-ε)u²/(dpε³)"""})
        st.rerun()

with qa4:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Chat history
if st.session_state.chat_history:
    st.markdown("### Chat History")
    chat_container = st.container(height=250)
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**👤 You:** {msg['content']}")
            else:
                st.markdown(f"**🤖 AI:** {msg['content']}")
            st.markdown("---")

# Chat input
col_in, col_btn = st.columns([5, 1])
with col_in:
    user_input = st.text_input("Ask a question...", key="chat_input", label_visibility="collapsed", placeholder="Type your question here...")
with col_btn:
    if st.button("Send", type="primary", use_container_width=True):
        if user_input:
            handle_chat(user_input)
            st.rerun()


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption(f"Anukaran AI © 2024 | Methane Decomposition Reactor Scale-Up | {st.session_state.temperature}°C | Target H₂: {st.session_state.ind_target_H2}%")
