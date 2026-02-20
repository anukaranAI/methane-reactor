"""
KinetiScale - Catalytic Reactor Simulator
Lab Scale Calibration -> Industrial Scale-Up Optimization
With AI Assistant & Transient Analysis
Supports any single gas-phase catalytic reaction.
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- LOCAL IMPORTS ---
from ai_assistant import GeminiAssistant

from core.chemistry import Reaction, ReactionTemplate
from core.reaction_templates import get_template, get_all_template_names, get_all_templates
from core.species_db import SPECIES_DATABASE, get_species, list_gas_species
from core.experimental_data import (
    get_experimental_data_from_template,
    get_available_temperatures_from_template,
)
from core.transient_model import (
    ReactorConfig, SteadyStateReactor, TransientReactor,
    DeactivationModel, DeactivationParams, calculate_solid_deposition,
)
from core.scaleup import (
    IndustrialScaleUp, ScaleUpResult, get_assumptions_text,
)
from core.optimizer import (
    BayesianOptimizer, OptimizationConfig, OptimizationResult,
    SensitivityAnalyzer, OPTIMIZATION_VARIABLES, OPTIMIZATION_OBJECTIVES,
    create_lab_objective, create_industrial_objective,
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="KinetiScale - Catalytic Reactor Simulator",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
def init_defaults_from_template(template: ReactionTemplate):
    """Set session state defaults from a reaction template."""
    rxn = template.reaction
    return {
        # Lab Scale
        'lab_flow_rate': template.default_lab_flow_mL_min,
        'lab_diameter': template.default_lab_diameter_cm,
        'lab_bed_height': template.default_lab_height_cm,
        'lab_catalyst_mass': template.default_lab_catalyst_mass_g,
        'lab_particle_size': template.default_lab_particle_size_um,
        'lab_particle_density': template.default_lab_particle_density,
        # Industrial Scale
        'ind_flow_rate': template.default_ind_flow_LPM,
        'ind_catalyst_mass': template.default_ind_catalyst_mass_kg,
        'ind_particle_size': template.default_ind_particle_size_um,
        'ind_particle_density': template.default_ind_particle_density,
        'ind_LD_min': 1.0,
        'ind_LD_max': 3.0,
        'ind_target_product': template.default_ind_target_product_pct,
        'ind_max_dP': 50.0,
        # Assumed Parameters
        'bed_porosity': 0.4,
        'particle_porosity': 0.5,
        'tortuosity': 3.0,
        'inlet_pressure': template.default_P_bar,
        'temperature': template.default_T_C,
        'inlet_reactant_pct': 100.0,
        # Kinetics
        'pre_exponential': template.default_A,
        'activation_energy': template.default_Ea / 1000,  # Store as kJ/mol in UI
        'deactivation_kd': template.default_kd,
    }


# Initialize template selection
if 'selected_template' not in st.session_state:
    st.session_state.selected_template = 'methane_decomposition'

# Get current template
current_template = get_template(st.session_state.selected_template)
rxn = current_template.reaction

# Initialize defaults if not set
if 'defaults_initialized' not in st.session_state:
    defaults = init_defaults_from_template(current_template)
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    st.session_state.defaults_initialized = True

# Other state
other_defaults = {
    'calibration_done': False,
    'lab_results': None,
    'lab_transient_results': None,
    'industrial_results': None,
    'industrial_transient_results': None,
    'chat_history': [],
    'auto_insight': '',
    'calibrated_A_value': current_template.default_A,
    'calibrated_Ea_value': current_template.default_Ea / 1000,
    'calibrated_kd_value': current_template.default_kd,
    'lab_opt_result': None,
    'ind_opt_result': None,
}

for key, value in other_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================================
# API KEY
# ============================================================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except Exception:
    GEMINI_API_KEY = ""


# ============================================================================
# HELPER: Get display names for species
# ============================================================================
def get_product_formula():
    return rxn.species[rxn.key_product].formula

def get_reactant_formula():
    return rxn.species[rxn.key_reactant].formula

def get_product_name():
    return rxn.species[rxn.key_product].display_name

def get_reactant_name():
    return rxn.species[rxn.key_reactant].display_name


# ============================================================================
# HELPER: Solid deposition calculations
# ============================================================================

def calculate_cumulative_solid(time_min, conversion_array, flow_rate_mol_s, catalyst_mass_kg):
    """Calculate cumulative solid product deposited over time (generic)."""
    if not rxn.solid_species:
        return np.zeros_like(time_min)

    solid_name = rxn.solid_species[0]
    solid_sp = rxn.species[solid_name]
    key_stoich = abs(rxn.stoichiometry[rxn.key_reactant])
    solid_stoich = abs(rxn.stoichiometry.get(solid_name, 0))

    mol_reacted_s = conversion_array * flow_rate_mol_s
    mol_solid_s = mol_reacted_s * solid_stoich / key_stoich
    dt_s = (time_min[1] - time_min[0]) * 60 if len(time_min) > 1 else 60
    cumulative_mol = np.cumsum(mol_solid_s) * dt_s
    cumulative_g_per_kg = cumulative_mol * solid_sp.MW * 1000 / catalyst_mass_kg
    return cumulative_g_per_kg


# ============================================================================
# HELPER: Create Reactor Configs
# ============================================================================

def get_inlet_composition():
    """Build inlet mole fractions dict from session state."""
    pct = st.session_state.inlet_reactant_pct / 100
    inlet = {}
    # For multi-reactant reactions, split proportionally using template defaults
    tmpl_inlet = current_template.default_inlet
    total_tmpl = sum(tmpl_inlet.values())
    for sp_name, frac in tmpl_inlet.items():
        inlet[sp_name] = pct * frac / total_tmpl if total_tmpl > 0 else pct
    return inlet


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
        reaction=rxn,
        inlet_mole_fractions=get_inlet_composition(),
        pre_exponential=A,
        activation_energy=Ea,
    )


def create_industrial_reactor_config(diameter_m: float, height_m: float, A: float = None, Ea: float = None):
    """Create ReactorConfig for industrial scale."""
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
        reaction=rxn,
        inlet_mole_fractions=get_inlet_composition(),
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
        return "AI not configured. Add GEMINI_API_KEY to secrets."
    try:
        ai = GeminiAssistant(GEMINI_API_KEY)
        full_prompt = f"""You are a chemical engineer AI for catalytic reactor design and scale-up.
Reaction: {rxn.format_equation()} ({rxn.display_name})
Context: {context}
Question: {prompt}
Give concise, helpful response with bullet points."""
        return ai.generate_response(full_prompt, context)
    except Exception as e:
        return f"Error: {str(e)}"


def generate_auto_insight(action: str, results: dict = None) -> str:
    prod = get_product_formula()
    react = get_reactant_formula()

    if action == "experimental_data":
        temps = get_available_temperatures_from_template(current_template)
        if not temps:
            return f"**No experimental data** available for {rxn.display_name}."
        closest = min(temps, key=lambda x: abs(x - st.session_state.temperature))
        exp_data = get_experimental_data_from_template(current_template, closest)
        if exp_data is None:
            return "No experimental data at this temperature."
        return f"""**Experimental Data @ {closest}\u00b0C**
- Initial {prod}: **{exp_data['product_percent'][0]:.2f}%** | Final: **{exp_data['product_percent'][-1]:.2f}%**
- {react} increases from {exp_data['reactant_percent'][0]:.1f}% to {exp_data['reactant_percent'][-1]:.1f}%
- Catalyst deactivation observed over time"""

    elif action == "lab_calibration" and results:
        return f"""**Lab Calibration @ {st.session_state.temperature}\u00b0C**
- Model {prod}: **{results.get('prod_t0', 0):.2f}%** vs Exp: **{results.get('exp_prod_t0', 0):.2f}%**
- RMSE: **{results.get('rmse', 0):.2f}%**
- {'Good fit!' if results.get('rmse', 100) < 5 else 'Adjust A or Ea'}"""

    elif action == "transient" and results:
        return f"""**Transient Analysis**
- Initial {prod}: **{results.get('prod_initial', 0):.2f}%**
- Solid deposited: **{results.get('solid_total', 0):.1f} g/kg catalyst**
- Time to 50% activity: **{results.get('half_life', 0):.0f} min**"""

    elif action == "industrial" and results:
        target = st.session_state.ind_target_product
        return f"""**Industrial Optimization**
- Optimal: D={results.get('diameter', 0):.1f}cm, L={results.get('height', 0):.1f}cm
- {prod}: **{results.get('product', 0):.2f}%** | \u0394P: {results.get('dP', 0):.1f} kPa
- Particle: {st.session_state.ind_particle_size} \u03bcm ({st.session_state.ind_particle_size/1000:.1f} mm)
- {'\u2705 Target met!' if results.get('product', 0) >= target else '\u26a0\ufe0f Below target'}"""

    return "Welcome! Select a reaction and run simulations to get insights."


def get_context_string() -> str:
    prod = get_product_formula()
    react = get_reactant_formula()
    parts = [
        f"Reaction: {rxn.format_equation()}",
        f"Temperature: {st.session_state.temperature}\u00b0C, Target {prod}: {st.session_state.ind_target_product}%",
        f"Lab: D={st.session_state.lab_diameter}cm, L={st.session_state.lab_bed_height}cm, Flow={st.session_state.lab_flow_rate}mL/min, Particle={st.session_state.lab_particle_size}\u03bcm",
        f"Industrial: Flow={st.session_state.ind_flow_rate}LPM, Particle={st.session_state.ind_particle_size}\u03bcm ({st.session_state.ind_particle_size/1000:.1f}mm)",
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

def plot_experiment_data(exp_data, target_pct):
    """Plot experimental data: product and reactant vs time"""
    prod = get_product_formula()
    react = get_reactant_formula()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(exp_data['TOS_min'], exp_data['product_percent'], 'bo-', markersize=8, linewidth=2, label=f'{prod} (Experimental)')
    axes[0].axhline(y=target_pct, color='green', linestyle='--', linewidth=2, label=f'Target ({target_pct}%)')
    axes[0].fill_between(exp_data['TOS_min'], exp_data['product_percent'], alpha=0.3)
    axes[0].set_xlabel('Time on Stream [min]')
    axes[0].set_ylabel(f'{prod} Composition [%]')
    axes[0].set_title(f'{prod} Production')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, max(45, exp_data['product_percent'].max() + 5)])

    axes[1].plot(exp_data['TOS_min'], exp_data['reactant_percent'], 'rs-', markersize=8, linewidth=2, label=f'{react} (Experimental)')
    axes[1].fill_between(exp_data['TOS_min'], exp_data['reactant_percent'], alpha=0.3, color='red')
    axes[1].set_xlabel('Time on Stream [min]')
    axes[1].set_ylabel(f'{react} Composition [%]')
    axes[1].set_title(f'{react} at Outlet (Unreacted)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_transient_results(results, target_pct, catalyst_mass_kg, flow_mol_s, title_prefix=""):
    """Plot transient results: product, reactant, activity, solid deposition"""
    prod = get_product_formula()
    react = get_reactant_formula()

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    time = results['time_min']
    product = results['product_percent']
    reactant = results['reactant_percent']
    activity = results['activity']
    conversion = results['conversion'] / 100

    cumulative_solid = calculate_cumulative_solid(time, conversion, flow_mol_s, catalyst_mass_kg)

    # Product vs Time
    axes[0, 0].plot(time, product, 'b-', linewidth=2, label=prod)
    axes[0, 0].axhline(y=target_pct, color='green', linestyle='--', linewidth=2, label=f'Target ({target_pct}%)')
    axes[0, 0].fill_between(time, target_pct, product, where=product>=target_pct, alpha=0.3, color='green')
    axes[0, 0].fill_between(time, product, target_pct, where=product<target_pct, alpha=0.3, color='red')
    axes[0, 0].set_xlabel('Time [min]')
    axes[0, 0].set_ylabel(f'{prod} [%]')
    axes[0, 0].set_title(f'{title_prefix}{prod} Production')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Reactant vs Time
    axes[0, 1].plot(time, reactant, 'r-', linewidth=2, label=react)
    axes[0, 1].set_xlabel('Time [min]')
    axes[0, 1].set_ylabel(f'{react} [%]')
    axes[0, 1].set_title(f'{title_prefix}{react} at Outlet')
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

    # Solid Deposition vs Time
    if rxn.solid_species:
        solid_name = rxn.species[rxn.solid_species[0]].display_name
    else:
        solid_name = "Deposit"
    axes[1, 1].plot(time, cumulative_solid, 'm-', linewidth=2, label=f'Cumulative {solid_name}')
    axes[1, 1].set_xlabel('Time [min]')
    axes[1, 1].set_ylabel(f'{solid_name} [g/kg catalyst]')
    axes[1, 1].set_title(f'{title_prefix}{solid_name} Deposition')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, cumulative_solid[-1] if len(cumulative_solid) > 0 else 0


def plot_calibration_comparison(results, exp_data, target_pct):
    """Plot calibration: Model vs Experimental"""
    prod = get_product_formula()
    react = get_reactant_formula()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(results['time_min'], results['product_percent'], 'b-', linewidth=2, label='Model')
    axes[0].scatter(exp_data['TOS_min'], exp_data['product_percent'], c='red', s=100, zorder=5, label='Experimental')
    axes[0].axhline(y=target_pct, color='green', linestyle='--', linewidth=2, label=f'Target ({target_pct}%)')
    axes[0].set_xlabel('Time on Stream [min]')
    axes[0].set_ylabel(f'{prod} [%]')
    axes[0].set_title(f'{prod}: Model vs Experimental')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, max(50, results['product_percent'].max() + 5)])

    axes[1].plot(results['time_min'], results['reactant_percent'], 'b-', linewidth=2, label='Model')
    axes[1].scatter(exp_data['TOS_min'], exp_data['reactant_percent'], c='red', s=100, zorder=5, label='Experimental')
    axes[1].set_xlabel('Time on Stream [min]')
    axes[1].set_ylabel(f'{react} [%]')
    axes[1].set_title(f'{react}: Model vs Experimental')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_parametric_study(parametric, optimal, target_pct, max_dP):
    """Plot parametric study: product, conversion, pressure drop vs L/D"""
    prod = get_product_formula()
    react = get_reactant_formula()
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    LD = [p['LD_ratio'] for p in parametric]
    prod_pct = [p['product_percent'] for p in parametric]
    conv = [p['conversion_percent'] for p in parametric]
    dP = [p['pressure_drop_kPa'] for p in parametric]

    axes[0].plot(LD, prod_pct, 'b-o', markersize=6, linewidth=2)
    axes[0].axhline(y=target_pct, color='green', linestyle='--', linewidth=2, label=f'Target ({target_pct}%)')
    axes[0].axvline(x=optimal.optimal_LD_ratio, color='red', linestyle=':', linewidth=2, label=f'Optimal L/D={optimal.optimal_LD_ratio:.2f}')
    axes[0].set_xlabel('L/D Ratio')
    axes[0].set_ylabel(f'{prod} [%]')
    axes[0].set_title(f'{prod} vs L/D')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(LD, conv, 'g-o', markersize=6, linewidth=2)
    axes[1].axvline(x=optimal.optimal_LD_ratio, color='red', linestyle=':', linewidth=2, label='Optimal')
    axes[1].set_xlabel('L/D Ratio')
    axes[1].set_ylabel('Conversion [%]')
    axes[1].set_title(f'{react} Conversion vs L/D')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(LD, dP, 'r-o', markersize=6, linewidth=2)
    axes[2].axhline(y=max_dP, color='orange', linestyle='--', linewidth=2, label=f'Max ({max_dP} kPa)')
    axes[2].axvline(x=optimal.optimal_LD_ratio, color='red', linestyle=':', linewidth=2, label='Optimal')
    axes[2].set_xlabel('L/D Ratio')
    axes[2].set_ylabel('Pressure Drop [kPa]')
    axes[2].set_title('\u0394P vs L/D')
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
    st.markdown("## KinetiScale \u2014 Catalytic Reactor Simulator")
    st.caption(f"**{rxn.format_equation()}** | {st.session_state.temperature}\u00b0C | Target {get_product_formula()}: {st.session_state.ind_target_product}%")

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.header("\u2699\ufe0f Configuration")

    # --- Reaction Selector ---
    st.markdown("### Reaction")
    template_names = get_all_template_names()
    template_keys = list(template_names.keys())
    current_idx = template_keys.index(st.session_state.selected_template) if st.session_state.selected_template in template_keys else 0

    new_selection = st.selectbox(
        "Select Reaction",
        template_keys,
        index=current_idx,
        format_func=lambda k: template_names[k],
    )

    if new_selection != st.session_state.selected_template:
        # Reset everything when reaction changes
        st.session_state.selected_template = new_selection
        new_tmpl = get_template(new_selection)
        defaults = init_defaults_from_template(new_tmpl)
        for key, value in defaults.items():
            st.session_state[key] = value
        st.session_state.calibration_done = False
        st.session_state.lab_results = None
        st.session_state.industrial_results = None
        st.session_state.lab_transient_results = None
        st.session_state.industrial_transient_results = None
        st.session_state.calibrated_A_value = new_tmpl.default_A
        st.session_state.calibrated_Ea_value = new_tmpl.default_Ea / 1000
        st.session_state.calibrated_kd_value = new_tmpl.default_kd
        st.session_state.lab_opt_result = None
        st.session_state.ind_opt_result = None
        st.session_state.defaults_initialized = True
        st.rerun()

    st.caption(f"{rxn.format_equation()}")
    if current_template.description:
        st.caption(current_template.description[:120] + "..." if len(current_template.description) > 120 else current_template.description)

    st.markdown("---")

    if st.button("Reset to Defaults", use_container_width=True):
        defaults = init_defaults_from_template(current_template)
        for key, value in defaults.items():
            st.session_state[key] = value
        st.session_state.calibration_done = False
        st.session_state.lab_results = None
        st.session_state.industrial_results = None
        st.session_state.lab_transient_results = None
        st.session_state.industrial_transient_results = None
        st.rerun()

    st.markdown("---")

    with st.expander("Operating Conditions", expanded=True):
        st.number_input("Temperature (\u00b0C)", 100.0, 1200.0,
                        value=st.session_state.temperature, step=10.0, key="temperature")
        st.number_input("Inlet Pressure (bar)", 0.5, 10.0,
                        value=st.session_state.inlet_pressure, step=0.1, key="inlet_pressure")
        st.number_input(f"Inlet {get_reactant_formula()} (%)", 10.0, 100.0,
                        value=st.session_state.inlet_reactant_pct, step=5.0, key="inlet_reactant_pct")

    with st.expander("Lab Scale", expanded=False):
        st.number_input("Flow Rate (mL/min)", 10.0, 500.0,
                        value=st.session_state.lab_flow_rate, step=5.0, key="lab_flow_rate")
        st.number_input("Diameter (cm)", 0.5, 10.0,
                        value=st.session_state.lab_diameter, step=0.1, key="lab_diameter")
        st.number_input("Bed Height (cm)", 0.5, 20.0,
                        value=st.session_state.lab_bed_height, step=0.1, key="lab_bed_height")
        st.number_input("Catalyst Mass (g)", 0.1, 100.0,
                        value=st.session_state.lab_catalyst_mass, step=0.1, key="lab_catalyst_mass")
        st.number_input("Particle Size (\u03bcm)", 1.0, 1000.0,
                        value=st.session_state.lab_particle_size, step=1.0, key="lab_particle_size",
                        help="Lab scale uses fine powder (typically 10-100 \u03bcm)")
        st.number_input("Particle Density (kg/m\u00b3)", 1000.0, 5000.0,
                        value=st.session_state.lab_particle_density, step=10.0, key="lab_particle_density")

    with st.expander("Industrial Scale", expanded=False):
        st.number_input("Flow Rate (LPM)", 100.0, 10000.0,
                        value=st.session_state.ind_flow_rate, step=100.0, key="ind_flow_rate")
        st.number_input("Catalyst Mass (kg)", 1.0, 500.0,
                        value=st.session_state.ind_catalyst_mass, step=1.0, key="ind_catalyst_mass")
        st.number_input("Particle Size (\u03bcm)", 100.0, 10000.0,
                        value=st.session_state.ind_particle_size, step=100.0, key="ind_particle_size",
                        help="Industrial reactors use larger particles (1-5 mm) for lower \u0394P")
        st.number_input("Particle Density (kg/m\u00b3)", 1000.0, 5000.0,
                        value=st.session_state.ind_particle_density, step=10.0, key="ind_particle_density")
        st.number_input(f"Target {get_product_formula()} (%)", 5.0, 80.0,
                        value=st.session_state.ind_target_product, step=1.0, key="ind_target_product")
        c1, c2 = st.columns(2)
        c1.number_input("L/D Min", 0.5, 5.0,
                        value=st.session_state.ind_LD_min, step=0.1, key="ind_LD_min")
        c2.number_input("L/D Max", 1.0, 10.0,
                        value=st.session_state.ind_LD_max, step=0.1, key="ind_LD_max")
        st.number_input("Max \u0394P (kPa)", 10.0, 200.0,
                        value=st.session_state.ind_max_dP, step=5.0, key="ind_max_dP")
        st.caption(f"Lab: {st.session_state.lab_particle_size} \u03bcm | Industrial: {st.session_state.ind_particle_size} \u03bcm ({st.session_state.ind_particle_size/1000:.1f} mm)")

    with st.expander("Assumptions", expanded=False):
        st.number_input("Bed Porosity (\u03b5)", 0.2, 0.6,
                        value=st.session_state.bed_porosity, step=0.01, key="bed_porosity")
        st.number_input("Particle Porosity (\u03b5p)", 0.2, 0.8,
                        value=st.session_state.particle_porosity, step=0.01, key="particle_porosity")
        st.number_input("Tortuosity (\u03c4)", 1.0, 6.0,
                        value=st.session_state.tortuosity, step=0.1, key="tortuosity")

    st.markdown("---")
    if st.session_state.calibration_done:
        st.success("Calibrated")
        st.caption(f"A={st.session_state.calibrated_A_value:.2e}")
        st.caption(f"Ea={st.session_state.calibrated_Ea_value:.0f} kJ/mol")
        st.caption(f"kd={st.session_state.calibrated_kd_value:.4f} 1/min")


# ============================================================================
# MAIN TABS
# ============================================================================
has_exp_data = current_template.experimental_data is not None

tabs = st.tabs([
    "Experimental Data",
    "Lab Calibration",
    "Transient Analysis",
    "Industrial Scale-Up",
    "Bayesian Optimization",
    "Comparison",
    "Settings"
])

prod_f = get_product_formula()
react_f = get_reactant_formula()

# ============================================================================
# TAB 1: EXPERIMENTAL DATA
# ============================================================================
with tabs[0]:
    st.header(f"Experimental Data \u2014 {rxn.display_name}")

    if not has_exp_data:
        st.info(f"No experimental data available for **{rxn.display_name}**. You can still use the **Lab Calibration** tab to set kinetic parameters manually and proceed with scale-up.")
    else:
        st.session_state.auto_insight = generate_auto_insight("experimental_data")

        available_temps = get_available_temperatures_from_template(current_template)
        closest_temp = min(available_temps, key=lambda x: abs(x - st.session_state.temperature))

        if st.session_state.temperature not in available_temps:
            st.warning(f"No data for {st.session_state.temperature}\u00b0C. Showing: {closest_temp}\u00b0C")

        exp_data = get_experimental_data_from_template(current_template, closest_temp)

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Raw Data")
            df = pd.DataFrame({
                'TOS (min)': exp_data['TOS_min'],
                f'{prod_f} (%)': exp_data['product_percent'],
                f'{react_f} (%)': exp_data['reactant_percent'],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)

            st.markdown("### Reaction")
            st.markdown(f"**{rxn.format_equation()}**")
            st.markdown(current_template.description)

        with col2:
            st.markdown(f"### {prod_f} and {react_f} vs Time")
            fig = plot_experiment_data(exp_data, st.session_state.ind_target_product)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("### Key Metrics")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(f"Initial {prod_f}", f"{exp_data['product_percent'][0]:.2f}%")
        m2.metric(f"Final {prod_f}", f"{exp_data['product_percent'][-1]:.2f}%")
        m3.metric(f"Initial {react_f}", f"{exp_data['reactant_percent'][0]:.1f}%")
        m4.metric(f"Final {react_f}", f"{exp_data['reactant_percent'][-1]:.1f}%")


# ============================================================================
# TAB 2: LAB CALIBRATION
# ============================================================================
with tabs[1]:
    st.header(f"Lab Scale Kinetic Calibration \u2014 {rxn.display_name}")

    st.markdown(f"**Lab Setup:** T={st.session_state.temperature}\u00b0C, Flow={st.session_state.lab_flow_rate} mL/min, "
                f"D={st.session_state.lab_diameter} cm, L={st.session_state.lab_bed_height} cm, "
                f"Particle={st.session_state.lab_particle_size} \u03bcm")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Kinetic Parameters")

        A_local = st.slider("Pre-exponential (log\u2081\u2080 A)", 3.0, 8.0,
                            value=np.log10(st.session_state.get('calibrated_A_value', current_template.default_A)),
                            step=0.1, key="A_calib")
        A_value = 10 ** A_local
        st.code(f"A = {A_value:.2e} 1/s")

        Ea_local = st.slider("Activation Energy [kJ/mol]", 50.0, 300.0,
                             value=float(st.session_state.get('calibrated_Ea_value', current_template.default_Ea/1000)),
                             step=5.0, key="Ea_calib")

        kd_local = st.slider("Deactivation kd [1/min]", 0.001, 0.1,
                             value=float(st.session_state.get('calibrated_kd_value', current_template.default_kd)),
                             step=0.001, format="%.4f", key="kd_calib")

        run_btn = st.button("Run Calibration", type="primary", use_container_width=True)

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

                if has_exp_data:
                    available_temps = get_available_temperatures_from_template(current_template)
                    closest_temp = min(available_temps, key=lambda x: abs(x - st.session_state.temperature))
                    exp_data = get_experimental_data_from_template(current_template, closest_temp)
                    model_prod = np.interp(exp_data['TOS_min'], results['time_min'], results['product_percent'])
                    rmse = np.sqrt(np.mean((model_prod - exp_data['product_percent'])**2))
                    st.session_state.auto_insight = generate_auto_insight("lab_calibration", {
                        'prod_t0': results['product_percent'][0],
                        'exp_prod_t0': exp_data['product_percent'][0],
                        'rmse': rmse
                    })
            st.success("Complete!")
            st.rerun()

        st.markdown("### Model vs Experimental")

        if st.session_state.lab_results is not None:
            results = st.session_state.lab_results

            if has_exp_data:
                available_temps = get_available_temperatures_from_template(current_template)
                closest_temp = min(available_temps, key=lambda x: abs(x - st.session_state.temperature))
                exp_data = get_experimental_data_from_template(current_template, closest_temp)

                model_prod = np.interp(exp_data['TOS_min'], results['time_min'], results['product_percent'])
                model_react = np.interp(exp_data['TOS_min'], results['time_min'], results['reactant_percent'])
                rmse_prod = np.sqrt(np.mean((model_prod - exp_data['product_percent'])**2))
                rmse_react = np.sqrt(np.mean((model_react - exp_data['reactant_percent'])**2))

                mc1, mc2, mc3, mc4 = st.columns(4)
                mc1.metric(f"Model {prod_f}(0)", f"{results['product_percent'][0]:.2f}%")
                mc2.metric(f"Exp {prod_f}(0)", f"{exp_data['product_percent'][0]:.2f}%")
                mc3.metric(f"RMSE {prod_f}", f"{rmse_prod:.2f}%")
                mc4.metric(f"RMSE {react_f}", f"{rmse_react:.2f}%")

                fig = plot_calibration_comparison(results, exp_data, st.session_state.ind_target_product)
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("Detailed Comparison Table"):
                    comp_df = pd.DataFrame({
                        'TOS (min)': exp_data['TOS_min'],
                        f'Exp {prod_f}': exp_data['product_percent'],
                        f'Model {prod_f}': model_prod,
                        f'Exp {react_f}': exp_data['reactant_percent'],
                        f'Model {react_f}': model_react,
                    })
                    st.dataframe(comp_df.style.format({
                        f'Exp {prod_f}': '{:.2f}', f'Model {prod_f}': '{:.2f}',
                        f'Exp {react_f}': '{:.2f}', f'Model {react_f}': '{:.2f}'
                    }), use_container_width=True, hide_index=True)
            else:
                # No experimental data - just show model results
                mc1, mc2 = st.columns(2)
                mc1.metric(f"Model {prod_f}(0)", f"{results['product_percent'][0]:.2f}%")
                mc2.metric(f"Final {prod_f}", f"{results['product_percent'][-1]:.2f}%")

                fig, ax = plt.subplots(1, 1, figsize=(10, 4))
                ax.plot(results['time_min'], results['product_percent'], 'b-', linewidth=2, label=f'{prod_f} (Model)')
                ax.axhline(y=st.session_state.ind_target_product, color='green', linestyle='--', label=f'Target')
                ax.set_xlabel('Time on Stream [min]')
                ax.set_ylabel(f'{prod_f} [%]')
                ax.set_title(f'{prod_f} Model Prediction')
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
        else:
            st.info("Adjust parameters and click **Run Calibration**")


# ============================================================================
# TAB 3: TRANSIENT ANALYSIS
# ============================================================================
with tabs[2]:
    st.header(f"Transient Analysis \u2014 {rxn.display_name}")

    if not st.session_state.calibration_done:
        st.warning("Calibrate kinetics in **Lab Calibration** tab first!")
    else:
        subtab1, subtab2 = st.tabs(["Lab Transient", "Industrial Transient"])

        with subtab1:
            st.markdown("### Lab Scale Time-Dependent Analysis")
            st.caption(f"Particle size: {st.session_state.lab_particle_size} \u03bcm")

            col1, col2 = st.columns([1, 3])

            with col1:
                t_final = st.slider("Simulation Time [min]", 60, 500, 210, key="t_lab_trans")

                if st.button("Run Lab Transient", type="primary", use_container_width=True):
                    with st.spinner("Running..."):
                        config = create_lab_reactor_config()
                        deact = DeactivationModel('first_order',
                            DeactivationParams(k_d=st.session_state.calibrated_kd_value))
                        reactor = TransientReactor(config, deact)
                        results = reactor.solve(t_final_min=t_final, dt_min=1.0)
                        st.session_state.lab_transient_results = results
                    st.success("Complete!")
                    st.rerun()

            with col2:
                if st.session_state.lab_transient_results:
                    results = st.session_state.lab_transient_results

                    flow_mol_s = get_molar_flow_rate(
                        st.session_state.lab_flow_rate,
                        st.session_state.inlet_pressure * 1e5,
                        st.session_state.temperature + 273.15
                    ) * (st.session_state.inlet_reactant_pct / 100)

                    fig, total_solid = plot_transient_results(
                        results,
                        st.session_state.ind_target_product,
                        st.session_state.lab_catalyst_mass / 1000,
                        flow_mol_s,
                        "Lab: "
                    )
                    st.pyplot(fig)
                    plt.close(fig)

                    half_idx = np.argmin(np.abs(results['activity'] - 0.5))
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric(f"Initial {prod_f}", f"{results['product_percent'][0]:.2f}%")
                    m2.metric(f"Final {prod_f}", f"{results['product_percent'][-1]:.2f}%")
                    m3.metric("Half-life", f"{results['time_min'][half_idx]:.0f} min")
                    m4.metric("Solid Deposited", f"{total_solid:.1f} g/kg")

        with subtab2:
            st.markdown("### Industrial Scale Time-Dependent Analysis")
            st.caption(f"Particle size: {st.session_state.ind_particle_size} \u03bcm ({st.session_state.ind_particle_size/1000:.1f} mm)")

            if st.session_state.industrial_results is None:
                st.warning("Run **Industrial Scale-Up** optimization first!")
            else:
                col1, col2 = st.columns([1, 3])

                with col1:
                    opt = st.session_state.industrial_results['optimal']
                    st.markdown(f"""**Optimized Reactor:**
- D: {opt.optimal_diameter_m*100:.1f} cm
- L: {opt.optimal_height_m*100:.1f} cm
- L/D: {opt.optimal_LD_ratio:.2f}
- Particle: {st.session_state.ind_particle_size} \u03bcm
""")
                    t_final_ind = st.slider("Time [min]", 60, 1000, 300, key="t_ind_trans")

                    if st.button("Run Industrial Transient", type="primary", use_container_width=True):
                        with st.spinner("Running..."):
                            config = create_industrial_reactor_config(
                                opt.optimal_diameter_m, opt.optimal_height_m
                            )
                            deact = DeactivationModel('first_order',
                                DeactivationParams(k_d=st.session_state.calibrated_kd_value))
                            reactor = TransientReactor(config, deact)
                            results = reactor.solve(t_final_min=t_final_ind, dt_min=1.0)
                            st.session_state.industrial_transient_results = results
                        st.success("Complete!")
                        st.rerun()

                with col2:
                    if st.session_state.industrial_transient_results:
                        results = st.session_state.industrial_transient_results

                        flow_mol_s = get_molar_flow_rate(
                            st.session_state.ind_flow_rate * 1000,
                            st.session_state.inlet_pressure * 1e5,
                            st.session_state.temperature + 273.15
                        ) * (st.session_state.inlet_reactant_pct / 100)

                        fig, total_solid = plot_transient_results(
                            results,
                            st.session_state.ind_target_product,
                            st.session_state.ind_catalyst_mass,
                            flow_mol_s,
                            "Industrial: "
                        )
                        st.pyplot(fig)
                        plt.close(fig)

                        below_target = results['product_percent'] < st.session_state.ind_target_product
                        if np.any(below_target):
                            regen_time = results['time_min'][np.argmax(below_target)]
                            st.warning(f"Regeneration needed at **{regen_time:.0f} min**")
                        else:
                            st.success(f"{prod_f} above target for entire simulation")

                        m1, m2, m3 = st.columns(3)
                        m1.metric(f"Initial {prod_f}", f"{results['product_percent'][0]:.2f}%")
                        m2.metric(f"Final {prod_f}", f"{results['product_percent'][-1]:.2f}%")
                        m3.metric("Solid Deposited", f"{total_solid:.1f} g/kg")


# ============================================================================
# TAB 4: INDUSTRIAL SCALE-UP
# ============================================================================
with tabs[3]:
    st.header(f"Industrial Scale-Up \u2014 {rxn.display_name}")

    if not st.session_state.calibration_done:
        st.warning("Calibrate kinetics first!")
    else:
        st.success(f"Calibrated: A={st.session_state.calibrated_A_value:.2e}, "
                   f"Ea={st.session_state.calibrated_Ea_value:.0f} kJ/mol")

        col1, col2 = st.columns([1, 2])

        with col1:
            st.markdown("### Industrial Specs")
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Flow Rate | {st.session_state.ind_flow_rate} LPM |
| Catalyst | {st.session_state.ind_catalyst_mass} kg |
| Particle Size | {st.session_state.ind_particle_size} \u03bcm ({st.session_state.ind_particle_size/1000:.1f} mm) |
| Target {prod_f} | {st.session_state.ind_target_product}% |
| L/D Range | {st.session_state.ind_LD_min}-{st.session_state.ind_LD_max} |
| Max \u0394P | {st.session_state.ind_max_dP} kPa |
""")

            if st.session_state.ind_particle_size < 500:
                st.warning(f"Particle size {st.session_state.ind_particle_size} \u03bcm may cause high \u0394P. Consider 1000-5000 \u03bcm for industrial scale.")

            if st.button("Optimize Reactor", type="primary", use_container_width=True):
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
                                'product_percent': results['outlet_product_percent'],
                                'conversion_percent': results['outlet_conversion'],
                                'pressure_drop_kPa': results['pressure_drop_kPa'],
                            })
                        except Exception as e:
                            st.warning(f"Error at L/D={LD:.2f}: {str(e)}")
                            continue

                    if not parametric:
                        st.error("No valid configurations found. Check particle size and flow rate.")
                    else:
                        target = st.session_state.ind_target_product
                        max_dp = st.session_state.ind_max_dP
                        valid = [p for p in parametric
                                if p['product_percent'] >= target
                                and p['pressure_drop_kPa'] <= max_dp]

                        if valid:
                            best = min(valid, key=lambda x: x['pressure_drop_kPa'])
                        else:
                            best = min(parametric, key=lambda x: abs(x['product_percent'] - target))

                        optimal = ScaleUpResult(
                            optimal_diameter_m=best['diameter_cm'] / 100,
                            optimal_height_m=best['height_cm'] / 100,
                            optimal_LD_ratio=best['LD_ratio'],
                            predicted_product_percent=best['product_percent'],
                            predicted_conversion=best['conversion_percent'],
                            pressure_drop_kPa=best['pressure_drop_kPa'],
                            meets_target=best['product_percent'] >= target,
                            meets_pressure_constraint=best['pressure_drop_kPa'] <= max_dp,
                            product_name=rxn.key_product,
                            reactant_name=rxn.key_reactant,
                        )

                        st.session_state.industrial_results = {
                            'parametric': parametric,
                            'optimal': optimal,
                        }

                        st.session_state.auto_insight = generate_auto_insight("industrial", {
                            'diameter': optimal.optimal_diameter_m * 100,
                            'height': optimal.optimal_height_m * 100,
                            'product': optimal.predicted_product_percent,
                            'dP': optimal.pressure_drop_kPa,
                        })

                st.success("Complete!")
                st.rerun()

        with col2:
            if st.session_state.industrial_results:
                opt = st.session_state.industrial_results['optimal']
                parametric = st.session_state.industrial_results['parametric']

                st.markdown("### Optimal Design")

                o1, o2, o3 = st.columns(3)
                o1.metric("Diameter", f"{opt.optimal_diameter_m*100:.1f} cm")
                o2.metric("Height", f"{opt.optimal_height_m*100:.1f} cm")
                o3.metric("L/D Ratio", f"{opt.optimal_LD_ratio:.2f}")

                o4, o5, o6 = st.columns(3)
                o4.metric(f"{prod_f} Outlet", f"{opt.predicted_product_percent:.2f}%",
                          delta=f"{opt.predicted_product_percent - st.session_state.ind_target_product:+.1f}%")
                o5.metric("Conversion", f"{opt.predicted_conversion:.2f}%")
                o6.metric("\u0394P", f"{opt.pressure_drop_kPa:.2f} kPa")

                if opt.meets_target and opt.meets_pressure_constraint:
                    st.success("All constraints satisfied!")
                else:
                    st.warning("Some constraints not met")

                fig = plot_parametric_study(parametric, opt,
                                            st.session_state.ind_target_product,
                                            st.session_state.ind_max_dP)
                st.pyplot(fig)
                plt.close(fig)

                with st.expander("All Geometries"):
                    st.dataframe(pd.DataFrame(parametric).style.format({
                        'LD_ratio': '{:.2f}',
                        'diameter_cm': '{:.1f}',
                        'height_cm': '{:.1f}',
                        'product_percent': '{:.2f}',
                        'conversion_percent': '{:.2f}',
                        'pressure_drop_kPa': '{:.2f}'
                    }), use_container_width=True, hide_index=True)


# ============================================================================
# TAB 5: BAYESIAN OPTIMIZATION
# ============================================================================
with tabs[4]:
    st.header(f"Bayesian Optimization \u2014 {rxn.display_name}")

    if not st.session_state.calibration_done:
        st.warning("Calibrate kinetics in **Lab Calibration** tab first!")
    else:
        opt_subtab1, opt_subtab2 = st.tabs(["Lab Optimization", "Industrial Optimization"])

        # ---- LAB OPTIMIZATION ----
        with opt_subtab1:
            st.markdown("### Optimize Lab-Scale Parameters")
            st.caption("Use Bayesian optimization (Gaussian Process + Expected Improvement) to find optimal operating conditions.")

            col_cfg, col_res = st.columns([1, 2])

            with col_cfg:
                lab_opt_vars = {
                    'temperature_C': 'Temperature (C)',
                    'flow_rate_mL_min': 'Flow Rate (mL/min)',
                    'particle_size_um': 'Particle Size (um)',
                    'bed_height_cm': 'Bed Height (cm)',
                    'catalyst_mass_g': 'Catalyst Mass (g)',
                    'bed_porosity': 'Bed Porosity',
                }

                selected_lab_vars = st.multiselect(
                    "Variables to Optimize",
                    list(lab_opt_vars.keys()),
                    default=['temperature_C', 'flow_rate_mL_min'],
                    format_func=lambda k: lab_opt_vars[k],
                    key="lab_opt_vars"
                )

                lab_objective = st.selectbox(
                    "Objective",
                    ['product_percent', 'conversion_percent', 'min_pressure_drop', 'product_per_dP'],
                    format_func=lambda k: OPTIMIZATION_OBJECTIVES[k],
                    key="lab_opt_objective"
                )

                lab_n_iter = st.slider("Iterations", 10, 50, 25, key="lab_opt_iter")

                # Bounds
                lab_bounds = {}
                if selected_lab_vars:
                    st.markdown("#### Bounds")
                    for var in selected_lab_vars:
                        info = OPTIMIZATION_VARIABLES[var]
                        default_lo, default_hi = info['default_bounds']
                        c1, c2 = st.columns(2)
                        lo = c1.number_input(f"{info['display']} min", value=float(default_lo), key=f"lab_lo_{var}")
                        hi = c2.number_input(f"{info['display']} max", value=float(default_hi), key=f"lab_hi_{var}")
                        lab_bounds[var] = (lo, hi)

                run_lab_opt = st.button("Run Lab Optimization", type="primary",
                                        use_container_width=True, key="run_lab_opt",
                                        disabled=len(selected_lab_vars) < 1)

            with col_res:
                if run_lab_opt and selected_lab_vars:
                    base_cfg = {
                        'diameter_m': st.session_state.lab_diameter / 100,
                        'bed_height_m': st.session_state.lab_bed_height / 100,
                        'particle_diameter_m': st.session_state.lab_particle_size * 1e-6,
                        'particle_density': st.session_state.lab_particle_density,
                        'particle_porosity': st.session_state.particle_porosity,
                        'tortuosity': st.session_state.tortuosity,
                        'bed_porosity': st.session_state.bed_porosity,
                        'catalyst_mass_kg': st.session_state.lab_catalyst_mass / 1000,
                        'temperature_K': st.session_state.temperature + 273.15,
                        'inlet_pressure_Pa': st.session_state.inlet_pressure * 1e5,
                        'flow_rate_m3_s': st.session_state.lab_flow_rate / 60 / 1e6,
                        'inlet_mole_fractions': get_inlet_composition(),
                        'pre_exponential': st.session_state.calibrated_A_value,
                        'activation_energy': st.session_state.calibrated_Ea_value * 1000,
                    }

                    obj_fn = create_lab_objective(rxn, base_cfg, selected_lab_vars, lab_objective)

                    config = OptimizationConfig(
                        variable_names=selected_lab_vars,
                        bounds=[lab_bounds[v] for v in selected_lab_vars],
                        n_iterations=lab_n_iter,
                        n_initial_points=min(5, lab_n_iter // 3),
                        maximize=(lab_objective != 'min_pressure_drop'),
                    )

                    valid, msg = config.validate()
                    if not valid:
                        st.error(msg)
                    else:
                        progress_bar = st.progress(0, text="Optimizing...")
                        status_text = st.empty()

                        def lab_callback(iteration, params, value):
                            progress_bar.progress(iteration / lab_n_iter,
                                text=f"Iteration {iteration}/{lab_n_iter}")

                        optimizer = BayesianOptimizer(config)
                        result = optimizer.optimize(obj_fn, callback=lab_callback)
                        progress_bar.progress(1.0, text="Complete!")

                        st.session_state.lab_opt_result = result
                        st.rerun()

                if 'lab_opt_result' in st.session_state and st.session_state.lab_opt_result is not None:
                    result = st.session_state.lab_opt_result

                    st.markdown("### Optimization Results")
                    st.success(result.message)

                    # Best parameters
                    st.markdown("#### Best Parameters")
                    best_cols = st.columns(min(len(result.best_params), 4))
                    for i, (name, val) in enumerate(result.best_params.items()):
                        info = OPTIMIZATION_VARIABLES.get(name, {'display': name, 'unit': ''})
                        best_cols[i % len(best_cols)].metric(info['display'], f"{val:.3g}")

                    obj_label = OPTIMIZATION_OBJECTIVES.get(
                        st.session_state.get('lab_opt_objective', 'product_percent'), 'Objective')
                    best_val = result.best_value
                    if st.session_state.get('lab_opt_objective') == 'min_pressure_drop':
                        best_val = -best_val
                    st.metric("Best Objective Value", f"{best_val:.4f}")

                    # Convergence plot
                    st.markdown("#### Convergence")
                    fig_conv, ax_conv = plt.subplots(1, 1, figsize=(10, 4))
                    ax_conv.plot(range(1, len(result.best_so_far)+1), result.best_so_far,
                                 'b-o', markersize=4, linewidth=2, label='Best so far')
                    ax_conv.plot(range(1, len(result.convergence)+1), result.convergence,
                                 'r.', alpha=0.4, markersize=6, label='Each trial')
                    ax_conv.set_xlabel('Iteration')
                    ax_conv.set_ylabel('Objective Value')
                    ax_conv.set_title('Bayesian Optimization Convergence')
                    ax_conv.legend()
                    ax_conv.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_conv)
                    plt.close(fig_conv)

                    # Trial history table
                    with st.expander("All Trials"):
                        trial_data = []
                        for i, trial in enumerate(result.all_trials):
                            row = {'Trial': i+1}
                            row.update(trial['params'])
                            row['Objective'] = trial['value']
                            trial_data.append(row)
                        st.dataframe(pd.DataFrame(trial_data), use_container_width=True, hide_index=True)
                else:
                    st.info("Select variables, set bounds, and click **Run Lab Optimization**.")

        # ---- INDUSTRIAL OPTIMIZATION ----
        with opt_subtab2:
            st.markdown("### Optimize Industrial-Scale Parameters")
            st.caption("Find optimal industrial reactor configuration using Bayesian optimization.")

            col_cfg2, col_res2 = st.columns([1, 2])

            with col_cfg2:
                ind_opt_vars = {
                    'temperature_C': 'Temperature (C)',
                    'LD_ratio': 'L/D Ratio',
                    'ind_flow_rate_LPM': 'Flow Rate (LPM)',
                    'ind_particle_size_um': 'Particle Size (um)',
                    'ind_catalyst_mass_kg': 'Catalyst Mass (kg)',
                    'bed_porosity': 'Bed Porosity',
                }

                selected_ind_vars = st.multiselect(
                    "Variables to Optimize",
                    list(ind_opt_vars.keys()),
                    default=['LD_ratio', 'ind_particle_size_um'],
                    format_func=lambda k: ind_opt_vars[k],
                    key="ind_opt_vars"
                )

                ind_objective = st.selectbox(
                    "Objective",
                    ['product_percent', 'conversion_percent', 'min_pressure_drop', 'product_per_dP'],
                    format_func=lambda k: OPTIMIZATION_OBJECTIVES[k],
                    index=3,
                    key="ind_opt_objective"
                )

                ind_n_iter = st.slider("Iterations", 10, 50, 25, key="ind_opt_iter")

                ind_bounds = {}
                if selected_ind_vars:
                    st.markdown("#### Bounds")
                    for var in selected_ind_vars:
                        info = OPTIMIZATION_VARIABLES[var]
                        default_lo, default_hi = info['default_bounds']
                        c1, c2 = st.columns(2)
                        lo = c1.number_input(f"{info['display']} min", value=float(default_lo), key=f"ind_lo_{var}")
                        hi = c2.number_input(f"{info['display']} max", value=float(default_hi), key=f"ind_hi_{var}")
                        ind_bounds[var] = (lo, hi)

                run_ind_opt = st.button("Run Industrial Optimization", type="primary",
                                        use_container_width=True, key="run_ind_opt",
                                        disabled=len(selected_ind_vars) < 1)

            with col_res2:
                if run_ind_opt and selected_ind_vars:
                    V_bed = st.session_state.ind_catalyst_mass / (
                        st.session_state.ind_particle_density * (1 - st.session_state.bed_porosity))
                    default_LD = 2.0
                    D_default = (4 * V_bed / (np.pi * default_LD)) ** (1/3)
                    L_default = default_LD * D_default

                    base_cfg = {
                        'diameter_m': D_default,
                        'bed_height_m': L_default,
                        'particle_diameter_m': st.session_state.ind_particle_size * 1e-6,
                        'particle_density': st.session_state.ind_particle_density,
                        'particle_porosity': st.session_state.particle_porosity,
                        'tortuosity': st.session_state.tortuosity,
                        'bed_porosity': st.session_state.bed_porosity,
                        'catalyst_mass_kg': st.session_state.ind_catalyst_mass,
                        'temperature_K': st.session_state.temperature + 273.15,
                        'inlet_pressure_Pa': st.session_state.inlet_pressure * 1e5,
                        'flow_rate_m3_s': st.session_state.ind_flow_rate / 60 / 1000,
                        'inlet_mole_fractions': get_inlet_composition(),
                        'pre_exponential': st.session_state.calibrated_A_value,
                        'activation_energy': st.session_state.calibrated_Ea_value * 1000,
                    }

                    obj_fn = create_industrial_objective(rxn, base_cfg, selected_ind_vars, ind_objective)

                    config = OptimizationConfig(
                        variable_names=selected_ind_vars,
                        bounds=[ind_bounds[v] for v in selected_ind_vars],
                        n_iterations=ind_n_iter,
                        n_initial_points=min(5, ind_n_iter // 3),
                        maximize=(ind_objective != 'min_pressure_drop'),
                    )

                    valid, msg = config.validate()
                    if not valid:
                        st.error(msg)
                    else:
                        progress_bar2 = st.progress(0, text="Optimizing...")

                        def ind_callback(iteration, params, value):
                            progress_bar2.progress(iteration / ind_n_iter,
                                text=f"Iteration {iteration}/{ind_n_iter}")

                        optimizer = BayesianOptimizer(config)
                        result = optimizer.optimize(obj_fn, callback=ind_callback)
                        progress_bar2.progress(1.0, text="Complete!")

                        st.session_state.ind_opt_result = result
                        st.rerun()

                if 'ind_opt_result' in st.session_state and st.session_state.ind_opt_result is not None:
                    result = st.session_state.ind_opt_result

                    st.markdown("### Optimization Results")
                    st.success(result.message)

                    st.markdown("#### Best Parameters")
                    best_cols = st.columns(min(len(result.best_params), 4))
                    for i, (name, val) in enumerate(result.best_params.items()):
                        info = OPTIMIZATION_VARIABLES.get(name, {'display': name, 'unit': ''})
                        best_cols[i % len(best_cols)].metric(info['display'], f"{val:.3g}")

                    obj_label = OPTIMIZATION_OBJECTIVES.get(
                        st.session_state.get('ind_opt_objective', 'product_percent'), 'Objective')
                    best_val = result.best_value
                    if st.session_state.get('ind_opt_objective') == 'min_pressure_drop':
                        best_val = -best_val
                    st.metric("Best Objective Value", f"{best_val:.4f}")

                    # Convergence plot
                    st.markdown("#### Convergence")
                    fig_conv2, ax_conv2 = plt.subplots(1, 1, figsize=(10, 4))
                    ax_conv2.plot(range(1, len(result.best_so_far)+1), result.best_so_far,
                                  'b-o', markersize=4, linewidth=2, label='Best so far')
                    ax_conv2.plot(range(1, len(result.convergence)+1), result.convergence,
                                  'r.', alpha=0.4, markersize=6, label='Each trial')
                    ax_conv2.set_xlabel('Iteration')
                    ax_conv2.set_ylabel('Objective Value')
                    ax_conv2.set_title('Bayesian Optimization Convergence')
                    ax_conv2.legend()
                    ax_conv2.grid(True, alpha=0.3)
                    plt.tight_layout()
                    st.pyplot(fig_conv2)
                    plt.close(fig_conv2)

                    # Sensitivity analysis
                    with st.expander("Sensitivity Analysis"):
                        st.caption("Running one-at-a-time sensitivity sweep...")
                        analyzer = SensitivityAnalyzer(
                            result.variable_names,
                            [ind_bounds[v] for v in result.variable_names]
                        )

                        best_obj_key = st.session_state.get('ind_opt_objective', 'product_percent')
                        sens_obj_fn = create_industrial_objective(
                            rxn, base_cfg, result.variable_names, best_obj_key)
                        importance = analyzer.analyze(sens_obj_fn, n_samples=15)

                        fig_sens, ax_sens = plt.subplots(1, 1, figsize=(10, 4))
                        names = [OPTIMIZATION_VARIABLES.get(n, {'display': n})['display'] for n in importance.keys()]
                        scores = list(importance.values())
                        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(scores)))
                        ax_sens.barh(names, scores, color=colors)
                        ax_sens.set_xlabel('Relative Importance')
                        ax_sens.set_title('Parameter Sensitivity')
                        ax_sens.grid(True, alpha=0.3, axis='x')
                        plt.tight_layout()
                        st.pyplot(fig_sens)
                        plt.close(fig_sens)

                    with st.expander("All Trials"):
                        trial_data = []
                        for i, trial in enumerate(result.all_trials):
                            row = {'Trial': i+1}
                            row.update(trial['params'])
                            row['Objective'] = trial['value']
                            trial_data.append(row)
                        st.dataframe(pd.DataFrame(trial_data), use_container_width=True, hide_index=True)
                else:
                    st.info("Select variables, set bounds, and click **Run Industrial Optimization**.")


# ============================================================================
# TAB 6: COMPARISON
# ============================================================================
with tabs[5]:
    st.header(f"Lab vs Industrial Comparison \u2014 {rxn.display_name}")

    if not st.session_state.calibration_done or st.session_state.industrial_results is None:
        st.warning("Complete Lab Calibration and Industrial Optimization first!")
    else:
        opt = st.session_state.industrial_results['optimal']

        lab_config = create_lab_reactor_config()
        lab_reactor = SteadyStateReactor(lab_config)
        lab_results = lab_reactor.solve(activity=1.0)

        ind_config = create_industrial_reactor_config(opt.optimal_diameter_m, opt.optimal_height_m)
        ind_reactor = SteadyStateReactor(ind_config)
        ind_results = ind_reactor.solve(activity=1.0)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Lab Scale")
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Diameter | {st.session_state.lab_diameter:.1f} cm |
| Height | {st.session_state.lab_bed_height:.2f} cm |
| Flow | {st.session_state.lab_flow_rate} mL/min |
| Catalyst | {st.session_state.lab_catalyst_mass} g |
| Particle | {st.session_state.lab_particle_size} \u03bcm |
| **{prod_f}** | **{lab_results['outlet_product_percent']:.2f}%** |
| **{react_f}** | **{lab_results['outlet_reactant_percent']:.2f}%** |
| **Conv** | **{lab_results['outlet_conversion']:.2f}%** |
| **\u0394P** | **{lab_results['pressure_drop_kPa']:.3f} kPa** |
""")

        with col2:
            st.markdown("### Industrial")
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Diameter | {opt.optimal_diameter_m*100:.1f} cm |
| Height | {opt.optimal_height_m*100:.1f} cm |
| L/D | {opt.optimal_LD_ratio:.2f} |
| Flow | {st.session_state.ind_flow_rate} LPM |
| Catalyst | {st.session_state.ind_catalyst_mass} kg |
| Particle | {st.session_state.ind_particle_size} \u03bcm |
| **{prod_f}** | **{ind_results['outlet_product_percent']:.2f}%** |
| **{react_f}** | **{ind_results['outlet_reactant_percent']:.2f}%** |
| **Conv** | **{ind_results['outlet_conversion']:.2f}%** |
| **\u0394P** | **{ind_results['pressure_drop_kPa']:.2f} kPa** |
""")

        with col3:
            st.markdown("### Scale Factors")
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

        axes[0].bar(x, [lab_results['outlet_product_percent'], ind_results['outlet_product_percent']],
                    color=['steelblue', 'darkorange'])
        axes[0].axhline(y=st.session_state.ind_target_product, color='green', linestyle='--', label='Target')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(['Lab', 'Industrial'])
        axes[0].set_ylabel(f'{prod_f} [%]')
        axes[0].set_title(f'{prod_f} Production')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')

        axes[1].bar(x, [lab_results['outlet_reactant_percent'], ind_results['outlet_reactant_percent']],
                    color=['steelblue', 'darkorange'])
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(['Lab', 'Industrial'])
        axes[1].set_ylabel(f'{react_f} [%]')
        axes[1].set_title(f'Unreacted {react_f}')
        axes[1].grid(True, alpha=0.3, axis='y')

        axes[2].bar(x, [lab_results['outlet_conversion'], ind_results['outlet_conversion']],
                    color=['steelblue', 'darkorange'])
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(['Lab', 'Industrial'])
        axes[2].set_ylabel('Conversion [%]')
        axes[2].set_title(f'{react_f} Conversion')
        axes[2].grid(True, alpha=0.3, axis='y')

        axes[3].bar(x, [lab_results['pressure_drop_kPa'], ind_results['pressure_drop_kPa']],
                    color=['steelblue', 'darkorange'])
        axes[3].axhline(y=st.session_state.ind_max_dP, color='red', linestyle='--', label='Max \u0394P')
        axes[3].set_xticks(x)
        axes[3].set_xticklabels(['Lab', 'Industrial'])
        axes[3].set_ylabel('\u0394P [kPa]')
        axes[3].set_title('Pressure Drop')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


# ============================================================================
# TAB 7: SETTINGS
# ============================================================================
with tabs[6]:
    st.header("Current Settings")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Reaction")
        st.markdown(f"**{rxn.display_name}:** {rxn.format_equation()}")
        st.markdown(f"Key reactant: {get_reactant_name()} ({react_f})")
        st.markdown(f"Key product: {get_product_name()} ({prod_f})")
        if rxn.solid_species:
            solids = ", ".join(rxn.species[s].display_name for s in rxn.solid_species)
            st.markdown(f"Solid products: {solids}")
        st.markdown(f"\u0394H = {rxn.delta_H/1000:.1f} kJ/mol ({'\u2191 endothermic' if rxn.delta_H > 0 else '\u2193 exothermic'})")

        st.markdown("### Operating Conditions")
        st.markdown(f"""
| Parameter | Value | Status |
|-----------|-------|--------|
| Temperature | {st.session_state.temperature} \u00b0C | Given |
| Pressure | {st.session_state.inlet_pressure} bar | Assumed |
| Inlet {react_f} | {st.session_state.inlet_reactant_pct}% | Given |
""")

        st.markdown("### Lab Scale")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Flow Rate | {st.session_state.lab_flow_rate} mL/min |
| Diameter | {st.session_state.lab_diameter} cm |
| Height | {st.session_state.lab_bed_height} cm |
| Catalyst | {st.session_state.lab_catalyst_mass} g |
| **Particle Size** | **{st.session_state.lab_particle_size} \u03bcm** |
| Particle Density | {st.session_state.lab_particle_density} kg/m\u00b3 |
""")

        st.markdown("### Industrial Scale")
        st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Flow Rate | {st.session_state.ind_flow_rate} LPM |
| Catalyst | {st.session_state.ind_catalyst_mass} kg |
| **Particle Size** | **{st.session_state.ind_particle_size} \u03bcm ({st.session_state.ind_particle_size/1000:.1f} mm)** |
| Particle Density | {st.session_state.ind_particle_density} kg/m\u00b3 |
| Target {prod_f} | {st.session_state.ind_target_product}% |
| L/D Range | {st.session_state.ind_LD_min}-{st.session_state.ind_LD_max} |
| Max \u0394P | {st.session_state.ind_max_dP} kPa |
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
| Pre-exponential (A) | {st.session_state.get('calibrated_A_value', current_template.default_A):.2e} 1/s | {status} |
| Activation Energy | {st.session_state.get('calibrated_Ea_value', current_template.default_Ea/1000):.0f} kJ/mol | {status} |
| Deactivation (kd) | {st.session_state.get('calibrated_kd_value', current_template.default_kd):.4f} 1/min | {status} |
""")

        st.markdown("### Scale-Up Note")
        st.info(f"""
**Particle Size Difference:**
- Lab: {st.session_state.lab_particle_size} \u03bcm (fine powder)
- Industrial: {st.session_state.ind_particle_size} \u03bcm ({st.session_state.ind_particle_size/1000:.1f} mm)
- Scale factor: {st.session_state.ind_particle_size/st.session_state.lab_particle_size:.0f}x

Industrial reactors use larger particles to maintain reasonable pressure drop at high flow rates. This affects effectiveness factor (\u03b7) but is necessary for practical operation.
""")

    # Export
    st.markdown("### Export")
    settings = {
        'Parameter': [
            'Reaction', f'Temperature (\u00b0C)', 'Pressure (bar)',
            'Lab Flow (mL/min)', 'Lab D (cm)', 'Lab L (cm)',
            'Lab Particle (\u03bcm)', 'Industrial Flow (LPM)', 'Industrial Catalyst (kg)',
            'Industrial Particle (\u03bcm)', 'A (1/s)', 'Ea (kJ/mol)', 'kd (1/min)'],
        'Value': [
            rxn.format_equation(),
            st.session_state.temperature, st.session_state.inlet_pressure,
            st.session_state.lab_flow_rate, st.session_state.lab_diameter,
            st.session_state.lab_bed_height, st.session_state.lab_particle_size,
            st.session_state.ind_flow_rate, st.session_state.ind_catalyst_mass,
            st.session_state.ind_particle_size,
            st.session_state.get('calibrated_A_value', current_template.default_A),
            st.session_state.get('calibrated_Ea_value', current_template.default_Ea/1000),
            st.session_state.get('calibrated_kd_value', current_template.default_kd)]
    }
    st.download_button("Download Settings", pd.DataFrame(settings).to_csv(index=False),
                       "kinetiscale_settings.csv", "text/csv", use_container_width=True)


# ============================================================================
# AI ASSISTANT - BOTTOM
# ============================================================================
st.markdown("---")
st.markdown("## AI Assistant")

st.info(st.session_state.auto_insight if st.session_state.auto_insight else "Welcome! Select a reaction and run simulations to get insights.")

qa1, qa2, qa3, qa4 = st.columns(4)
with qa1:
    if st.button("Explain", use_container_width=True):
        response = get_ai_response("Explain the results", get_context_string())
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
with qa2:
    if st.button("Tips", use_container_width=True):
        response = get_ai_response(f"How to improve {prod_f} production and reduce catalyst deactivation?", get_context_string())
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()
with qa3:
    if st.button("Equations", use_container_width=True):
        eq_text = f"""**Reaction:** {rxn.format_equation()}

**Rate:** r = k\u00b7a\u00b7\u03b7\u00b7C_{react_f}\u00b7(1-\u03b5)

**Arrhenius:** k = A\u00b7exp(-Ea/RT)

**Deactivation:** da/dt = -kd\u00b7a

**Ergun \u0394P:** -dP/dz = 150\u03bc(1-\u03b5)\u00b2u/(dp\u00b2\u03b5\u00b3) + 1.75\u03c1(1-\u03b5)u\u00b2/(dp\u03b5\u00b3)

**Note:** Larger particles (\u2191dp) reduce \u0394P but also reduce \u03b7"""
        st.session_state.chat_history.append({"role": "assistant", "content": eq_text})
        st.rerun()
with qa4:
    if st.button("Clear", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

if st.session_state.chat_history:
    st.markdown("### Chat")
    for msg in st.session_state.chat_history[-5:]:
        if msg["role"] == "user":
            st.markdown(f"**You:** {msg['content']}")
        else:
            st.markdown(f"**AI:** {msg['content']}")
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
st.caption(f"KinetiScale | {rxn.format_equation()} | {st.session_state.temperature}\u00b0C | Target: {st.session_state.ind_target_product}% | Lab: {st.session_state.lab_particle_size}\u03bcm | Industrial: {st.session_state.ind_particle_size}\u03bcm")
