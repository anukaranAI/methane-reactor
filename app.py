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
    LAB_CONFIG, INDUSTRIAL_CONFIG,
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
    initial_sidebar_state="collapsed"
)

# ============================================================================
# SESSION STATE
# ============================================================================
defaults = {
    'calibrated_A': 5.0e4,
    'calibrated_Ea': 150000.0,
    'calibrated_kd': 0.008,
    'calibration_done': False,
    'lab_results': None,
    'lab_transient_results': None,
    'industrial_results': None,
    'industrial_transient_results': None,
    'chat_history': [],
    'auto_insight': '',
}

for key, value in defaults.items():
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

Provide a helpful, concise response. Use bullet points for clarity. Include relevant numbers from the context."""
        
        response = ai.generate_response(full_prompt, context)
        return response
    except Exception as e:
        return f"⚠️ AI Error: {str(e)}"


def generate_auto_insight(action: str, results: dict = None) -> str:
    """Generate automatic insight based on user action"""
    
    if action == "experimental_data":
        exp_data = get_experimental_data(800)
        return f"""**📊 Experimental Data Loaded (800°C)**
- Initial H₂: **{exp_data['H2_percent'][0]:.2f}%**
- Final H₂: **{exp_data['H2_percent'][-1]:.2f}%** (after 210 min)
- Decay: **{((exp_data['H2_percent'][0] - exp_data['H2_percent'][-1])/exp_data['H2_percent'][0]*100):.1f}%**

💡 The catalyst shows significant deactivation over time due to carbon deposition."""

    elif action == "lab_calibration" and results:
        return f"""**🔬 Lab Calibration Results**
- Model H₂ at t=0: **{results.get('H2_t0', 0):.2f}%**
- Experimental H₂: **{results.get('exp_H2_t0', 24.13):.2f}%**
- RMSE: **{results.get('rmse', 0):.2f}%**

💡 {'Good fit! Ready for scale-up.' if results.get('rmse', 100) < 5 else 'Try adjusting A or Ea for better fit.'}"""

    elif action == "industrial_optimization" and results:
        return f"""**🏭 Industrial Optimization Complete**
- Optimal Diameter: **{results.get('diameter', 0):.1f} cm**
- Optimal Height: **{results.get('height', 0):.1f} cm**
- L/D Ratio: **{results.get('LD', 0):.2f}**
- Predicted H₂: **{results.get('H2', 0):.2f}%**
- Pressure Drop: **{results.get('dP', 0):.2f} kPa**

💡 {'✅ Target H₂ of 30% achieved!' if results.get('H2', 0) >= 30 else '⚠️ H₂ below target. Consider adjusting parameters.'}"""

    elif action == "transient_lab" and results:
        return f"""**⏱️ Lab Transient Analysis**
- Initial H₂: **{results.get('H2_initial', 0):.2f}%**
- Final H₂: **{results.get('H2_final', 0):.2f}%**
- Catalyst Half-life: **{results.get('half_life', 0):.0f} min**

💡 Plan catalyst regeneration based on acceptable H₂ threshold."""

    elif action == "transient_industrial" and results:
        return f"""**⏱️ Industrial Transient Analysis**
- Initial H₂: **{results.get('H2_initial', 0):.2f}%**
- Time above 30% H₂: **{results.get('time_above_target', 'N/A')}**

💡 Schedule catalyst regeneration before H₂ drops below 30%."""

    return "👋 Welcome! Run a simulation to get AI insights."


def get_context_string() -> str:
    """Build context string for AI from current state"""
    parts = [
        f"Temperature: 800°C, Target H2: 30%",
        f"Lab: D={LAB_CONFIG.reactor_diameter_cm}cm, L={LAB_CONFIG.bed_height_cm}cm, Flow={LAB_CONFIG.flow_rate_mL_min}mL/min, Catalyst={LAB_CONFIG.catalyst_mass_g}g",
        f"Industrial: Flow={INDUSTRIAL_CONFIG.flow_rate_LPM}LPM, Catalyst={INDUSTRIAL_CONFIG.catalyst_mass_kg}kg, L/D range={INDUSTRIAL_CONFIG.LD_ratio_min}-{INDUSTRIAL_CONFIG.LD_ratio_max}",
    ]
    
    if st.session_state.calibration_done:
        parts.append(f"Calibrated kinetics: A={st.session_state.calibrated_A:.2e} 1/s, Ea={st.session_state.calibrated_Ea/1000:.0f} kJ/mol, kd={st.session_state.calibrated_kd:.4f} 1/min")
    
    if st.session_state.lab_results:
        parts.append(f"Lab results: Initial H2={st.session_state.lab_results['H2_percent'][0]:.2f}%")
    
    if st.session_state.industrial_results:
        opt = st.session_state.industrial_results.get('optimal')
        if opt:
            parts.append(f"Industrial optimal: D={opt.optimal_diameter_m*100:.1f}cm, L={opt.optimal_height_m*100:.1f}cm, H2={opt.predicted_H2_percent:.2f}%, ΔP={opt.pressure_drop_kPa:.2f}kPa")
    
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
    st.caption("CH₄ → C + 2H₂ | **800°C** | Target H₂: **30%** | Lab → Industrial")

st.markdown("---")

# ============================================================================
# SIDEBAR - PARAMETERS SUMMARY
# ============================================================================
with st.sidebar:
    st.header("📋 Parameters")
    
    st.markdown("### 🔬 Lab Scale")
    st.markdown(f"""
- Flow: {LAB_CONFIG.flow_rate_mL_min} mL/min
- D: {LAB_CONFIG.reactor_diameter_cm} cm
- L: {LAB_CONFIG.bed_height_cm} cm
- Catalyst: {LAB_CONFIG.catalyst_mass_g} g
- Particle: {LAB_CONFIG.particle_size_um} μm
""")
    
    st.markdown("### 🏭 Industrial Scale")
    st.markdown(f"""
- Flow: {INDUSTRIAL_CONFIG.flow_rate_LPM} LPM
- Catalyst: {INDUSTRIAL_CONFIG.catalyst_mass_kg} kg
- Particle: {INDUSTRIAL_CONFIG.particle_size_um} μm
- L/D: {INDUSTRIAL_CONFIG.LD_ratio_min}-{INDUSTRIAL_CONFIG.LD_ratio_max}
""")
    
    st.markdown("### ⚙️ Assumptions")
    st.markdown(f"""
- Bed porosity: {LAB_CONFIG.bed_porosity}
- Particle porosity: {LAB_CONFIG.particle_porosity}
- Max ΔP: {INDUSTRIAL_CONFIG.max_pressure_drop_kPa} kPa
""")
    
    if st.session_state.calibration_done:
        st.markdown("### ✅ Calibrated")
        st.markdown(f"""
- A: {st.session_state.calibrated_A:.2e}
- Ea: {st.session_state.calibrated_Ea/1000:.0f} kJ/mol
- kd: {st.session_state.calibrated_kd:.4f}
""")

# ============================================================================
# MAIN TABS
# ============================================================================
tabs = st.tabs([
    "📊 Experimental Data",
    "🔬 Lab Calibration",
    "⏱️ Transient Analysis",
    "🏭 Industrial Scale-Up",
    "📈 Comparison",
    "📋 Assumptions"
])

# ============================================================================
# TAB 1: EXPERIMENTAL DATA
# ============================================================================
with tabs[0]:
    st.header("📊 Experimental Data @ 800°C")
    
    st.session_state.auto_insight = generate_auto_insight("experimental_data")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Raw Data")
        exp_data = get_experimental_data(800)
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
        m4.metric("Target", "30%")
    
    with col2:
        st.markdown("### H₂ vs Time Plot")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(exp_data['TOS_min'], exp_data['H2_percent'], 'o-', 
                color='#1f77b4', markersize=10, linewidth=2, label='Experimental H₂')
        ax.axhline(y=30, color='green', linestyle='--', linewidth=2, label='Target (30%)')
        ax.fill_between(exp_data['TOS_min'], exp_data['H2_percent'], alpha=0.3)
        
        ax.set_xlabel('Time on Stream [min]', fontsize=12)
        ax.set_ylabel('H₂ Composition [%]', fontsize=12)
        ax.set_title('Catalyst Deactivation @ 800°C', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 35])
        
        st.pyplot(fig)
        plt.close(fig)

# ============================================================================
# TAB 2: LAB CALIBRATION
# ============================================================================
with tabs[1]:
    st.header("🔬 Lab Scale Kinetic Calibration")
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown("### Kinetic Parameters")
        st.markdown("Adjust to match experimental data:")
        
        A_exp = st.slider(
            "Pre-exponential (A) - log₁₀ scale",
            min_value=3.0, max_value=8.0,
            value=np.log10(st.session_state.calibrated_A),
            step=0.1,
            key="A_slider"
        )
        A_value = 10 ** A_exp
        st.code(f"A = {A_value:.2e} 1/s")
        
        Ea_value = st.slider(
            "Activation Energy [kJ/mol]",
            min_value=100.0, max_value=250.0,
            value=st.session_state.calibrated_Ea / 1000,
            step=5.0,
            key="Ea_slider"
        )
        
        kd_value = st.slider(
            "Deactivation rate kd [1/min]",
            min_value=0.001, max_value=0.05,
            value=st.session_state.calibrated_kd,
            step=0.001,
            format="%.4f",
            key="kd_slider"
        )
        
        if st.button("▶️ Run Lab Simulation", type="primary", use_container_width=True):
            with st.spinner("Running simulation..."):
                config = ReactorConfig.from_lab_config(LAB_CONFIG, A=A_value, Ea=Ea_value*1000)
                deact = DeactivationModel('first_order', DeactivationParams(k_d=kd_value))
                reactor = TransientReactor(config, deact)
                results = reactor.solve(t_final_min=210, dt_min=1.0)
                
                st.session_state.lab_results = results
                st.session_state.calibrated_A = A_value
                st.session_state.calibrated_Ea = Ea_value * 1000
                st.session_state.calibrated_kd = kd_value
                st.session_state.calibration_done = True
                
                exp_data = get_experimental_data(800)
                model_H2 = np.interp(exp_data['TOS_min'], results['time_min'], results['H2_percent'])
                rmse = np.sqrt(np.mean((model_H2 - exp_data['H2_percent'])**2))
                
                st.session_state.auto_insight = generate_auto_insight("lab_calibration", {
                    'H2_t0': results['H2_percent'][0],
                    'exp_H2_t0': exp_data['H2_percent'][0],
                    'rmse': rmse
                })
            
            st.success("✅ Calibration complete!")
            st.rerun()
    
    with col2:
        st.markdown("### Model vs Experimental")
        
        if st.session_state.lab_results is not None:
            results = st.session_state.lab_results
            exp_data = get_experimental_data(800)
            
            model_H2 = np.interp(exp_data['TOS_min'], results['time_min'], results['H2_percent'])
            rmse = np.sqrt(np.mean((model_H2 - exp_data['H2_percent'])**2))
            
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Model H₂(t=0)", f"{results['H2_percent'][0]:.2f}%")
            mc2.metric("Exp H₂(t=0)", f"{exp_data['H2_percent'][0]:.2f}%")
            mc3.metric("RMSE", f"{rmse:.2f}%")
            
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(results['time_min'], results['H2_percent'], 'b-', linewidth=2, label='Model')
            ax.scatter(exp_data['TOS_min'], exp_data['H2_percent'], c='red', s=100, zorder=5, label='Experimental')
            ax.axhline(y=30, color='green', linestyle='--', linewidth=2, label='Target 30%')
            ax.set_xlabel('Time on Stream [min]', fontsize=12)
            ax.set_ylabel('H₂ [%]', fontsize=12)
            ax.set_title('Lab Calibration Results', fontsize=14, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim([0, 40])
            st.pyplot(fig)
            plt.close(fig)
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
                        config = ReactorConfig.from_lab_config(
                            LAB_CONFIG,
                            A=st.session_state.calibrated_A,
                            Ea=st.session_state.calibrated_Ea
                        )
                        deact = DeactivationModel('first_order', 
                            DeactivationParams(k_d=st.session_state.calibrated_kd))
                        reactor = TransientReactor(config, deact)
                        results = reactor.solve(t_final_min=t_final_lab, dt_min=1.0)
                        
                        st.session_state.lab_transient_results = results
                        
                        half_life_idx = np.argmin(np.abs(results['activity'] - 0.5))
                        half_life = results['time_min'][half_life_idx]
                        
                        st.session_state.auto_insight = generate_auto_insight("transient_lab", {
                            'H2_initial': results['H2_percent'][0],
                            'H2_final': results['H2_percent'][-1],
                            'half_life': half_life
                        })
                    
                    st.success("✅ Complete!")
                    st.rerun()
            
            with col2:
                if st.session_state.lab_transient_results:
                    results = st.session_state.lab_transient_results
                    
                    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                    
                    axes[0].plot(results['time_min'], results['H2_percent'], 'b-', linewidth=2)
                    axes[0].axhline(y=30, color='green', linestyle='--', label='Target')
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
                            scaleup = IndustrialScaleUp(
                                calibrated_A=st.session_state.calibrated_A,
                                calibrated_Ea=st.session_state.calibrated_Ea,
                                calibrated_kd=st.session_state.calibrated_kd,
                            )
                            results = scaleup.transient_industrial(opt.optimal_LD_ratio, t_final_ind)
                            st.session_state.industrial_transient_results = results
                            
                            below_target = results['H2_percent'] < 30
                            if np.any(below_target):
                                time_above = results['time_min'][np.argmax(below_target)]
                            else:
                                time_above = t_final_ind
                            
                            st.session_state.auto_insight = generate_auto_insight("transient_industrial", {
                                'H2_initial': results['H2_percent'][0],
                                'time_above_target': f"{time_above:.0f} min"
                            })
                        
                        st.success("✅ Complete!")
                        st.rerun()
                
                with col2:
                    if st.session_state.industrial_transient_results:
                        results = st.session_state.industrial_transient_results
                        
                        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                        
                        axes[0].plot(results['time_min'], results['H2_percent'], 'b-', linewidth=2)
                        axes[0].axhline(y=30, color='green', linestyle='--', linewidth=2, label='Target')
                        axes[0].fill_between(results['time_min'], 30, results['H2_percent'],
                                            where=results['H2_percent']>=30, alpha=0.3, color='green')
                        axes[0].fill_between(results['time_min'], results['H2_percent'], 30,
                                            where=results['H2_percent']<30, alpha=0.3, color='red')
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

# ============================================================================
# TAB 4: INDUSTRIAL SCALE-UP
# ============================================================================
with tabs[3]:
    st.header("🏭 Industrial Scale-Up Optimization")
    
    if not st.session_state.calibration_done:
        st.warning("⚠️ Please calibrate kinetics first!")
    else:
        st.success(f"✅ Using calibrated kinetics: A={st.session_state.calibrated_A:.2e}, Ea={st.session_state.calibrated_Ea/1000:.0f} kJ/mol")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Industrial Specs")
            st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Flow Rate | {INDUSTRIAL_CONFIG.flow_rate_LPM} LPM |
| Catalyst | {INDUSTRIAL_CONFIG.catalyst_mass_kg} kg |
| Particle | {INDUSTRIAL_CONFIG.particle_size_um} μm |
| L/D Range | {INDUSTRIAL_CONFIG.LD_ratio_min}-{INDUSTRIAL_CONFIG.LD_ratio_max} |
| Target H₂ | {INDUSTRIAL_CONFIG.target_H2_percent}% |
| Max ΔP | {INDUSTRIAL_CONFIG.max_pressure_drop_kPa} kPa |
""")
            
            if st.button("🚀 Optimize Reactor", type="primary", use_container_width=True):
                with st.spinner("Optimizing..."):
                    scaleup = IndustrialScaleUp(
                        calibrated_A=st.session_state.calibrated_A,
                        calibrated_Ea=st.session_state.calibrated_Ea,
                        calibrated_kd=st.session_state.calibrated_kd,
                    )
                    
                    parametric = scaleup.parametric_study(n_points=21)
                    optimal = scaleup.optimize_geometry()
                    
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
                o4.metric("H₂", f"{opt.predicted_H2_percent:.2f}%")
                o5.metric("Conv", f"{opt.predicted_conversion:.2f}%")
                o6.metric("ΔP", f"{opt.pressure_drop_kPa:.2f} kPa")
                
                # Plot
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                
                LD = [p['LD_ratio'] for p in parametric]
                H2 = [p['H2_percent'] for p in parametric]
                conv = [p['conversion_percent'] for p in parametric]
                dP = [p['pressure_drop_kPa'] for p in parametric]
                
                axes[0].plot(LD, H2, 'b-o', markersize=5)
                axes[0].axhline(y=30, color='g', linestyle='--')
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
                axes[2].axhline(y=50, color='orange', linestyle='--')
                axes[2].axvline(x=opt.optimal_LD_ratio, color='r', linestyle=':')
                axes[2].set_xlabel('L/D')
                axes[2].set_ylabel('ΔP [kPa]')
                axes[2].set_title('Pressure Drop vs L/D')
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

# ============================================================================
# TAB 5: COMPARISON
# ============================================================================
with tabs[4]:
    st.header("📈 Lab vs Industrial Comparison")
    
    if not st.session_state.calibration_done or st.session_state.industrial_results is None:
        st.warning("⚠️ Complete Lab Calibration and Industrial Optimization first!")
    else:
        scaleup = IndustrialScaleUp(
            calibrated_A=st.session_state.calibrated_A,
            calibrated_Ea=st.session_state.calibrated_Ea,
            calibrated_kd=st.session_state.calibrated_kd,
        )
        
        opt = st.session_state.industrial_results['optimal']
        comparison = scaleup.compare_scales(industrial_LD=opt.optimal_LD_ratio)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🔬 Lab")
            lab = comparison['lab']
            st.markdown(f"""
- D: {lab['diameter_cm']:.1f} cm
- L: {lab['height_cm']:.2f} cm
- Flow: {lab['flow_rate']}
- **H₂: {lab['H2_percent']:.2f}%**
""")
        
        with col2:
            st.markdown("### 🏭 Industrial")
            ind = comparison['industrial']
            st.markdown(f"""
- D: {ind['diameter_cm']:.1f} cm
- L: {ind['height_cm']:.1f} cm
- L/D: {ind['LD_ratio']:.1f}
- **H₂: {ind['H2_percent']:.2f}%**
""")
        
        with col3:
            st.markdown("### 📊 Scale")
            sf = comparison['scale_factors']
            st.markdown(f"""
- Flow: **{sf['flow']:.0f}x**
- Catalyst: **{sf['catalyst']:.0f}x**
""")
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(2)
        width = 0.35
        ax.bar(x - width/2, [lab['H2_percent'], lab['conversion']], width, label='Lab', color='steelblue')
        ax.bar(x + width/2, [ind['H2_percent'], ind['conversion']], width, label='Industrial', color='darkorange')
        ax.axhline(y=30, color='green', linestyle='--', label='Target H₂')
        ax.set_ylabel('Percentage')
        ax.set_xticks(x)
        ax.set_xticklabels(['H₂ (%)', 'Conversion (%)'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        st.pyplot(fig)
        plt.close(fig)

# ============================================================================
# TAB 6: ASSUMPTIONS
# ============================================================================
with tabs[5]:
    st.header("📋 Model Assumptions")
    st.markdown(get_assumptions_text())

# ============================================================================
# AI ASSISTANT - BOTTOM PANEL (FULL WIDTH)
# ============================================================================
st.markdown("---")
st.markdown("## 🤖 AI Assistant")

# Insight box
with st.container():
    st.info(st.session_state.auto_insight if st.session_state.auto_insight else "👋 Welcome! Run a simulation to get AI insights.")

# Quick action buttons
qa_col1, qa_col2, qa_col3, qa_col4 = st.columns(4)

with qa_col1:
    if st.button("📊 Explain Results", use_container_width=True):
        context = get_context_string()
        response = get_ai_response("Explain the current simulation results in simple terms. What do they mean?", context)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

with qa_col2:
    if st.button("💡 Optimization Tips", use_container_width=True):
        context = get_context_string()
        response = get_ai_response("Give me tips to improve H2 production and reactor efficiency.", context)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()

with qa_col3:
    if st.button("📐 Key Equations", use_container_width=True):
        st.session_state.chat_history.append({"role": "assistant", "content": """**Key Equations:**

**Reaction:** CH₄ → C + 2H₂

**Rate Law:** r = k · a · η · C_CH₄ · (1-ε)

**Arrhenius:** k = A · exp(-Ea/RT)

**Deactivation:** da/dt = -kd · a

**Ergun (ΔP):** -dP/dz = 150μ(1-ε)²u/(dp²ε³) + 1.75ρ(1-ε)u²/(dpε³)

Where:
- k = rate constant
- a = catalyst activity (0-1)
- η = effectiveness factor
- ε = bed porosity
- kd = deactivation rate constant
"""})
        st.rerun()

with qa_col4:
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# Chat history display
if st.session_state.chat_history:
    st.markdown("### Chat History")
    chat_container = st.container(height=300)
    
    with chat_container:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"**👤 You:** {msg['content']}")
            else:
                st.markdown(f"**🤖 AI:** {msg['content']}")
            st.markdown("---")

# Chat input
col_input, col_send = st.columns([5, 1])
with col_input:
    user_input = st.text_input("Ask a question about the simulation...", key="chat_input", label_visibility="collapsed", placeholder="Type your question here...")
with col_send:
    if st.button("Send", type="primary", use_container_width=True):
        if user_input:
            handle_chat(user_input)
            st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("Anukaran AI © 2024 | Methane Decomposition Reactor Scale-Up | 800°C | Target H₂: 30%")
```

---

## ✅ **Changes Made**

| Issue | Fix |
|-------|-----|
| **White text** | Removed custom CSS, using Streamlit native styling |
| **AI Panel position** | Moved to **bottom** (full width) |
| **Chat area** | Larger scrollable container (300px height) |
| **Better UX** | Quick action buttons in a row, clear layout |

---

## 🖥️ **New Layout**
```
┌─────────────────────────────────────────────────────────────────┐
│  HEADER: Anukaran AI - Reactor Scale-Up                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [Tab1] [Tab2] [Tab3] [Tab4] [Tab5] [Tab6]                     │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                                                         │   │
│  │              MAIN CONTENT AREA                          │   │
│  │              (Full Width)                               │   │
│  │                                                         │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│  🤖 AI ASSISTANT (Full Width at Bottom)                        │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 💡 Auto-Insight: "Your calibration shows..."            │   │
│  └─────────────────────────────────────────────────────────┘   │
│  [Explain] [Tips] [Equations] [Clear]                          │
│                                                                 │
│  Chat History:                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ 👤 You: Can you explain the results?                    │   │
│  │ 🤖 AI: The model shows H₂ production of 24%...          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  [Type your question here...                    ] [Send]       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
