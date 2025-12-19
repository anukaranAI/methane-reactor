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
import time

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
    calibrate_kinetics, fit_deactivation,
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
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 0.9rem;
        color: #666;
        margin-top: 0;
    }
    .ai-message {
        background-color: #f0f7ff;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
        border-left: 3px solid #1f77b4;
    }
    .user-message {
        background-color: #e8f5e9;
        border-radius: 10px;
        padding: 10px 15px;
        margin: 5px 0;
        border-left: 3px solid #4caf50;
    }
    .insight-box {
        background-color: #fff8e1;
        border-radius: 8px;
        padding: 12px;
        margin: 10px 0;
        border-left: 4px solid #ffc107;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        text-align: center;
        border: 1px solid #dee2e6;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
    }
</style>
""", unsafe_allow_html=True)

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
    'current_tab': 'Experimental Data',
    'auto_insight': '',
    'last_action': '',
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
        return f"""📊 **Experimental Data Loaded (800°C)**

- Initial H₂: **{exp_data['H2_percent'][0]:.2f}%**
- Final H₂: **{exp_data['H2_percent'][-1]:.2f}%** (after 210 min)
- Decay: **{((exp_data['H2_percent'][0] - exp_data['H2_percent'][-1])/exp_data['H2_percent'][0]*100):.1f}%**

💡 The catalyst shows significant deactivation over time. This is typical for methane decomposition due to carbon deposition."""

    elif action == "lab_calibration" and results:
        return f"""🔬 **Lab Calibration Results**

- Model H₂ at t=0: **{results.get('H2_t0', 0):.2f}%**
- Experimental H₂: **{results.get('exp_H2_t0', 24.13):.2f}%**
- RMSE: **{results.get('rmse', 0):.2f}%**

💡 {'Good fit! Ready for scale-up.' if results.get('rmse', 100) < 5 else 'Try adjusting A or Ea for better fit.'}"""

    elif action == "industrial_optimization" and results:
        return f"""🏭 **Industrial Optimization Complete**

- Optimal Diameter: **{results.get('diameter', 0):.1f} cm**
- Optimal Height: **{results.get('height', 0):.1f} cm**
- L/D Ratio: **{results.get('LD', 0):.2f}**
- Predicted H₂: **{results.get('H2', 0):.2f}%**
- Pressure Drop: **{results.get('dP', 0):.2f} kPa**

💡 {'✅ Target H₂ of 30% achieved!' if results.get('H2', 0) >= 30 else '⚠️ H₂ below target. Consider adjusting parameters.'}"""

    elif action == "transient_lab" and results:
        return f"""⏱️ **Lab Transient Analysis**

- Initial H₂: **{results.get('H2_initial', 0):.2f}%**
- Final H₂: **{results.get('H2_final', 0):.2f}%**
- Time to 50% activity: **{results.get('half_life', 0):.0f} min**

💡 Catalyst deactivation follows first-order kinetics. Plan regeneration accordingly."""

    elif action == "transient_industrial" and results:
        time_to_target = results.get('time_below_target', 'N/A')
        return f"""⏱️ **Industrial Transient Analysis**

- Initial H₂: **{results.get('H2_initial', 0):.2f}%**
- Time above 30% H₂: **{time_to_target}**
- Recommended cycle: **{results.get('recommended_cycle', 'N/A')}**

💡 Schedule catalyst regeneration before H₂ drops below target."""

    return "👋 Welcome! Select a tab to begin analysis."


def get_context_string() -> str:
    """Build context string for AI from current state"""
    context_parts = []
    
    context_parts.append(f"Temperature: 800°C, Target H2: 30%")
    context_parts.append(f"Lab: D={LAB_CONFIG.reactor_diameter_cm}cm, L={LAB_CONFIG.bed_height_cm}cm, Flow={LAB_CONFIG.flow_rate_mL_min}mL/min")
    context_parts.append(f"Industrial: Flow={INDUSTRIAL_CONFIG.flow_rate_LPM}LPM, Catalyst={INDUSTRIAL_CONFIG.catalyst_mass_kg}kg")
    
    if st.session_state.calibration_done:
        context_parts.append(f"Calibrated: A={st.session_state.calibrated_A:.2e}, Ea={st.session_state.calibrated_Ea/1000:.0f}kJ/mol, kd={st.session_state.calibrated_kd:.4f}")
    
    if st.session_state.lab_results:
        context_parts.append(f"Lab H2: {st.session_state.lab_results['H2_percent'][0]:.2f}%")
    
    if st.session_state.industrial_results:
        opt = st.session_state.industrial_results.get('optimal')
        if opt:
            context_parts.append(f"Industrial optimal: D={opt.optimal_diameter_m*100:.1f}cm, L={opt.optimal_height_m*100:.1f}cm, H2={opt.predicted_H2_percent:.2f}%")
    
    return "\n".join(context_parts)


def handle_chat(user_input: str):
    """Handle chat input"""
    if user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        context = get_context_string()
        response = get_ai_response(user_input, context)
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})


# ============================================================================
# MAIN LAYOUT: LEFT (Main Content) | RIGHT (AI Panel)
# ============================================================================

# Header
st.markdown("""
<div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
    <img src="https://raw.githubusercontent.com/anukaranAI/methane-reactor/main/AnukaranNew7.png" width="80">
    <div>
        <p class="main-header">Methane Decomposition Reactor Scale-Up</p>
        <p class="sub-header">CH₄ → C + 2H₂ | 800°C | Target H₂: 30% | Lab → Industrial</p>
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Main columns: Content (70%) | AI Panel (30%)
main_col, ai_col = st.columns([7, 3])

# ============================================================================
# RIGHT COLUMN: AI ASSISTANT PANEL
# ============================================================================
with ai_col:
    st.markdown("### 🤖 AI Assistant")
    
    # Auto-insight box
    st.markdown('<div class="insight-box">', unsafe_allow_html=True)
    if st.session_state.auto_insight:
        st.markdown(st.session_state.auto_insight)
    else:
        st.markdown("👋 **Welcome!** Run a simulation to get AI insights.")
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Quick action buttons
    st.markdown("**Quick Actions:**")
    qa1, qa2 = st.columns(2)
    
    with qa1:
        if st.button("📊 Explain Results", use_container_width=True, key="qa_explain"):
            context = get_context_string()
            response = get_ai_response("Explain the current simulation results in simple terms.", context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    with qa2:
        if st.button("💡 Optimization Tips", use_container_width=True, key="qa_tips"):
            context = get_context_string()
            response = get_ai_response("Give me tips to optimize the reactor for maximum H2 production.", context)
            st.session_state.chat_history.append({"role": "assistant", "content": response})
    
    qa3, qa4 = st.columns(2)
    
    with qa3:
        if st.button("📐 Show Equations", use_container_width=True, key="qa_equations"):
            st.session_state.chat_history.append({"role": "assistant", "content": """**Key Equations:**

**Reaction:** CH₄ → C + 2H₂

**Rate Law:**
r = k·a·η·C_CH₄·(1-ε)

**Arrhenius:**
k = A·exp(-Ea/RT)

**Deactivation:**
da/dt = -kd·a

**Ergun (ΔP):**
-dP/dz = 150μ(1-ε)²u/(dp²ε³) + 1.75ρ(1-ε)u²/(dpε³)"""})
    
    with qa4:
        if st.button("🔄 Clear Chat", use_container_width=True, key="qa_clear"):
            st.session_state.chat_history = []
            st.rerun()
    
    st.markdown("---")
    
    # Chat history
    st.markdown("**Chat History:**")
    chat_container = st.container(height=300)
    
    with chat_container:
        for msg in st.session_state.chat_history[-10:]:  # Show last 10 messages
            if msg["role"] == "user":
                st.markdown(f'<div class="user-message">👤 {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="ai-message">🤖 {msg["content"]}</div>', unsafe_allow_html=True)
    
    # Chat input
    st.markdown("---")
    chat_input = st.text_input("💬 Ask a question...", key="chat_input", placeholder="Type your question here...")
    
    if st.button("Send", use_container_width=True, key="send_chat"):
        if chat_input:
            handle_chat(chat_input)
            st.rerun()

# ============================================================================
# LEFT COLUMN: MAIN CONTENT TABS
# ============================================================================
with main_col:
    tabs = st.tabs([
        "📊 Experimental Data",
        "🔬 Lab Calibration",
        "⏱️ Transient Analysis",
        "🏭 Industrial Scale-Up",
        "📈 Comparison",
        "📋 Assumptions"
    ])
    
    # ========================================================================
    # TAB 1: EXPERIMENTAL DATA
    # ========================================================================
    with tabs[0]:
        st.header("📊 Experimental Data @ 800°C")
        
        # Auto-update insight
        if st.session_state.last_action != "experimental_data":
            st.session_state.auto_insight = generate_auto_insight("experimental_data")
            st.session_state.last_action = "experimental_data"
        
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
            
            # Key metrics
            st.markdown("### Key Metrics")
            m1, m2 = st.columns(2)
            m1.metric("Initial H₂", f"{exp_data['H2_percent'][0]:.2f}%")
            m2.metric("Final H₂", f"{exp_data['H2_percent'][-1]:.2f}%")
            
            m3, m4 = st.columns(2)
            decay = (exp_data['H2_percent'][0] - exp_data['H2_percent'][-1]) / exp_data['H2_percent'][0] * 100
            m3.metric("Decay", f"{decay:.1f}%")
            m4.metric("Target", "30%", delta=f"{exp_data['H2_percent'][0]-30:+.1f}%")
        
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
            ax.set_xlim([0, 220])
            
            st.pyplot(fig)
            plt.close(fig)

    # ========================================================================
    # TAB 2: LAB CALIBRATION
    # ========================================================================
    with tabs[1]:
        st.header("🔬 Lab Scale Kinetic Calibration")
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            st.markdown("### Kinetic Parameters")
            
            st.markdown("**Adjust to match experimental data:**")
            
            A_exp = st.slider(
                "Pre-exponential (A) - log₁₀",
                min_value=3.0, max_value=8.0,
                value=np.log10(st.session_state.calibrated_A),
                step=0.1,
                key="A_slider_v2"
            )
            A_value = 10 ** A_exp
            st.markdown(f"**A = {A_value:.2e} 1/s**")
            
            Ea_value = st.slider(
                "Activation Energy [kJ/mol]",
                min_value=100.0, max_value=250.0,
                value=st.session_state.calibrated_Ea / 1000,
                step=5.0,
                key="Ea_slider_v2"
            )
            
            kd_value = st.slider(
                "Deactivation rate kd [1/min]",
                min_value=0.001, max_value=0.05,
                value=st.session_state.calibrated_kd,
                step=0.001,
                format="%.4f",
                key="kd_slider_v2"
            )
            
            st.markdown("---")
            
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
                    
                    # Calculate RMSE
                    exp_data = get_experimental_data(800)
                    model_H2 = np.interp(exp_data['TOS_min'], results['time_min'], results['H2_percent'])
                    rmse = np.sqrt(np.mean((model_H2 - exp_data['H2_percent'])**2))
                    
                    # Update auto-insight
                    st.session_state.auto_insight = generate_auto_insight("lab_calibration", {
                        'H2_t0': results['H2_percent'][0],
                        'exp_H2_t0': exp_data['H2_percent'][0],
                        'rmse': rmse
                    })
                    st.session_state.last_action = "lab_calibration"
                
                st.success("✅ Calibration complete!")
                st.rerun()
        
        with col2:
            st.markdown("### Model vs Experimental")
            
            if st.session_state.lab_results is not None:
                results = st.session_state.lab_results
                exp_data = get_experimental_data(800)
                
                model_H2 = np.interp(exp_data['TOS_min'], results['time_min'], results['H2_percent'])
                rmse = np.sqrt(np.mean((model_H2 - exp_data['H2_percent'])**2))
                
                # Metrics
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Model H₂(t=0)", f"{results['H2_percent'][0]:.2f}%")
                mc2.metric("Exp H₂(t=0)", f"{exp_data['H2_percent'][0]:.2f}%")
                mc3.metric("RMSE", f"{rmse:.2f}%", delta="Good" if rmse < 5 else "Adjust")
                
                # Plot
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
                
                # Comparison table
                with st.expander("📋 Point-by-Point Comparison"):
                    comp_df = pd.DataFrame({
                        'TOS': exp_data['TOS_min'],
                        'Exp H₂': exp_data['H2_percent'],
                        'Model H₂': model_H2,
                        'Error': model_H2 - exp_data['H2_percent'],
                    })
                    st.dataframe(comp_df.style.format({
                        'Exp H₂': '{:.2f}', 'Model H₂': '{:.2f}', 'Error': '{:+.2f}'
                    }), use_container_width=True, hide_index=True)
            else:
                st.info("👈 Adjust parameters and click **Run Lab Simulation**")

    # ========================================================================
    # TAB 3: TRANSIENT ANALYSIS
    # ========================================================================
    with tabs[2]:
        st.header("⏱️ Transient Analysis")
        
        if not st.session_state.calibration_done:
            st.warning("⚠️ Please calibrate kinetics in **Lab Calibration** tab first!")
        else:
            subtab1, subtab2 = st.tabs(["🔬 Lab Scale Transient", "🏭 Industrial Transient"])
            
            # --- LAB TRANSIENT ---
            with subtab1:
                st.markdown("### Lab Scale Time-Dependent Analysis")
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**Simulation Settings:**")
                    
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
                            
                            # Find half-life
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
                        
                        # H2 over time
                        axes[0].plot(results['time_min'], results['H2_percent'], 'b-', linewidth=2)
                        axes[0].axhline(y=30, color='green', linestyle='--', label='Target')
                        axes[0].set_xlabel('Time [min]')
                        axes[0].set_ylabel('H₂ [%]')
                        axes[0].set_title('H₂ Production over Time')
                        axes[0].legend()
                        axes[0].grid(True, alpha=0.3)
                        
                        # Activity decay
                        axes[1].plot(results['time_min'], results['activity'], 'r-', linewidth=2)
                        axes[1].axhline(y=0.5, color='orange', linestyle='--', label='50% Activity')
                        axes[1].set_xlabel('Time [min]')
                        axes[1].set_ylabel('Catalyst Activity')
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
            
            # --- INDUSTRIAL TRANSIENT ---
            with subtab2:
                st.markdown("### Industrial Scale Time-Dependent Analysis")
                
                if st.session_state.industrial_results is None:
                    st.warning("⚠️ Run **Industrial Scale-Up** optimization first!")
                else:
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        opt = st.session_state.industrial_results['optimal']
                        
                        st.markdown(f"""**Reactor Configuration:**
- Diameter: {opt.optimal_diameter_m*100:.1f} cm
- Height: {opt.optimal_height_m*100:.1f} cm
- L/D: {opt.optimal_LD_ratio:.2f}
""")
                        
                        t_final_ind = st.slider("Simulation Time [min]", 60, 1000, 300, key="t_ind")
                        
                        if st.button("▶️ Run Industrial Transient", type="primary", use_container_width=True):
                            with st.spinner("Running..."):
                                scaleup = IndustrialScaleUp(
                                    calibrated_A=st.session_state.calibrated_A,
                                    calibrated_Ea=st.session_state.calibrated_Ea,
                                    calibrated_kd=st.session_state.calibrated_kd,
                                )
                                
                                results = scaleup.transient_industrial(
                                    LD_ratio=opt.optimal_LD_ratio,
                                    t_final_min=t_final_ind
                                )
                                
                                st.session_state.industrial_transient_results = results
                                
                                # Find time below target
                                below_target = results['H2_percent'] < 30
                                if np.any(below_target):
                                    time_below = results['time_min'][np.argmax(below_target)]
                                    time_str = f"{time_below:.0f} min"
                                else:
                                    time_str = f"> {t_final_ind} min"
                                
                                st.session_state.auto_insight = generate_auto_insight("transient_industrial", {
                                    'H2_initial': results['H2_percent'][0],
                                    'time_below_target': time_str,
                                    'recommended_cycle': f"{int(time_below*0.8)} min" if np.any(below_target) else "N/A"
                                })
                            
                            st.success("✅ Complete!")
                            st.rerun()
                    
                    with col2:
                        if st.session_state.industrial_transient_results:
                            results = st.session_state.industrial_transient_results
                            
                            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                            
                            # H2 over time with target zone
                            axes[0].plot(results['time_min'], results['H2_percent'], 'b-', linewidth=2)
                            axes[0].axhline(y=30, color='green', linestyle='--', linewidth=2, label='Target 30%')
                            axes[0].fill_between(results['time_min'], 30, results['H2_percent'], 
                                                where=results['H2_percent']>=30, alpha=0.3, color='green', label='Above Target')
                            axes[0].fill_between(results['time_min'], results['H2_percent'], 30,
                                                where=results['H2_percent']<30, alpha=0.3, color='red', label='Below Target')
                            axes[0].set_xlabel('Time [min]')
                            axes[0].set_ylabel('H₂ [%]')
                            axes[0].set_title('Industrial H₂ Production')
                            axes[0].legend()
                            axes[0].grid(True, alpha=0.3)
                            
                            # Activity
                            axes[1].plot(results['time_min'], results['activity'], 'r-', linewidth=2)
                            axes[1].set_xlabel('Time [min]')
                            axes[1].set_ylabel('Activity')
                            axes[1].set_title('Catalyst Activity')
                            axes[1].set_ylim([0, 1.1])
                            axes[1].grid(True, alpha=0.3)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                            plt.close(fig)
                            
                            # Time to regeneration
                            below_target = results['H2_percent'] < 30
                            if np.any(below_target):
                                regen_time = results['time_min'][np.argmax(below_target)]
                                st.warning(f"⚠️ **Regeneration needed at {regen_time:.0f} min** (H₂ drops below 30%)")
                            else:
                                st.success(f"✅ H₂ stays above 30% for entire {results['time_min'][-1]:.0f} min simulation")

    # ========================================================================
    # TAB 4: INDUSTRIAL SCALE-UP
    # ========================================================================
    with tabs[3]:
        st.header("🏭 Industrial Scale-Up Optimization")
        
        if not st.session_state.calibration_done:
            st.warning("⚠️ Please calibrate kinetics in **Lab Calibration** tab first!")
        else:
            st.success(f"✅ Using: A={st.session_state.calibrated_A:.2e}, Ea={st.session_state.calibrated_Ea/1000:.0f} kJ/mol, kd={st.session_state.calibrated_kd:.4f}")
            
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.markdown("### Industrial Specs")
                st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Flow Rate | {INDUSTRIAL_CONFIG.flow_rate_LPM} LPM |
| Catalyst | {INDUSTRIAL_CONFIG.catalyst_mass_kg} kg |
| Particle Size | {INDUSTRIAL_CONFIG.particle_size_um} μm |
| L/D Range | {INDUSTRIAL_CONFIG.LD_ratio_min} - {INDUSTRIAL_CONFIG.LD_ratio_max} |
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
                        st.session_state.last_action = "industrial_optimization"
                    
                    st.success("✅ Optimization complete!")
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
                             delta=f"{opt.predicted_H2_percent-30:.1f}% vs target")
                    o5.metric("Conversion", f"{opt.predicted_conversion:.2f}%")
                    o6.metric("ΔP", f"{opt.pressure_drop_kPa:.2f} kPa")
                    
                    # Status
                    if opt.meets_target and opt.meets_pressure_constraint:
                        st.success("✅ All constraints satisfied!")
                    elif not opt.meets_target:
                        st.warning("⚠️ H₂ below target. Adjust kinetics or constraints.")
                    elif not opt.meets_pressure_constraint:
                        st.warning("⚠️ Pressure drop too high.")
                    
                    # Parametric plot
                    st.markdown("### Parametric Study")
                    
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
                else:
                    st.info("👈 Click **Optimize Reactor** to find optimal design")

    # ========================================================================
    # TAB 5: COMPARISON
    # ========================================================================
    with tabs[4]:
        st.header("📈 Lab vs Industrial Comparison")
        
        if not st.session_state.calibration_done:
            st.warning("⚠️ Complete calibration first!")
        elif st.session_state.industrial_results is None:
            st.warning("⚠️ Run industrial optimization first!")
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
                st.markdown("### 🔬 Lab Scale")
                lab = comparison['lab']
                st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Diameter | {lab['diameter_cm']:.1f} cm |
| Height | {lab['height_cm']:.2f} cm |
| Flow | {lab['flow_rate']} |
| Catalyst | {lab['catalyst_mass']} |
| **H₂** | **{lab['H2_percent']:.2f}%** |
| **Conv** | **{lab['conversion']:.2f}%** |
| ΔP | {lab['pressure_drop_kPa']:.3f} kPa |
""")
            
            with col2:
                st.markdown("### 🏭 Industrial")
                ind = comparison['industrial']
                st.markdown(f"""
| Parameter | Value |
|-----------|-------|
| Diameter | {ind['diameter_cm']:.1f} cm |
| Height | {ind['height_cm']:.1f} cm |
| L/D | {ind['LD_ratio']:.1f} |
| Flow | {ind['flow_rate']} |
| Catalyst | {ind['catalyst_mass']} |
| **H₂** | **{ind['H2_percent']:.2f}%** |
| **Conv** | **{ind['conversion']:.2f}%** |
| ΔP | {ind['pressure_drop_kPa']:.2f} kPa |
""")
            
            with col3:
                st.markdown("### 📊 Scale Factors")
                sf = comparison['scale_factors']
                st.markdown(f"""
| Parameter | Factor |
|-----------|--------|
| Flow | **{sf['flow']:.0f}x** |
| Catalyst | **{sf['catalyst']:.0f}x** |
| Diameter | {ind['diameter_cm']/lab['diameter_cm']:.1f}x |
| Height | {ind['height_cm']/lab['height_cm']:.1f}x |
""")
            
            # Visual comparison
            st.markdown("---")
            
            fig, ax = plt.subplots(figsize=(10, 5))
            
            metrics = ['H₂ (%)', 'Conversion (%)']
            lab_vals = [lab['H2_percent'], lab['conversion']]
            ind_vals = [ind['H2_percent'], ind['conversion']]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax.bar(x - width/2, lab_vals, width, label='Lab', color='steelblue')
            ax.bar(x + width/2, ind_vals, width, label='Industrial', color='darkorange')
            ax.axhline(y=30, color='green', linestyle='--', linewidth=2, label='Target H₂')
            
            ax.set_ylabel('Percentage')
            ax.set_title('Performance Comparison', fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(metrics)
            ax.legend()
            ax.grid(True, alpha=0.3, axis='y')
            
            st.pyplot(fig)
            plt.close(fig)
            
            # Download
            st.download_button(
                "📥 Download Report",
                pd.DataFrame({
                    'Parameter': ['Lab D', 'Lab L', 'Lab H2', 'Ind D', 'Ind L', 'Ind H2', 'Scale Factor'],
                    'Value': [f"{lab['diameter_cm']:.1f}cm", f"{lab['height_cm']:.2f}cm", f"{lab['H2_percent']:.2f}%",
                             f"{ind['diameter_cm']:.1f}cm", f"{ind['height_cm']:.1f}cm", f"{ind['H2_percent']:.2f}%",
                             f"{sf['flow']:.0f}x"]
                }).to_csv(index=False),
                "scaleup_report.csv",
                "text/csv"
            )

    # ========================================================================
    # TAB 6: ASSUMPTIONS
    # ========================================================================
    with tabs[5]:
        st.header("📋 Model Assumptions")
        st.markdown(get_assumptions_text())

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9rem;'>
    Anukaran AI © 2024 | Methane Decomposition Reactor Scale-Up | 800°C | Target H₂: 30%
</div>
""", unsafe_allow_html=True)
