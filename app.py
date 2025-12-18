"""
╔══════════════════════════════════════════════════════════════════╗
║                         ANUKARAN AI                              ║
║          Methane Decomposition Reactor Simulator                 ║
║                                                                  ║
║   Lab Scale Calibration → Industrial Scale-Up Optimization       ║
║                     Temperature: 800°C                           ║
║                     Target H2: 30%                               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# --- LOCAL IMPORTS ---
from reactor_model import ReactorConfig as OldReactorConfig, MethaneDecompositionReactor
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
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
<style>
    .big-metric {
        font-size: 2rem;
        font-weight: bold;
        color: #1f77b4;
    }
    .success-box {
        padding: 1rem;
        background-color: #d4edda;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .warning-box {
        padding: 1rem;
        background-color: #fff3cd;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .info-box {
        padding: 1rem;
        background-color: #e7f3ff;
        border-radius: 0.5rem;
        border-left: 4px solid #0066cc;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE
# ============================================================================
if 'calibrated_A' not in st.session_state:
    st.session_state.calibrated_A = 5.0e4
if 'calibrated_Ea' not in st.session_state:
    st.session_state.calibrated_Ea = 150000.0
if 'calibrated_kd' not in st.session_state:
    st.session_state.calibrated_kd = 0.008
if 'calibration_done' not in st.session_state:
    st.session_state.calibration_done = False
if 'lab_results' not in st.session_state:
    st.session_state.lab_results = None
if 'industrial_results' not in st.session_state:
    st.session_state.industrial_results = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ============================================================================
# API KEY
# ============================================================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    GEMINI_API_KEY = ""

# ============================================================================
# HEADER
# ============================================================================
col1, col2 = st.columns([1, 5])
with col1:
    st.image("https://raw.githubusercontent.com/anukaranAI/methane-reactor/main/AnukaranNew7.png", width=120)
with col2:
    st.title("Methane Decomposition Reactor")
    st.markdown("**Lab Scale Calibration → Industrial Scale-Up** | CH₄ → C + 2H₂ | **800°C** | Target H₂: **30%**")

st.markdown("---")

# ============================================================================
# SIDEBAR - ASSUMPTIONS & PARAMETERS
# ============================================================================
with st.sidebar:
    st.header("📋 Configuration")
    
    st.markdown("### 🌡️ Operating Conditions")
    st.info("**Temperature: 800°C** (Fixed)")
    st.info("**Target H₂: 30%** (At outlet)")
    
    st.markdown("---")
    st.markdown("### 🔬 Lab Scale (Given)")
    st.markdown(f"""
    - Flow: **{LAB_CONFIG.flow_rate_mL_min} mL/min**
    - CH₄ inlet: **{LAB_CONFIG.inlet_CH4_percent}%**
    - Diameter: **{LAB_CONFIG.reactor_diameter_cm} cm**
    - Bed height: **{LAB_CONFIG.bed_height_cm} cm**
    - Catalyst: **{LAB_CONFIG.catalyst_mass_g} g**
    - Particle: **{LAB_CONFIG.particle_size_um} μm**
    """)
    
    st.markdown("---")
    st.markdown("### 🏭 Industrial Scale (Given)")
    st.markdown(f"""
    - Flow: **{INDUSTRIAL_CONFIG.flow_rate_LPM} LPM**
    - CH₄ inlet: **{INDUSTRIAL_CONFIG.inlet_CH4_percent}%**
    - Catalyst: **{INDUSTRIAL_CONFIG.catalyst_mass_kg} kg**
    - Particle: **{INDUSTRIAL_CONFIG.particle_size_um} μm**
    - L/D ratio: **{INDUSTRIAL_CONFIG.LD_ratio_min}-{INDUSTRIAL_CONFIG.LD_ratio_max}**
    """)
    
    st.markdown("---")
    st.markdown("### ⚙️ Assumed Parameters")
    st.markdown(f"""
    - Bed porosity: **{LAB_CONFIG.bed_porosity}**
    - Particle porosity: **{LAB_CONFIG.particle_porosity}**
    - Tortuosity: **{LAB_CONFIG.tortuosity}**
    - Max ΔP: **{INDUSTRIAL_CONFIG.max_pressure_drop_kPa} kPa**
    """)
    
    st.markdown("---")
    if st.session_state.calibration_done:
        st.success("✅ Kinetics Calibrated")
        st.markdown(f"""
        - A: **{st.session_state.calibrated_A:.2e}** 1/s
        - Ea: **{st.session_state.calibrated_Ea/1000:.1f}** kJ/mol
        - kd: **{st.session_state.calibrated_kd:.4f}** 1/min
        """)

# ============================================================================
# MAIN TABS
# ============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Experimental Data",
    "🔬 Lab Calibration", 
    "🏭 Industrial Scale-Up",
    "📈 Comparison",
    "📋 Assumptions"
])

# ============================================================================
# TAB 1: EXPERIMENTAL DATA
# ============================================================================
with tab1:
    st.header("📊 Experimental Data @ 800°C")
    
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
        
        st.markdown("### Key Observations")
        st.markdown(f"""
        - **Initial H₂ (TOS=0):** {exp_data['H2_percent'][0]:.2f}%
        - **Final H₂ (TOS=210):** {exp_data['H2_percent'][-1]:.2f}%
        - **H₂ Decay:** {exp_data['H2_percent'][0] - exp_data['H2_percent'][-1]:.2f}%
        - **Decay Rate:** {(1 - exp_data['H2_percent'][-1]/exp_data['H2_percent'][0])*100:.1f}%
        """)
    
    with col2:
        st.markdown("### H₂ Composition vs Time")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.plot(exp_data['TOS_min'], exp_data['H2_percent'], 'o-', 
                color='#1f77b4', markersize=10, linewidth=2, label='H₂ %')
        ax.axhline(y=30, color='green', linestyle='--', linewidth=2, label='Target (30%)')
        
        ax.set_xlabel('Time on Stream [min]', fontsize=12)
        ax.set_ylabel('H₂ Composition [%]', fontsize=12)
        ax.set_title('Experimental Data @ 800°C', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 35])
        
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("### Interpretation")
        st.info("""
        📌 **Catalyst Deactivation Observed:**
        - H₂ production decreases over time
        - Caused by carbon deposition on catalyst
        - Need to model this for accurate industrial design
        """)

# ============================================================================
# TAB 2: LAB CALIBRATION
# ============================================================================
with tab2:
    st.header("🔬 Lab Scale Kinetic Calibration")
    
    st.markdown("""
    **Objective:** Find kinetic parameters (A, Ea, kd) that match experimental data.
    
    **Method:**
    1. Calibrate A and Ea to match H₂% at TOS=0 (fresh catalyst)
    2. Calibrate kd to match the deactivation curve over time
    """)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### Step 1: Kinetic Parameters")
        
        st.markdown("#### Adjust to match TOS=0 data")
        
        A_exp = st.slider(
            "Pre-exponential (A) - log₁₀ scale",
            min_value=2.0, max_value=10.0,
            value=np.log10(st.session_state.calibrated_A),
            step=0.1,
            key="A_slider"
        )
        A_value = 10 ** A_exp
        st.markdown(f"**A = {A_value:.2e} 1/s**")
        
        Ea_value = st.slider(
            "Activation Energy (Ea) [kJ/mol]",
            min_value=80.0, max_value=250.0,
            value=st.session_state.calibrated_Ea / 1000,
            step=5.0,
            key="Ea_slider"
        )
        st.markdown(f"**Ea = {Ea_value:.0f} kJ/mol**")
        
        st.markdown("---")
        st.markdown("### Step 2: Deactivation")
        
        kd_value = st.slider(
            "Deactivation rate (kd) [1/min]",
            min_value=0.001, max_value=0.05,
            value=st.session_state.calibrated_kd,
            step=0.001,
            format="%.4f",
            key="kd_slider"
        )
        
        if st.button("🔄 Run Lab Simulation", type="primary", use_container_width=True):
            with st.spinner("Simulating lab reactor..."):
                # Create config
                config = ReactorConfig.from_lab_config(
                    LAB_CONFIG, 
                    A=A_value, 
                    Ea=Ea_value * 1000
                )
                
                # Run transient simulation
                deact = DeactivationModel('first_order', DeactivationParams(k_d=kd_value))
                reactor = TransientReactor(config, deact)
                results = reactor.solve(t_final_min=210, dt_min=1.0)
                
                st.session_state.lab_results = results
                st.session_state.calibrated_A = A_value
                st.session_state.calibrated_Ea = Ea_value * 1000
                st.session_state.calibrated_kd = kd_value
                st.session_state.calibration_done = True
            
            st.success("✅ Simulation complete!")
    
    with col2:
        st.markdown("### Results: Model vs Experimental")
        
        if st.session_state.lab_results is not None:
            results = st.session_state.lab_results
            exp_data = get_experimental_data(800)
            
            # Calculate RMSE
            model_H2_at_exp = np.interp(exp_data['TOS_min'], results['time_min'], results['H2_percent'])
            rmse = np.sqrt(np.mean((model_H2_at_exp - exp_data['H2_percent'])**2))
            
            # Metrics
            m1, m2, m3 = st.columns(3)
            m1.metric("Model H₂ (t=0)", f"{results['H2_percent'][0]:.2f}%")
            m2.metric("Exp H₂ (t=0)", f"{exp_data['H2_percent'][0]:.2f}%")
            m3.metric("RMSE", f"{rmse:.2f}%")
            
            # Plot
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            
            # H2 comparison
            axes[0].plot(results['time_min'], results['H2_percent'], 'b-', 
                        linewidth=2, label='Model')
            axes[0].scatter(exp_data['TOS_min'], exp_data['H2_percent'], 
                           c='red', s=80, zorder=5, label='Experimental')
            axes[0].axhline(y=30, color='green', linestyle='--', label='Target 30%')
            axes[0].set_xlabel('Time on Stream [min]')
            axes[0].set_ylabel('H₂ [%]')
            axes[0].set_title('H₂ Composition')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            axes[0].set_ylim([0, 40])
            
            # Activity
            axes[1].plot(results['time_min'], results['activity'], 'g-', linewidth=2)
            axes[1].set_xlabel('Time on Stream [min]')
            axes[1].set_ylabel('Activity [-]')
            axes[1].set_title('Catalyst Activity Decay')
            axes[1].grid(True, alpha=0.3)
            axes[1].set_ylim([0, 1.1])
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
            
            # Error table
            st.markdown("#### Point-by-Point Comparison")
            comp_df = pd.DataFrame({
                'TOS (min)': exp_data['TOS_min'],
                'Exp H₂ (%)': exp_data['H2_percent'],
                'Model H₂ (%)': model_H2_at_exp,
                'Error (%)': model_H2_at_exp - exp_data['H2_percent'],
            })
            st.dataframe(comp_df.style.format({
                'Exp H₂ (%)': '{:.2f}',
                'Model H₂ (%)': '{:.2f}',
                'Error (%)': '{:+.2f}',
            }), use_container_width=True, hide_index=True)
            
            if rmse < 3.0:
                st.success(f"✅ Good fit! RMSE = {rmse:.2f}% < 3%")
            elif rmse < 5.0:
                st.warning(f"⚠️ Acceptable fit. RMSE = {rmse:.2f}%")
            else:
                st.error(f"❌ Poor fit. RMSE = {rmse:.2f}%. Adjust parameters.")
        else:
            st.info("👈 Adjust parameters and click **Run Lab Simulation**")

# ============================================================================
# TAB 3: INDUSTRIAL SCALE-UP
# ============================================================================
with tab3:
    st.header("🏭 Industrial Scale-Up Optimization")
    
    if not st.session_state.calibration_done:
        st.warning("⚠️ Please calibrate kinetics in the **Lab Calibration** tab first!")
    else:
        st.success(f"✅ Using calibrated kinetics: A={st.session_state.calibrated_A:.2e}, Ea={st.session_state.calibrated_Ea/1000:.0f} kJ/mol")
        
        st.markdown("""
        **Objective:** Find optimal reactor geometry (D, L) for industrial scale.
        
        **Constraints:**
        - L/D ratio: 1-3
        - Target H₂: 30%
        - Max pressure drop: 50 kPa
        """)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Configuration")
            
            st.markdown(f"""
            **Given:**
            - Flow rate: {INDUSTRIAL_CONFIG.flow_rate_LPM} LPM
            - Catalyst mass: {INDUSTRIAL_CONFIG.catalyst_mass_kg} kg
            - Particle size: {INDUSTRIAL_CONFIG.particle_size_um} μm
            """)
            
            LD_select = st.slider(
                "Select L/D Ratio",
                min_value=1.0, max_value=3.0,
                value=2.0, step=0.1,
                key="LD_select"
            )
            
            if st.button("🚀 Optimize Geometry", type="primary", use_container_width=True):
                with st.spinner("Running industrial optimization..."):
                    scaleup = IndustrialScaleUp(
                        calibrated_A=st.session_state.calibrated_A,
                        calibrated_Ea=st.session_state.calibrated_Ea,
                        calibrated_kd=st.session_state.calibrated_kd,
                    )
                    
                    # Parametric study
                    parametric = scaleup.parametric_study(n_points=21)
                    
                    # Optimal result
                    optimal = scaleup.optimize_geometry()
                    
                    st.session_state.industrial_results = {
                        'parametric': parametric,
                        'optimal': optimal,
                    }
                
                st.success("✅ Optimization complete!")
            
            if st.button("📊 Evaluate Selected L/D", use_container_width=True):
                with st.spinner("Evaluating..."):
                    scaleup = IndustrialScaleUp(
                        calibrated_A=st.session_state.calibrated_A,
                        calibrated_Ea=st.session_state.calibrated_Ea,
                        calibrated_kd=st.session_state.calibrated_kd,
                    )
                    result = scaleup.evaluate_geometry(LD_select)
                    
                    st.markdown("### Selected Geometry Results")
                    st.markdown(f"""
                    | Parameter | Value |
                    |-----------|-------|
                    | **Diameter** | {result['diameter_cm']:.1f} cm |
                    | **Height** | {result['height_cm']:.1f} cm |
                    | **L/D Ratio** | {result['LD_ratio']:.1f} |
                    | **H₂ Outlet** | {result['H2_percent']:.2f}% |
                    | **Conversion** | {result['conversion_percent']:.2f}% |
                    | **ΔP** | {result['pressure_drop_kPa']:.2f} kPa |
                    | **Meets H₂ Target** | {'✅' if result['meets_H2_target'] else '❌'} |
                    | **Pressure OK** | {'✅' if result['meets_pressure'] else '❌'} |
                    """)
        
        with col2:
            if st.session_state.industrial_results is not None:
                results = st.session_state.industrial_results
                parametric = results['parametric']
                optimal = results['optimal']
                
                st.markdown("### 🏆 Optimal Design")
                
                o1, o2, o3 = st.columns(3)
                o1.metric("Diameter", f"{optimal.optimal_diameter_m*100:.1f} cm")
                o2.metric("Height", f"{optimal.optimal_height_m*100:.1f} cm")
                o3.metric("L/D Ratio", f"{optimal.optimal_LD_ratio:.2f}")
                
                o4, o5, o6 = st.columns(3)
                o4.metric("H₂ Outlet", f"{optimal.predicted_H2_percent:.2f}%", 
                         delta=f"{optimal.predicted_H2_percent-30:.1f}% from target")
                o5.metric("Conversion", f"{optimal.predicted_conversion:.2f}%")
                o6.metric("Pressure Drop", f"{optimal.pressure_drop_kPa:.2f} kPa")
                
                if optimal.meets_target and optimal.meets_pressure_constraint:
                    st.success("✅ Optimal design meets all constraints!")
                elif not optimal.meets_target:
                    st.warning(f"⚠️ H₂ target not met. Consider adjusting parameters.")
                elif not optimal.meets_pressure_constraint:
                    st.warning(f"⚠️ Pressure drop exceeds limit.")
                
                # Parametric study plot
                st.markdown("### Parametric Study: L/D vs Performance")
                
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                
                LD_vals = [p['LD_ratio'] for p in parametric]
                H2_vals = [p['H2_percent'] for p in parametric]
                conv_vals = [p['conversion_percent'] for p in parametric]
                dP_vals = [p['pressure_drop_kPa'] for p in parametric]
                
                # H2
                axes[0].plot(LD_vals, H2_vals, 'b-o', linewidth=2, markersize=6)
                axes[0].axhline(y=30, color='green', linestyle='--', label='Target 30%')
                axes[0].axvline(x=optimal.optimal_LD_ratio, color='red', linestyle=':', label='Optimal')
                axes[0].set_xlabel('L/D Ratio')
                axes[0].set_ylabel('H₂ Outlet [%]')
                axes[0].set_title('H₂ vs L/D')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)
                
                # Conversion
                axes[1].plot(LD_vals, conv_vals, 'g-o', linewidth=2, markersize=6)
                axes[1].axvline(x=optimal.optimal_LD_ratio, color='red', linestyle=':', label='Optimal')
                axes[1].set_xlabel('L/D Ratio')
                axes[1].set_ylabel('Conversion [%]')
                axes[1].set_title('Conversion vs L/D')
                axes[1].legend()
                axes[1].grid(True, alpha=0.3)
                
                # Pressure drop
                axes[2].plot(LD_vals, dP_vals, 'r-o', linewidth=2, markersize=6)
                axes[2].axhline(y=50, color='orange', linestyle='--', label='Max 50 kPa')
                axes[2].axvline(x=optimal.optimal_LD_ratio, color='red', linestyle=':', label='Optimal')
                axes[2].set_xlabel('L/D Ratio')
                axes[2].set_ylabel('Pressure Drop [kPa]')
                axes[2].set_title('ΔP vs L/D')
                axes[2].legend()
                axes[2].grid(True, alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Geometry table
                st.markdown("### All Geometries Evaluated")
                geom_df = pd.DataFrame(parametric)
                geom_df = geom_df[['LD_ratio', 'diameter_cm', 'height_cm', 
                                   'H2_percent', 'conversion_percent', 'pressure_drop_kPa',
                                   'meets_H2_target', 'meets_pressure']]
                geom_df.columns = ['L/D', 'D (cm)', 'L (cm)', 'H₂ (%)', 'Conv (%)', 'ΔP (kPa)', 'H₂ OK', 'ΔP OK']
                st.dataframe(geom_df.style.format({
                    'L/D': '{:.1f}',
                    'D (cm)': '{:.1f}',
                    'L (cm)': '{:.1f}',
                    'H₂ (%)': '{:.2f}',
                    'Conv (%)': '{:.2f}',
                    'ΔP (kPa)': '{:.2f}',
                }), use_container_width=True, hide_index=True)
            else:
                st.info("👈 Click **Optimize Geometry** to find optimal design")

# ============================================================================
# TAB 4: COMPARISON
# ============================================================================
with tab4:
    st.header("📈 Lab vs Industrial Comparison")
    
    if not st.session_state.calibration_done:
        st.warning("⚠️ Please complete calibration first!")
    elif st.session_state.industrial_results is None:
        st.warning("⚠️ Please run industrial optimization first!")
    else:
        scaleup = IndustrialScaleUp(
            calibrated_A=st.session_state.calibrated_A,
            calibrated_Ea=st.session_state.calibrated_Ea,
            calibrated_kd=st.session_state.calibrated_kd,
        )
        
        optimal = st.session_state.industrial_results['optimal']
        comparison = scaleup.compare_scales(industrial_LD=optimal.optimal_LD_ratio)
        
        st.markdown("### Scale-Up Summary")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("#### 🔬 Lab Scale")
            lab = comparison['lab']
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Diameter | {lab['diameter_cm']:.1f} cm |
            | Height | {lab['height_cm']:.2f} cm |
            | Flow | {lab['flow_rate']} |
            | Catalyst | {lab['catalyst_mass']} |
            | **H₂ Outlet** | **{lab['H2_percent']:.2f}%** |
            | **Conversion** | **{lab['conversion']:.2f}%** |
            | ΔP | {lab['pressure_drop_kPa']:.3f} kPa |
            """)
        
        with col2:
            st.markdown("#### 🏭 Industrial Scale")
            ind = comparison['industrial']
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Diameter | {ind['diameter_cm']:.1f} cm |
            | Height | {ind['height_cm']:.1f} cm |
            | L/D Ratio | {ind['LD_ratio']:.1f} |
            | Flow | {ind['flow_rate']} |
            | Catalyst | {ind['catalyst_mass']} |
            | **H₂ Outlet** | **{ind['H2_percent']:.2f}%** |
            | **Conversion** | **{ind['conversion']:.2f}%** |
            | ΔP | {ind['pressure_drop_kPa']:.2f} kPa |
            """)
        
        with col3:
            st.markdown("#### 📊 Scale Factors")
            sf = comparison['scale_factors']
            st.markdown(f"""
            | Parameter | Factor |
            |-----------|--------|
            | Flow Rate | {sf['flow']:.0f}x |
            | Catalyst Mass | {sf['catalyst']:.0f}x |
            | Diameter | {ind['diameter_cm']/lab['diameter_cm']:.1f}x |
            | Height | {ind['height_cm']/lab['height_cm']:.1f}x |
            """)
        
        # Visual comparison
        st.markdown("### Visual Comparison")
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart - Performance
        metrics = ['H₂ (%)', 'Conversion (%)']
        lab_vals = [lab['H2_percent'], lab['conversion']]
        ind_vals = [ind['H2_percent'], ind['conversion']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        axes[0].bar(x - width/2, lab_vals, width, label='Lab', color='steelblue')
        axes[0].bar(x + width/2, ind_vals, width, label='Industrial', color='darkorange')
        axes[0].axhline(y=30, color='green', linestyle='--', linewidth=2, label='Target H₂')
        axes[0].set_ylabel('Percentage')
        axes[0].set_title('Performance Comparison')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(metrics)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # Reactor size comparison (schematic)
        axes[1].set_xlim(0, 100)
        axes[1].set_ylim(0, 100)
        
        # Lab reactor (small, left)
        lab_scale = 5
        axes[1].add_patch(plt.Rectangle((10, 40), lab['diameter_cm']*lab_scale, 
                                         lab['height_cm']*lab_scale, 
                                         facecolor='steelblue', edgecolor='black', linewidth=2))
        axes[1].text(10 + lab['diameter_cm']*lab_scale/2, 35, 'Lab', ha='center', fontsize=10)
        
        # Industrial reactor (larger, right)
        ind_scale = 0.8
        rect_x = 50
        rect_y = 20
        axes[1].add_patch(plt.Rectangle((rect_x, rect_y), ind['diameter_cm']*ind_scale, 
                                         ind['height_cm']*ind_scale, 
                                         facecolor='darkorange', edgecolor='black', linewidth=2))
        axes[1].text(rect_x + ind['diameter_cm']*ind_scale/2, rect_y - 5, 'Industrial', ha='center', fontsize=10)
        
        axes[1].set_title('Reactor Size (Not to Scale)')
        axes[1].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Download report
        st.markdown("### 📥 Export Results")
        
        report_data = {
            'Parameter': ['Temperature', 'Target H2', 'Lab Diameter', 'Lab Height', 
                         'Industrial Diameter', 'Industrial Height', 'Industrial L/D',
                         'Lab H2%', 'Industrial H2%', 'Flow Scale Factor', 'Catalyst Scale Factor'],
            'Value': ['800°C', '30%', f"{lab['diameter_cm']:.1f} cm", f"{lab['height_cm']:.2f} cm",
                     f"{ind['diameter_cm']:.1f} cm", f"{ind['height_cm']:.1f} cm", f"{ind['LD_ratio']:.1f}",
                     f"{lab['H2_percent']:.2f}%", f"{ind['H2_percent']:.2f}%",
                     f"{sf['flow']:.0f}x", f"{sf['catalyst']:.0f}x"],
        }
        report_df = pd.DataFrame(report_data)
        
        st.download_button(
            "📥 Download Scale-Up Report",
            report_df.to_csv(index=False),
            "scaleup_report.csv",
            "text/csv"
        )

# ============================================================================
# TAB 5: ASSUMPTIONS
# ============================================================================
with tab5:
    st.header("📋 Model Assumptions & Documentation")
    
    st.markdown(get_assumptions_text())
    
    st.markdown("---")
    st.markdown("### Model Equations")
    
    with st.expander("📐 Reactor Model"):
        st.markdown("""
        **Species Balance (Plug Flow):**
        $$\\frac{dF_{CH_4}}{dz} = -r_{bed} \\cdot A_c$$
        $$\\frac{dF_{H_2}}{dz} = +2 \\cdot r_{bed} \\cdot A_c$$
        
        **Reaction Rate:**
        $$r_{bed} = k(T) \\cdot a(t) \\cdot \\eta \\cdot C_{CH_4} \\cdot (1-\\varepsilon)$$
        
        **Arrhenius Rate Constant:**
        $$k(T) = A \\cdot \\exp\\left(-\\frac{E_a}{RT}\\right)$$
        """)
    
    with st.expander("⚡ Deactivation Model"):
        st.markdown("""
        **First-Order Deactivation:**
        $$\\frac{da}{dt} = -k_d \\cdot a$$
        
        **Solution:**
        $$a(t) = \\exp(-k_d \\cdot t)$$
        
        Where:
        - $a$ = catalyst activity (0 to 1)
        - $k_d$ = deactivation rate constant [1/min]
        """)
    
    with st.expander("📏 Pressure Drop (Ergun Equation)"):
        st.markdown("""
        $$-\\frac{dP}{dz} = \\frac{150\\mu(1-\\varepsilon)^2}{d_p^2\\varepsilon^3}u + \\frac{1.75\\rho(1-\\varepsilon)}{d_p\\varepsilon^3}u^2$$
        
        Where:
        - $\\mu$ = gas viscosity
        - $\\varepsilon$ = bed porosity
        - $d_p$ = particle diameter
        - $u$ = superficial velocity
        - $\\rho$ = gas density
        """)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Anukaran AI © 2024 | Methane Decomposition Reactor Scale-Up Tool</p>
    <p>Temperature: 800°C | Target H₂: 30% | Lab → Industrial</p>
</div>
""", unsafe_allow_html=True)
