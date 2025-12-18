"""
╔══════════════════════════════════════════════════════════════════╗
║                         ANUKARAN AI                              ║
║          Methane Decomposition Reactor Simulator                 ║
║         Steady-State | Transient | AI-Powered Optimization       ║
╚══════════════════════════════════════════════════════════════════╝
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time

# --- LOCAL IMPORTS ---
from reactor_model import ReactorConfig, MethaneDecompositionReactor, MW_C, MW_H2
from ai_assistant import GeminiAssistant
from core.templates import OPTIMIZATION_TEMPLATES, get_template, get_template_names
from core.optimizer import (
    OptimizationConfig,
    BayesianOptimizer,
    create_objective_function,
    get_base_config_from_session
)
from core.transient_model import (
    DeactivationParams,
    DeactivationModel,
    TransientReactor,
    create_transient_config,
    fit_deactivation_parameter,
    run_validation_at_temperature
)
from core.experimental_data import (
    EXPERIMENTAL_DATA,
    get_experimental_data,
    get_available_temperatures,
    get_all_initial_values
)
from utils.plotting import (
    create_convergence_plot,
    create_trials_table_data,
    create_summary_stats
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Anukaran AI - Reactor Simulator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# API KEY
# ============================================================================
try:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except (FileNotFoundError, KeyError):
    GEMINI_API_KEY = ""

# ============================================================================
# SESSION STATE
# ============================================================================
if 'simulation_data' not in st.session_state:
    st.session_state.simulation_data = None
if 'config_data' not in st.session_state:
    st.session_state.config_data = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'optimization_result' not in st.session_state:
    st.session_state.optimization_result = None
if 'validation_results' not in st.session_state:
    st.session_state.validation_results = {}
if 'fitted_kd' not in st.session_state:
    st.session_state.fitted_kd = {770: 0.01, 800: 0.015, 830: 0.02}

# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def run_simulation():
    """Run single simulation with current parameters"""
    try:
        total_y = st.session_state.y_ch4 + st.session_state.y_h2 + st.session_state.y_n2
        if total_y == 0:
            total_y = 1.0
        
        config = ReactorConfig(
            diameter=st.session_state.d_reac / 100,
            bed_height=st.session_state.h_bed / 100,
            particle_diameter=st.session_state.d_part * 1e-6,
            catalyst_density=st.session_state.rho_cat,
            particle_porosity=st.session_state.eps_part,
            tortuosity=st.session_state.tau,
            bed_porosity=st.session_state.eps_bed,
            catalyst_mass=st.session_state.mass_cat / 1000,
            inlet_temperature=st.session_state.t_in + 273.15,
            inlet_pressure=st.session_state.p_in * 1e5,
            flow_rate=st.session_state.flow / 60 / 1e6,
            y_CH4_in=st.session_state.y_ch4 / total_y,
            y_H2_in=st.session_state.y_h2 / total_y,
            y_N2_in=st.session_state.y_n2 / total_y,
            pre_exponential=st.session_state.pre_exp,
            activation_energy=st.session_state.act_e * 1000,
            beta=st.session_state.beta,
            heat_of_reaction=st.session_state.dh * 1e6
        )
        
        reactor = MethaneDecompositionReactor(config, isothermal=st.session_state.iso_check)
        st.session_state.simulation_data = reactor.solve()
        st.session_state.config_data = config
        
    except Exception as e:
        st.error(f"Simulation Failed: {e}")


def handle_ai_request(prompt_text):
    """Handle AI chat request"""
    st.session_state.chat_history.append({"role": "user", "content": prompt_text})
    
    context_str = "No simulation run yet."
    if st.session_state.simulation_data:
        r = st.session_state.simulation_data
        cfg = st.session_state.config_data
        context_str = (
            f"Inlet: T={cfg.inlet_temperature-273.15:.0f}°C, P={cfg.inlet_pressure/1e5:.1f}bar. "
            f"Results: Conversion={r['X_CH4'][-1]*100:.2f}%, "
            f"H2 Yield={r['V_dot_H2_Nm3_h'][-1]:.4f} Nm³/h."
        )

    if GEMINI_API_KEY:
        ai = GeminiAssistant(GEMINI_API_KEY)
        response = ai.generate_response(prompt_text, context_str)
    else:
        response = "⚠️ AI not configured. Add GEMINI_API_KEY to secrets."
    
    st.session_state.chat_history.append({"role": "assistant", "content": response})


# ============================================================================
# HEADER
# ============================================================================
col_logo, col_title = st.columns([1, 4])
with col_logo:
    st.image("https://raw.githubusercontent.com/anukaranAI/methane-reactor/main/AnukaranNew7.png", width=150)
with col_title:
    st.title("Methane Decomposition Reactor Simulator")
    st.caption("Steady-State | Transient with Deactivation | AI-Powered Optimization")

st.markdown("---")

# ============================================================================
# SIDEBAR
# ============================================================================
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
    
    st.button("▶️ Run Simulation", type="primary", use_container_width=True, on_click=run_simulation)

# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================
tab_sim, tab_transient, tab_opt, tab_chat = st.tabs([
    "📊 Steady-State", 
    "⏱️ Transient/Validation", 
    "🎯 Optimizer", 
    "🤖 AI Assistant"
])

# ============================================================================
# TAB 1: STEADY-STATE SIMULATION
# ============================================================================
with tab_sim:
    if st.session_state.simulation_data:
        r = st.session_state.simulation_data
        cfg = st.session_state.config_data
        
        # Metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Conversion", f"{r['X_CH4'][-1]*100:.2f} %")
        m2.metric("H₂ Generation", f"{r['V_dot_H2_Nm3_h'][-1]:.4f} Nm³/h")
        m3.metric("H₂ Mass Flow", f"{r['m_dot_H2_kg_s'][-1]*3600:.4f} kg/h")
        m4.metric("Pressure Drop", f"{(cfg.inlet_pressure - r['P'][-1])/1000:.2f} kPa")
        
        # Plots
        plot_tabs = st.tabs(["Conversion", "Flow Rates", "Composition", "Temperature", "Pressure", "H₂ Yield"])
        z_cm = r['z'] * 100
        
        with plot_tabs[0]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(z_cm, r['X_CH4']*100, 'b-', lw=2)
            ax.set_xlabel("Axial Position [cm]")
            ax.set_ylabel("CH₄ Conversion [%]")
            ax.set_title("Conversion Profile")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        with plot_tabs[1]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(z_cm, r['F_CH4']*1000, 'r-', lw=2, label='CH₄')
            ax.plot(z_cm, r['F_H2']*1000, 'g-', lw=2, label='H₂')
            ax.set_xlabel("Axial Position [cm]")
            ax.set_ylabel("Molar Flow [mmol/s]")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        with plot_tabs[2]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(z_cm, r['y_CH4']*100, 'r-', lw=2, label='CH₄')
            ax.plot(z_cm, r['y_H2']*100, 'g-', lw=2, label='H₂')
            ax.plot(z_cm, r['y_N2']*100, 'b--', lw=2, label='N₂')
            ax.set_xlabel("Axial Position [cm]")
            ax.set_ylabel("Mole Fraction [%]")
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        with plot_tabs[3]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(z_cm, r['T']-273.15, 'orange', lw=2)
            ax.set_xlabel("Axial Position [cm]")
            ax.set_ylabel("Temperature [°C]")
            ax.set_title("Temperature Profile")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        with plot_tabs[4]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(z_cm, r['P']/1e5, 'purple', lw=2)
            ax.set_xlabel("Axial Position [cm]")
            ax.set_ylabel("Pressure [bar]")
            ax.set_title("Pressure Profile")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        with plot_tabs[5]:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(z_cm, r['V_dot_H2_Nm3_h'], 'green', lw=2)
            ax.set_xlabel("Axial Position [cm]")
            ax.set_ylabel("H₂ Production [Nm³/h]")
            ax.set_title("Cumulative H₂ Production")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            plt.close(fig)
        
        # Download
        df = pd.DataFrame({
            'z_cm': z_cm,
            'Conversion_%': r['X_CH4']*100,
            'T_C': r['T']-273.15,
            'P_bar': r['P']/1e5,
            'H2_Nm3h': r['V_dot_H2_Nm3_h']
        })
        st.download_button("💾 Download Results CSV", df.to_csv(index=False), "simulation_results.csv", "text/csv")
    else:
        st.info("👈 Configure parameters in sidebar and click **Run Simulation**")

# ============================================================================
# TAB 2: TRANSIENT / VALIDATION
# ============================================================================
with tab_transient:
    st.subheader("⏱️ Transient Simulation with Catalyst Deactivation")
    st.markdown("Compare model predictions with experimental data over time.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Experimental Conditions")
        
        # Temperature selection
        available_temps = get_available_temperatures()
        selected_temp = st.selectbox(
            "Select Temperature",
            options=available_temps,
            format_func=lambda x: f"{x} °C",
            key="val_temp"
        )
        
        st.markdown("#### Deactivation Parameters")
        
        deact_model_type = st.selectbox(
            "Deactivation Model",
            options=['first_order', 'second_order', 'coking'],
            format_func=lambda x: {
                'first_order': '1st Order: da/dt = -kd·a',
                'second_order': '2nd Order: da/dt = -kd·a²',
                'coking': 'Coking: da/dt = -kd·a·C_CH4'
            }[x],
            key="deact_model"
        )
        
        k_d_value = st.number_input(
            "Deactivation Rate (kd) [1/min]",
            min_value=0.001,
            max_value=0.5,
            value=st.session_state.fitted_kd.get(selected_temp, 0.01),
            format="%.4f",
            key="k_d_input"
        )
        
        t_final = st.slider("Simulation Time (min)", 30, 300, 210, key="t_final")
        
        # Buttons
        col_btn1, col_btn2 = st.columns(2)
        
        run_transient = col_btn1.button("▶️ Run Transient", type="primary", use_container_width=True)
        fit_kd = col_btn2.button("🔧 Auto-Fit kd", use_container_width=True)
    
    with col2:
        # Show experimental data table
        st.markdown("#### Experimental Data")
        exp_data = get_experimental_data(selected_temp)
        if exp_data:
            exp_df = pd.DataFrame({
                'TOS (min)': exp_data['TOS_min'],
                'H2 (%)': exp_data['H2_percent'],
                'CH4 (%)': exp_data['CH4_percent']
            })
            st.dataframe(exp_df, use_container_width=True, hide_index=True)
    
    # Auto-fit kd
    if fit_kd:
        with st.spinner(f"Fitting deactivation parameter for {selected_temp}°C..."):
            config = create_transient_config(temperature_C=selected_temp)
            exp_data = get_experimental_data(selected_temp)
            
            best_kd, best_rmse = fit_deactivation_parameter(
                exp_data['H2_percent'],
                exp_data['TOS_min'],
                config,
                k_d_range=(0.001, 0.1),
                n_trials=30
            )
            
            st.session_state.fitted_kd[selected_temp] = best_kd
            st.success(f"✅ Best kd = {best_kd:.4f} (RMSE = {best_rmse:.2f}%)")
            st.rerun()
    
    # Run transient simulation
    if run_transient:
        with st.spinner("Running transient simulation..."):
            # Create config
            config = create_transient_config(temperature_C=selected_temp)
            
            # Create deactivation model
            deact_params = DeactivationParams(k_d=k_d_value)
            deact_model = DeactivationModel(deact_model_type, deact_params)
            
            # Run simulation
            reactor = TransientReactor(config, deact_model)
            results = reactor.solve(t_final_min=t_final, dt_min=1.0)
            
            # Store results
            st.session_state.validation_results[selected_temp] = {
                'model': results,
                'k_d': k_d_value
            }
        
        st.success("✅ Transient simulation complete!")
    
    # Display results
    if selected_temp in st.session_state.validation_results:
        st.markdown("---")
        st.markdown("### 📊 Results Comparison")
        
        model_results = st.session_state.validation_results[selected_temp]['model']
        exp_data = get_experimental_data(selected_temp)
        
        # Calculate metrics
        model_H2_at_exp = np.interp(exp_data['TOS_min'], model_results['time_min'], model_results['H2_percent'])
        rmse = np.sqrt(np.mean((model_H2_at_exp - exp_data['H2_percent'])**2))
        
        # Metrics
        met1, met2, met3 = st.columns(3)
        met1.metric("Temperature", f"{selected_temp} °C")
        met2.metric("RMSE (H2%)", f"{rmse:.2f} %")
        met3.metric("kd used", f"{st.session_state.validation_results[selected_temp]['k_d']:.4f}")
        
        # Plot comparison
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # H2 comparison
        axes[0].plot(model_results['time_min'], model_results['H2_percent'], 'b-', lw=2, label='Model')
        axes[0].scatter(exp_data['TOS_min'], exp_data['H2_percent'], c='red', s=80, marker='o', label='Experimental', zorder=5)
        axes[0].set_xlabel('Time on Stream [min]')
        axes[0].set_ylabel('H₂ Composition [%]')
        axes[0].set_title(f'H₂ Composition at {selected_temp}°C')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Activity profile
        axes[1].plot(model_results['time_min'], model_results['activity'], 'g-', lw=2)
        axes[1].set_xlabel('Time on Stream [min]')
        axes[1].set_ylabel('Catalyst Activity [-]')
        axes[1].set_title('Catalyst Deactivation')
        axes[1].set_ylim([0, 1.1])
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        
        # Comparison table
        st.markdown("#### Point-by-Point Comparison")
        comparison_df = pd.DataFrame({
            'TOS (min)': exp_data['TOS_min'],
            'Exp H2 (%)': exp_data['H2_percent'],
            'Model H2 (%)': model_H2_at_exp,
            'Error (%)': model_H2_at_exp - exp_data['H2_percent'],
            'Activity': np.interp(exp_data['TOS_min'], model_results['time_min'], model_results['activity'])
        })
        st.dataframe(comparison_df.style.format({
            'Exp H2 (%)': '{:.2f}',
            'Model H2 (%)': '{:.2f}',
            'Error (%)': '{:+.2f}',
            'Activity': '{:.3f}'
        }), use_container_width=True, hide_index=True)
        
        # Download
        st.download_button(
            "💾 Download Comparison CSV",
            comparison_df.to_csv(index=False),
            f"validation_{selected_temp}C.csv",
            "text/csv"
        )
    
    # Multi-temperature comparison
    st.markdown("---")
    st.markdown("### 🌡️ All Temperatures Comparison")
    
    if st.button("▶️ Run All Temperatures", use_container_width=True):
        progress = st.progress(0)
        status = st.empty()
        
        for i, temp in enumerate(available_temps):
            status.text(f"Running {temp}°C...")
            
            config = create_transient_config(temperature_C=temp)
            kd = st.session_state.fitted_kd.get(temp, 0.01)
            deact_params = DeactivationParams(k_d=kd)
            deact_model = DeactivationModel('first_order', deact_params)
            
            reactor = TransientReactor(config, deact_model)
            results = reactor.solve(t_final_min=210, dt_min=1.0)
            
            st.session_state.validation_results[temp] = {
                'model': results,
                'k_d': kd
            }
            
            progress.progress((i + 1) / len(available_temps))
        
        status.text("✅ All simulations complete!")
    
    # Plot all temperatures
    if len(st.session_state.validation_results) == len(available_temps):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {'770': 'blue', '800': 'orange', '830': 'gray'}
        
        for temp in available_temps:
            model = st.session_state.validation_results[temp]['model']
            exp = get_experimental_data(temp)
            color = colors.get(str(temp), 'black')
            
            ax.plot(model['time_min'], model['H2_percent'], '-', color=color, lw=2, label=f'Model {temp}°C')
            ax.scatter(exp['TOS_min'], exp['H2_percent'], c=color, s=60, marker='o', label=f'Exp {temp}°C')
        
        ax.set_xlabel('Time on Stream [min]', fontsize=12)
        ax.set_ylabel('H₂ Composition [%]', fontsize=12)
        ax.set_title('Model vs Experimental - All Temperatures', fontsize=14)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close(fig)

# ============================================================================
# TAB 3: OPTIMIZER
# ============================================================================
with tab_opt:
    st.subheader("🎯 Bayesian Optimization")
    st.markdown("Find optimal operating conditions using AI-powered search.")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Select Optimization Goal")
        template_names = get_template_names()
        selected_template_key = st.selectbox(
            "Choose a preset:",
            options=list(template_names.keys()),
            format_func=lambda x: template_names[x],
            key="opt_template"
        )
        
        template = get_template(selected_template_key)
        st.info(f"**{template['description']}**")
        
        n_iterations = st.slider("Iterations", 10, 100, template['suggested_iterations'], key="opt_iter")
    
    with col2:
        st.markdown("#### Variable Bounds")
        variable_bounds = {}
        for var in template['variables']:
            c1, c2 = st.columns(2)
            with c1:
                min_val = st.number_input(f"{var['label']} Min", value=float(var['min']), key=f"opt_min_{var['key']}")
            with c2:
                max_val = st.number_input(f"{var['label']} Max", value=float(var['max']), key=f"opt_max_{var['key']}")
            variable_bounds[var['key']] = (min_val, max_val)
    
    if st.button("🚀 Start Optimization", type="primary", use_container_width=True):
        base_config = get_base_config_from_session(st.session_state)
        variable_names = list(variable_bounds.keys())
        bounds = list(variable_bounds.values())
        
        config = OptimizationConfig(
            variable_names=variable_names,
            bounds=bounds,
            n_iterations=n_iterations,
            n_initial_points=min(5, n_iterations // 3),
            maximize=(template['objective']['direction'] == 'maximize')
        )
        
        objective_fn = create_objective_function(
            MethaneDecompositionReactor,
            base_config,
            variable_names,
            template['objective']['target']
        )
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_callback(iteration, params, value):
            progress_bar.progress(iteration / n_iterations)
            status_text.write(f"🔄 Iteration {iteration}/{n_iterations} | Value: {value:.4f}")
        
        optimizer = BayesianOptimizer(config)
        
        with st.spinner("Running optimization..."):
            result = optimizer.optimize(objective_fn, callback=update_callback)
        
        st.session_state.optimization_result = result
        progress_bar.progress(1.0)
        status_text.write("✅ Optimization Complete!")
        st.success(f"**Best Value: {result.best_value:.4f}**")
    
    if st.session_state.optimization_result:
        result = st.session_state.optimization_result
        
        st.markdown("---")
        st.markdown("### 🏆 Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Best Parameters")
            for param, value in result.best_params.items():
                st.metric(param, f"{value:.2f}")
        
        with col2:
            st.markdown("#### Convergence")
            fig = create_convergence_plot(result.convergence, result.best_so_far)
            st.pyplot(fig)
            plt.close(fig)

# ============================================================================
# TAB 4: AI ASSISTANT
# ============================================================================
with tab_chat:
    st.subheader("🤖 AI Assistant")
    
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("📊 Analyze", use_container_width=True):
        handle_ai_request("Analyze the current simulation results.")
    if col2.button("🔧 Optimize Tips", use_container_width=True):
        handle_ai_request("How to optimize H2 yield?")
    if col3.button("📚 Explain", use_container_width=True):
        handle_ai_request("Explain methane decomposition kinetics.")
    if col4.button("🗑️ Clear", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    
    chat_container = st.container(height=400)
    for msg in st.session_state.chat_history:
        with chat_container:
            st.chat_message(msg["role"]).write(msg["content"])
    
    if prompt := st.chat_input("Ask about the simulation..."):
        handle_ai_request(prompt)
        st.rerun()

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.caption("Anukaran AI © 2024 | Methane Decomposition Reactor Simulator | Steady-State + Transient + Optimization")
