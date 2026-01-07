"""
╔══════════════════════════════════════════════════════════════════════════╗
║                    VALIDATION MODULE                                      ║
║                                                                          ║
║   Validates reactor models against experimental data                     ║
║   - Lab scale validation at 770°C, 800°C, 830°C                         ║
║   - Scale-up validation                                                  ║
║   - Generates validation reports                                         ║
╚══════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# ============================================================================
# EXPERIMENTAL DATA
# ============================================================================

EXPERIMENTAL_DATA = {
    770: {
        'T_K': 1043,
        'H2_percent': 11.41,
        'X': 0.0605,
        'r_W': 1.53e-3,  # mol/(kg·s)
    },
    800: {
        'T_K': 1073,
        'H2_percent': 24.13,
        'X': 0.137,
        'r_W': 3.48e-3,
    },
    830: {
        'T_K': 1103,
        'H2_percent': 40.69,
        'X': 0.255,
        'r_W': 6.47e-3,
    },
}

# Lab configuration (from your PDF)
LAB_REFERENCE = {
    'diameter_m': 0.018,
    'bed_height_m': 0.03,
    'catalyst_mass_kg': 2.2e-3,
    'particle_diameter_m': 10e-6,
    'flow_rate_LPM': 0.075,  # 75 mL/min
    'pressure_bar': 1.0,
    'bulk_density_kg_m3': 334.2,
}

# Industrial configurations
INDUSTRIAL_REFERENCES = {
    'Industrial 1': {
        'diameter_m': 2.909845357,
        'bed_height_m': 0.0256471263,
        'catalyst_mass_kg': 57.0,
        'particle_diameter_m': 10e-6,
        'flow_rate_LPM': 1960,
        'pressure_bar': 1.0,
        'Us_target': 0.0049,
        'description': 'Same superficial velocity as lab',
    },
    'Industrial 2': {
        'diameter_m': 0.5,
        'bed_height_m': 0.8686373914,
        'catalyst_mass_kg': 57.0,
        'particle_diameter_m': 10e-6,
        'flow_rate_LPM': 1960,
        'pressure_bar': 1.0,
        'Us_target': 0.1664,
        'description': 'Higher superficial velocity',
    },
    'Industrial 3': {
        'diameter_m': 1.301322404,
        'bed_height_m': 0.1282356315,
        'catalyst_mass_kg': 57.0,
        'particle_diameter_m': 10e-6,
        'flow_rate_LPM': 1960,
        'pressure_bar': 1.0,
        'Us_target': 0.0246,
        'description': 'Intermediate design',
    },
}

# Calibrated kinetics (from Arrhenius fit)
CALIBRATED_KINETICS = {
    'A': 3.6607e10,      # s⁻¹
    'Ea': 266439.0,      # J/mol (266.4 kJ/mol)
}


# ============================================================================
# VALIDATION RESULT CLASS
# ============================================================================

@dataclass
class ValidationResult:
    """Results from validation test"""
    temperature_C: int
    experimental_H2: float
    model_H2: float
    experimental_X: float
    model_X: float
    H2_error_abs: float
    H2_error_rel: float
    X_error_rel: float
    passed: bool
    
    def __str__(self):
        status = "✅ PASS" if self.passed else "❌ FAIL"
        return (f"T={self.temperature_C}°C: H₂_exp={self.experimental_H2:.2f}%, "
                f"H₂_model={self.model_H2:.2f}%, Error={self.H2_error_abs:.2f}% {status}")


@dataclass 
class ScaleUpValidation:
    """Results from scale-up validation"""
    lab_H2: float
    industrial_H2: float
    lab_tau: float
    industrial_tau: float
    tau_ratio: float
    H2_match: bool
    pressure_drop_kPa: float
    feasible: bool
    
    def __str__(self):
        match_status = "✅" if self.H2_match else "❌"
        feasible_status = "✅" if self.feasible else "❌"
        return (f"Lab H₂={self.lab_H2:.2f}%, Ind H₂={self.industrial_H2:.2f}% "
                f"[Match: {match_status}] ΔP={self.pressure_drop_kPa:.1f} kPa [Feasible: {feasible_status}]")


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_kinetics(model_func, config_func, tolerance_percent: float = 2.0) -> List[ValidationResult]:
    """
    Validate model against experimental data at all temperatures.
    
    Args:
        model_func: Function that takes config and returns dict with 'H2_percent' and 'conversion'
        config_func: Function that takes temperature_C and returns config object
        tolerance_percent: Acceptable H₂% error (default 2%)
    
    Returns:
        List of ValidationResult objects
    """
    results = []
    
    for T_C, exp_data in EXPERIMENTAL_DATA.items():
        config = config_func(T_C)
        model_result = model_func(config)
        
        H2_model = model_result.get('H2_percent', model_result.get('outlet_H2_percent', 0))
        X_model = model_result.get('conversion', model_result.get('outlet_conversion', 0))
        
        H2_error_abs = abs(H2_model - exp_data['H2_percent'])
        H2_error_rel = H2_error_abs / exp_data['H2_percent'] * 100
        X_error_rel = abs(X_model - exp_data['X']) / exp_data['X'] * 100
        
        passed = H2_error_abs <= tolerance_percent
        
        results.append(ValidationResult(
            temperature_C=T_C,
            experimental_H2=exp_data['H2_percent'],
            model_H2=H2_model,
            experimental_X=exp_data['X'],
            model_X=X_model,
            H2_error_abs=H2_error_abs,
            H2_error_rel=H2_error_rel,
            X_error_rel=X_error_rel,
            passed=passed,
        ))
    
    return results


def validate_scaleup(lab_result: Dict, industrial_result: Dict, 
                     max_pressure_drop_kPa: float = 100.0) -> ScaleUpValidation:
    """
    Validate scale-up consistency.
    
    Args:
        lab_result: Results dict from lab scale simulation
        industrial_result: Results dict from industrial scale simulation
        max_pressure_drop_kPa: Maximum acceptable pressure drop
    
    Returns:
        ScaleUpValidation object
    """
    lab_H2 = lab_result.get('H2_percent', lab_result.get('outlet_H2_percent', 0))
    ind_H2 = industrial_result.get('H2_percent', industrial_result.get('outlet_H2_percent', 0))
    lab_tau = lab_result.get('space_time_s', 0)
    ind_tau = industrial_result.get('space_time_s', 0)
    ind_dP = industrial_result.get('pressure_drop_kPa', 0)
    
    tau_ratio = ind_tau / lab_tau if lab_tau > 0 else 0
    H2_match = abs(lab_H2 - ind_H2) < 1.0  # Within 1%
    feasible = ind_dP < max_pressure_drop_kPa
    
    return ScaleUpValidation(
        lab_H2=lab_H2,
        industrial_H2=ind_H2,
        lab_tau=lab_tau,
        industrial_tau=ind_tau,
        tau_ratio=tau_ratio,
        H2_match=H2_match,
        pressure_drop_kPa=ind_dP,
        feasible=feasible,
    )


def generate_validation_report(validation_results: List[ValidationResult],
                                scaleup_results: List[ScaleUpValidation] = None) -> str:
    """
    Generate a text validation report.
    """
    report = []
    report.append("=" * 70)
    report.append("REACTOR MODEL VALIDATION REPORT")
    report.append("=" * 70)
    
    # Kinetics validation
    report.append("\n📊 KINETICS VALIDATION (vs Experimental Data)")
    report.append("-" * 50)
    report.append(f"{'T (°C)':<10} {'H₂_exp':<12} {'H₂_model':<12} {'Error':<10} {'Status':<10}")
    report.append("-" * 55)
    
    all_passed = True
    for vr in validation_results:
        status = "✅ PASS" if vr.passed else "❌ FAIL"
        if not vr.passed:
            all_passed = False
        report.append(f"{vr.temperature_C:<10} {vr.experimental_H2:<12.2f} "
                     f"{vr.model_H2:<12.2f} {vr.H2_error_abs:<10.2f} {status:<10}")
    
    avg_error = np.mean([vr.H2_error_abs for vr in validation_results])
    max_error = max([vr.H2_error_abs for vr in validation_results])
    
    report.append("-" * 55)
    report.append(f"Average Error: {avg_error:.2f}%")
    report.append(f"Maximum Error: {max_error:.2f}%")
    report.append(f"Overall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}")
    
    # Scale-up validation
    if scaleup_results:
        report.append("\n📏 SCALE-UP VALIDATION")
        report.append("-" * 50)
        
        for i, su in enumerate(scaleup_results):
            report.append(f"\nConfiguration {i+1}:")
            report.append(f"  Lab H₂: {su.lab_H2:.2f}%")
            report.append(f"  Industrial H₂: {su.industrial_H2:.2f}%")
            report.append(f"  τ ratio: {su.tau_ratio:.3f}")
            report.append(f"  H₂ Match: {'✅ Yes' if su.H2_match else '❌ No'}")
            report.append(f"  ΔP: {su.pressure_drop_kPa:.2f} kPa")
            report.append(f"  Feasible: {'✅ Yes' if su.feasible else '❌ No (ΔP too high)'}")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


def get_validation_dataframe(validation_results: List[ValidationResult]) -> pd.DataFrame:
    """Convert validation results to pandas DataFrame for display."""
    data = []
    for vr in validation_results:
        data.append({
            'Temperature (°C)': vr.temperature_C,
            'T (K)': vr.temperature_C + 273,
            'H₂ Exp (%)': vr.experimental_H2,
            'H₂ Model (%)': vr.model_H2,
            'Error (%)': vr.H2_error_abs,
            'X Exp': vr.experimental_X,
            'X Model': vr.model_X,
            'Status': '✅ Pass' if vr.passed else '❌ Fail',
        })
    return pd.DataFrame(data)


# ============================================================================
# EXPECTED VALUES
# ============================================================================

def get_expected_values() -> pd.DataFrame:
    """
    Get expected values for validation.
    
    Returns DataFrame with expected H₂% for lab and industrial configs.
    """
    data = [
        {
            'Configuration': 'Lab Scale',
            'D (m)': LAB_REFERENCE['diameter_m'],
            'H_bed (m)': LAB_REFERENCE['bed_height_m'],
            'W (kg)': LAB_REFERENCE['catalyst_mass_kg'],
            'Q (LPM)': LAB_REFERENCE['flow_rate_LPM'],
            'τ (s)': '~39.4',
            'Expected H₂ (%)': '~24% (at 800°C)',
            'Expected ΔP (kPa)': '<1',
            'Status': 'Baseline',
        },
    ]
    
    for name, config in INDUSTRIAL_REFERENCES.items():
        data.append({
            'Configuration': name,
            'D (m)': config['diameter_m'],
            'H_bed (m)': config['bed_height_m'],
            'W (kg)': config['catalyst_mass_kg'],
            'Q (LPM)': config['flow_rate_LPM'],
            'τ (s)': '~39.1',
            'Expected H₂ (%)': '~24% (same as lab)',
            'Expected ΔP (kPa)': 'Varies',
            'Status': config['description'],
        })
    
    return pd.DataFrame(data)


# ============================================================================
# MAIN TEST
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VALIDATION MODULE TEST")
    print("=" * 70)
    
    # Test with fluidized bed model
    try:
        from fluidized_bed_model import solve_cstr, get_lab_config
        
        results = validate_kinetics(
            model_func=solve_cstr,
            config_func=get_lab_config,
            tolerance_percent=2.0
        )
        
        print("\nValidation Results:")
        for r in results:
            print(f"  {r}")
        
        report = generate_validation_report(results)
        print("\n" + report)
        
    except ImportError:
        print("Could not import fluidized_bed_model. Run from correct directory.")
    
    # Show expected values
    print("\n📋 EXPECTED VALUES:")
    print(get_expected_values().to_string(index=False))
