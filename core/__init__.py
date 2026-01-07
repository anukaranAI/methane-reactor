"""
Anukaran AI - Core Module
=========================
Reactor simulation, optimization, and scale-up.
"""

from .templates import (
    OPTIMIZATION_TEMPLATES,
    get_template,
    get_template_names,
    get_variable_bounds,
    get_variable_names
)

from .optimizer import (
    OptimizationResult,
    OptimizationConfig,
    BayesianOptimizer,
    SensitivityAnalyzer,
    create_objective_function,
    get_base_config_from_session
)

from .transient_model import (
    DeactivationParams,
    DeactivationModel,
    ReactorConfig,
    SteadyStateReactor,
    TransientReactor,
    calibrate_kinetics,
    fit_deactivation,
)

from .experimental_data import (
    EXPERIMENTAL_DATA,
    TOS_MINUTES,
    LabScaleConfig,
    IndustrialScaleConfig,
    LAB_CONFIG,
    INDUSTRIAL_CONFIG,
    get_experimental_data,
    get_available_temperatures,
    get_initial_values,
    get_all_initial_values,
    get_scale_comparison,
)

from .scaleup import (
    ScaleUpResult,
    IndustrialScaleUp,
    get_assumptions_text,
)

# Fluidized bed model (CSTR)
from .fluidized_bed_model import (
    FluidizedBedConfig, ReactorModelType,
    solve_cstr, solve_pfr, solve_reactor,
    get_lab_config, get_industrial_config_1, get_industrial_config_2, get_industrial_config_3,
    validate_against_experimental, conversion_to_H2_percent, H2_percent_to_conversion,
    CALIBRATED_KINETICS, EXPERIMENTAL_DATA_FBR,
)

# Validation module
from .validation import (
    validate_kinetics, validate_scaleup, generate_validation_report,
    get_validation_dataframe, get_expected_values,
    EXPERIMENTAL_DATA, LAB_REFERENCE, INDUSTRIAL_REFERENCES,
)

__all__ = [
    # Templates
    'OPTIMIZATION_TEMPLATES', 'get_template', 'get_template_names',
    'get_variable_bounds', 'get_variable_names',
    # Optimizer
    'OptimizationResult', 'OptimizationConfig', 'BayesianOptimizer',
    'SensitivityAnalyzer', 'create_objective_function', 'get_base_config_from_session',
    # Transient
    'DeactivationParams', 'DeactivationModel', 'ReactorConfig',
    'SteadyStateReactor', 'TransientReactor',
    'calibrate_kinetics', 'fit_deactivation',
    # Data
    'EXPERIMENTAL_DATA', 'TOS_MINUTES', 'LabScaleConfig', 'IndustrialScaleConfig',
    'LAB_CONFIG', 'INDUSTRIAL_CONFIG', 'get_experimental_data',
    'get_available_temperatures', 'get_initial_values', 'get_all_initial_values',
    'get_scale_comparison',
    # Scale-up
    'ScaleUpResult', 'IndustrialScaleUp', 'get_assumptions_text',
]
