"""
KinetiScale - Core Module
=========================
Generic catalytic reactor simulation, optimization, and scale-up.
"""

# Chemistry abstraction
from .chemistry import (
    Species,
    Reaction,
    ReactionTemplate,
    R_GAS,
)

# Species database
from .species_db import (
    SPECIES_DATABASE,
    get_species,
    list_gas_species,
    list_solid_species,
)

# Reaction templates
from .reaction_templates import (
    get_template,
    get_all_template_names,
    get_all_templates,
)

# Transport properties
from .transport import (
    gas_mixture_viscosity,
    gas_mixture_density,
    gas_mixture_Cp,
    effective_diffusivity,
    ergun_pressure_drop,
)

# Reactor solvers
from .transient_model import (
    DeactivationParams,
    DeactivationModel,
    ReactorConfig,
    SteadyStateReactor,
    TransientReactor,
    calculate_solid_deposition,
    calibrate_kinetics,
    fit_deactivation,
    arrhenius_rate_constant,
    effectiveness_factor,
)

# Reactor configurations
from .experimental_data import (
    LabScaleConfig,
    IndustrialScaleConfig,
    LAB_CONFIG,
    INDUSTRIAL_CONFIG,
    get_experimental_data_from_template,
    get_available_temperatures_from_template,
)

# Scale-up
from .scaleup import (
    ScaleUpResult,
    IndustrialScaleUp,
    get_assumptions_text,
)

__all__ = [
    # Chemistry
    'Species', 'Reaction', 'ReactionTemplate', 'R_GAS',
    # Species DB
    'SPECIES_DATABASE', 'get_species', 'list_gas_species', 'list_solid_species',
    # Templates
    'get_template', 'get_all_template_names', 'get_all_templates',
    # Transport
    'gas_mixture_viscosity', 'gas_mixture_density', 'gas_mixture_Cp',
    'effective_diffusivity', 'ergun_pressure_drop',
    # Solvers
    'DeactivationParams', 'DeactivationModel', 'ReactorConfig',
    'SteadyStateReactor', 'TransientReactor',
    'calculate_solid_deposition', 'calibrate_kinetics', 'fit_deactivation',
    'arrhenius_rate_constant', 'effectiveness_factor',
    # Configs
    'LabScaleConfig', 'IndustrialScaleConfig', 'LAB_CONFIG', 'INDUSTRIAL_CONFIG',
    'get_experimental_data_from_template', 'get_available_temperatures_from_template',
    # Scale-up
    'ScaleUpResult', 'IndustrialScaleUp', 'get_assumptions_text',
]
