"""
Pre-built reaction templates for KinetiScale.

Each template provides a complete reaction definition with default parameters,
allowing users to start simulating immediately without manual configuration.
"""

import numpy as np
from .chemistry import Reaction, ReactionTemplate
from .species_db import get_species


# ============================================================================
# EXPERIMENTAL DATA (for Methane Decomposition only)
# ============================================================================

_TOS_MINUTES = np.array([0, 30, 60, 90, 120, 150, 180, 210])

_METHANE_EXPERIMENTAL_DATA = {
    770: {
        "TOS_min": _TOS_MINUTES,
        "product_percent": np.array([11.41, 20.28, 16.1, 14.08, 12.81, 11.82, 11.12, 10.33]),
        "reactant_percent": np.array([24.87, 79.71, 83.89, 85.91, 87.18, 88.17, 88.87, 89.66]),
    },
    800: {
        "TOS_min": _TOS_MINUTES,
        "product_percent": np.array([24.13, 24.5, 20.51, 17.8, 16.74, 15.51, 14.44, 13.43]),
        "reactant_percent": np.array([65.0, 75.49, 79.48, 82.19, 83.25, 84.48, 85.55, 86.56]),
    },
    830: {
        "TOS_min": _TOS_MINUTES,
        "product_percent": np.array([40.69, 31.68, 25.23, 23.9, 22.29, 20.36, 19.8, 18.79]),
        "reactant_percent": np.array([44.64, 68.31, 74.76, 76.09, 77.7, 79.63, 80.19, 81.2]),
    },
}


# ============================================================================
# REACTION TEMPLATES
# ============================================================================

def _build_methane_decomposition() -> ReactionTemplate:
    """CH₄ → C(s) + 2H₂  (catalytic methane decomposition on Ni)"""
    species = {
        "CH4": get_species("CH4"),
        "H2": get_species("H2"),
        "C": get_species("C"),
        "N2": get_species("N2"),
    }
    rxn = Reaction(
        name="methane_decomposition",
        display_name="Methane Decomposition",
        stoichiometry={"CH4": -1, "C": +1, "H2": +2},
        species=species,
        key_reactant="CH4",
        key_product="H2",
        delta_H=74870.0,  # J/mol, endothermic
        inert="N2",
    )
    return ReactionTemplate(
        reaction=rxn,
        default_A=5.0e4,
        default_Ea=150000.0,
        default_beta=0.0,
        default_kd=0.008,
        default_deact_order=1,
        default_inlet={"CH4": 1.0},
        default_T_C=800.0,
        default_P_bar=1.0,
        default_lab_flow_mL_min=75.0,
        default_lab_diameter_cm=1.8,
        default_lab_height_cm=2.59,
        default_lab_catalyst_mass_g=2.2,
        default_lab_particle_size_um=10.0,
        default_lab_particle_density=2821.56,
        default_ind_flow_LPM=1960.0,
        default_ind_catalyst_mass_kg=57.0,
        default_ind_particle_size_um=3000.0,
        default_ind_particle_density=2821.56,
        default_ind_target_product_pct=30.0,
        experimental_data=_METHANE_EXPERIMENTAL_DATA,
        description=(
            "Catalytic decomposition of methane over Ni-based catalyst. "
            "Produces hydrogen and solid carbon. Endothermic reaction "
            "(ΔH = +74.87 kJ/mol). Carbon deposition causes catalyst deactivation."
        ),
    )


def _build_steam_methane_reforming() -> ReactionTemplate:
    """CH₄ + H₂O → CO + 3H₂  (steam methane reforming)"""
    species = {
        "CH4": get_species("CH4"),
        "H2O": get_species("H2O"),
        "CO": get_species("CO"),
        "H2": get_species("H2"),
        "N2": get_species("N2"),
    }
    rxn = Reaction(
        name="steam_methane_reforming",
        display_name="Steam Methane Reforming",
        stoichiometry={"CH4": -1, "H2O": -1, "CO": +1, "H2": +3},
        species=species,
        key_reactant="CH4",
        key_product="H2",
        delta_H=206000.0,  # J/mol, highly endothermic
        inert="N2",
    )
    return ReactionTemplate(
        reaction=rxn,
        default_A=1.0e5,
        default_Ea=180000.0,
        default_beta=0.0,
        default_kd=0.005,
        default_deact_order=1,
        default_inlet={"CH4": 0.25, "H2O": 0.75},
        default_T_C=850.0,
        default_P_bar=1.0,
        default_lab_flow_mL_min=100.0,
        default_lab_diameter_cm=2.0,
        default_lab_height_cm=3.0,
        default_lab_catalyst_mass_g=3.0,
        default_lab_particle_size_um=50.0,
        default_lab_particle_density=2500.0,
        default_ind_flow_LPM=2000.0,
        default_ind_catalyst_mass_kg=60.0,
        default_ind_particle_size_um=3000.0,
        default_ind_particle_density=2500.0,
        default_ind_target_product_pct=40.0,
        experimental_data=None,
        description=(
            "Steam reforming of methane over Ni-based catalyst. "
            "Primary industrial route for hydrogen production. "
            "Highly endothermic (ΔH = +206 kJ/mol). Operates at 700-900°C."
        ),
    )


def _build_water_gas_shift() -> ReactionTemplate:
    """CO + H₂O → CO₂ + H₂  (water-gas shift)"""
    species = {
        "CO": get_species("CO"),
        "H2O": get_species("H2O"),
        "CO2": get_species("CO2"),
        "H2": get_species("H2"),
        "N2": get_species("N2"),
    }
    rxn = Reaction(
        name="water_gas_shift",
        display_name="Water-Gas Shift",
        stoichiometry={"CO": -1, "H2O": -1, "CO2": +1, "H2": +1},
        species=species,
        key_reactant="CO",
        key_product="H2",
        delta_H=-41200.0,  # J/mol, mildly exothermic
        inert="N2",
    )
    return ReactionTemplate(
        reaction=rxn,
        default_A=2.0e4,
        default_Ea=120000.0,
        default_beta=0.0,
        default_kd=0.003,
        default_deact_order=1,
        default_inlet={"CO": 0.30, "H2O": 0.70},
        default_T_C=400.0,
        default_P_bar=1.0,
        default_lab_flow_mL_min=80.0,
        default_lab_diameter_cm=1.5,
        default_lab_height_cm=3.0,
        default_lab_catalyst_mass_g=2.5,
        default_lab_particle_size_um=50.0,
        default_lab_particle_density=3000.0,
        default_ind_flow_LPM=1500.0,
        default_ind_catalyst_mass_kg=50.0,
        default_ind_particle_size_um=3000.0,
        default_ind_particle_density=3000.0,
        default_ind_target_product_pct=25.0,
        experimental_data=None,
        description=(
            "Water-gas shift reaction over Fe-Cr or Cu-Zn catalyst. "
            "Converts CO and steam to CO₂ and H₂. Mildly exothermic "
            "(ΔH = -41.2 kJ/mol). Used downstream of reformers."
        ),
    )


def _build_ammonia_decomposition() -> ReactionTemplate:
    """2NH₃ → N₂ + 3H₂  (ammonia cracking)"""
    species = {
        "NH3": get_species("NH3"),
        "N2": get_species("N2"),
        "H2": get_species("H2"),
        "Ar": get_species("Ar"),
    }
    rxn = Reaction(
        name="ammonia_decomposition",
        display_name="Ammonia Decomposition",
        stoichiometry={"NH3": -2, "N2": +1, "H2": +3},
        species=species,
        key_reactant="NH3",
        key_product="H2",
        delta_H=46100.0,  # J/mol NH3, endothermic (per 2 mol NH3 = 92.2 kJ)
        inert="Ar",
    )
    return ReactionTemplate(
        reaction=rxn,
        default_A=8.0e4,
        default_Ea=160000.0,
        default_beta=0.0,
        default_kd=0.002,
        default_deact_order=1,
        default_inlet={"NH3": 1.0},
        default_T_C=600.0,
        default_P_bar=1.0,
        default_lab_flow_mL_min=60.0,
        default_lab_diameter_cm=1.5,
        default_lab_height_cm=2.5,
        default_lab_catalyst_mass_g=2.0,
        default_lab_particle_size_um=20.0,
        default_lab_particle_density=3200.0,
        default_ind_flow_LPM=1500.0,
        default_ind_catalyst_mass_kg=45.0,
        default_ind_particle_size_um=3000.0,
        default_ind_particle_density=3200.0,
        default_ind_target_product_pct=35.0,
        experimental_data=None,
        description=(
            "Catalytic decomposition of ammonia over Ru or Ni catalyst. "
            "Emerging route for hydrogen carrier/storage. "
            "Endothermic (ΔH = +46.1 kJ/mol). Operates at 500-700°C."
        ),
    )


def _build_ethanol_dehydrogenation() -> ReactionTemplate:
    """C₂H₅OH → CH₃CHO + H₂  (ethanol dehydrogenation)"""
    species = {
        "C2H5OH": get_species("C2H5OH"),
        "CH3CHO": get_species("CH3CHO"),
        "H2": get_species("H2"),
        "N2": get_species("N2"),
    }
    rxn = Reaction(
        name="ethanol_dehydrogenation",
        display_name="Ethanol Dehydrogenation",
        stoichiometry={"C2H5OH": -1, "CH3CHO": +1, "H2": +1},
        species=species,
        key_reactant="C2H5OH",
        key_product="H2",
        delta_H=68900.0,  # J/mol, endothermic
        inert="N2",
    )
    return ReactionTemplate(
        reaction=rxn,
        default_A=3.0e4,
        default_Ea=130000.0,
        default_beta=0.0,
        default_kd=0.006,
        default_deact_order=1,
        default_inlet={"C2H5OH": 1.0},
        default_T_C=300.0,
        default_P_bar=1.0,
        default_lab_flow_mL_min=50.0,
        default_lab_diameter_cm=1.5,
        default_lab_height_cm=2.5,
        default_lab_catalyst_mass_g=1.5,
        default_lab_particle_size_um=30.0,
        default_lab_particle_density=2800.0,
        default_ind_flow_LPM=1000.0,
        default_ind_catalyst_mass_kg=40.0,
        default_ind_particle_size_um=3000.0,
        default_ind_particle_density=2800.0,
        default_ind_target_product_pct=20.0,
        experimental_data=None,
        description=(
            "Catalytic dehydrogenation of ethanol to acetaldehyde and hydrogen "
            "over Cu or Cu-Cr catalyst. Endothermic (ΔH = +68.9 kJ/mol). "
            "Operates at 250-350°C."
        ),
    )


# ============================================================================
# TEMPLATE REGISTRY
# ============================================================================

# Lazy-build cache
_TEMPLATE_CACHE = {}


def _ensure_cache():
    if not _TEMPLATE_CACHE:
        _TEMPLATE_CACHE["methane_decomposition"] = _build_methane_decomposition()
        _TEMPLATE_CACHE["steam_methane_reforming"] = _build_steam_methane_reforming()
        _TEMPLATE_CACHE["water_gas_shift"] = _build_water_gas_shift()
        _TEMPLATE_CACHE["ammonia_decomposition"] = _build_ammonia_decomposition()
        _TEMPLATE_CACHE["ethanol_dehydrogenation"] = _build_ethanol_dehydrogenation()


def get_template(name: str) -> ReactionTemplate:
    """Get a reaction template by name."""
    _ensure_cache()
    if name not in _TEMPLATE_CACHE:
        available = ", ".join(sorted(_TEMPLATE_CACHE.keys()))
        raise KeyError(f"Template '{name}' not found. Available: {available}")
    return _TEMPLATE_CACHE[name]


def get_all_template_names() -> dict:
    """Return {template_key: display_name} for all templates."""
    _ensure_cache()
    return {key: t.reaction.display_name for key, t in _TEMPLATE_CACHE.items()}


def get_all_templates() -> dict:
    """Return all templates as {key: ReactionTemplate}."""
    _ensure_cache()
    return dict(_TEMPLATE_CACHE)
