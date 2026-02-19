"""
Built-in species property database for KinetiScale.

Contains thermophysical properties for common chemical species used in
catalytic reactor simulations. Properties are stored as simple correlation
parameters (power-law viscosity, polynomial Cp, Chapman-Enskog diffusivity).

All units are SI:
  - MW: kg/mol
  - mu_ref: Pa·s (at mu_T_ref K)
  - Cp_a, Cp_b, Cp_c: J/(mol·K)  (Cp = a + b*T + c*T^2)
  - D_ref: m²/s (at D_T_ref K, D_P_ref Pa)
"""

from .chemistry import Species


SPECIES_DATABASE = {

    # ── Hydrocarbons ──────────────────────────────────────────────────

    "CH4": Species(
        name="CH4", display_name="Methane", formula="CH₄",
        MW=16.04e-3, phase="gas",
        mu_ref=1.02e-5, mu_T_ref=300, mu_exp=0.87,
        Cp_a=35.69, Cp_b=0.0275, Cp_c=0.0,
        D_ref=1.87e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "C2H6": Species(
        name="C2H6", display_name="Ethane", formula="C₂H₆",
        MW=30.07e-3, phase="gas",
        mu_ref=9.4e-6, mu_T_ref=300, mu_exp=0.84,
        Cp_a=52.63, Cp_b=0.0375, Cp_c=0.0,
        D_ref=1.08e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "C2H4": Species(
        name="C2H4", display_name="Ethylene", formula="C₂H₄",
        MW=28.05e-3, phase="gas",
        mu_ref=1.03e-5, mu_T_ref=300, mu_exp=0.82,
        Cp_a=42.90, Cp_b=0.0335, Cp_c=0.0,
        D_ref=1.18e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "C3H8": Species(
        name="C3H8", display_name="Propane", formula="C₃H₈",
        MW=44.10e-3, phase="gas",
        mu_ref=8.1e-6, mu_T_ref=300, mu_exp=0.85,
        Cp_a=73.60, Cp_b=0.0450, Cp_c=0.0,
        D_ref=8.7e-6, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "C3H6": Species(
        name="C3H6", display_name="Propylene", formula="C₃H₆",
        MW=42.08e-3, phase="gas",
        mu_ref=8.4e-6, mu_T_ref=300, mu_exp=0.83,
        Cp_a=64.30, Cp_b=0.0400, Cp_c=0.0,
        D_ref=9.2e-6, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    # ── Alcohols & Oxygenates ─────────────────────────────────────────

    "C2H5OH": Species(
        name="C2H5OH", display_name="Ethanol", formula="C₂H₅OH",
        MW=46.07e-3, phase="gas",
        mu_ref=8.8e-6, mu_T_ref=300, mu_exp=0.89,
        Cp_a=65.60, Cp_b=0.0452, Cp_c=0.0,
        D_ref=1.04e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "CH3OH": Species(
        name="CH3OH", display_name="Methanol", formula="CH₃OH",
        MW=32.04e-3, phase="gas",
        mu_ref=9.7e-6, mu_T_ref=300, mu_exp=0.88,
        Cp_a=44.10, Cp_b=0.0318, Cp_c=0.0,
        D_ref=1.33e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "CH3CHO": Species(
        name="CH3CHO", display_name="Acetaldehyde", formula="CH₃CHO",
        MW=44.05e-3, phase="gas",
        mu_ref=8.6e-6, mu_T_ref=300, mu_exp=0.86,
        Cp_a=55.32, Cp_b=0.0380, Cp_c=0.0,
        D_ref=1.05e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "CH2O": Species(
        name="CH2O", display_name="Formaldehyde", formula="CH₂O",
        MW=30.03e-3, phase="gas",
        mu_ref=1.0e-5, mu_T_ref=300, mu_exp=0.85,
        Cp_a=35.40, Cp_b=0.0225, Cp_c=0.0,
        D_ref=1.40e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    # ── Common Gases ──────────────────────────────────────────────────

    "H2": Species(
        name="H2", display_name="Hydrogen", formula="H₂",
        MW=2.016e-3, phase="gas",
        mu_ref=8.76e-6, mu_T_ref=300, mu_exp=0.68,
        Cp_a=28.84, Cp_b=0.00192, Cp_c=0.0,
        D_ref=6.10e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "N2": Species(
        name="N2", display_name="Nitrogen", formula="N₂",
        MW=28.01e-3, phase="gas",
        mu_ref=1.78e-5, mu_T_ref=300, mu_exp=0.67,
        Cp_a=29.12, Cp_b=0.00293, Cp_c=0.0,
        D_ref=1.80e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "O2": Species(
        name="O2", display_name="Oxygen", formula="O₂",
        MW=32.00e-3, phase="gas",
        mu_ref=2.02e-5, mu_T_ref=300, mu_exp=0.69,
        Cp_a=29.38, Cp_b=0.00418, Cp_c=0.0,
        D_ref=1.78e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "H2O": Species(
        name="H2O", display_name="Water (steam)", formula="H₂O",
        MW=18.015e-3, phase="gas",
        mu_ref=1.05e-5, mu_T_ref=300, mu_exp=0.91,
        Cp_a=33.56, Cp_b=0.00688, Cp_c=0.0,
        D_ref=2.20e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "CO": Species(
        name="CO", display_name="Carbon Monoxide", formula="CO",
        MW=28.01e-3, phase="gas",
        mu_ref=1.72e-5, mu_T_ref=300, mu_exp=0.69,
        Cp_a=29.15, Cp_b=0.00411, Cp_c=0.0,
        D_ref=1.85e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "CO2": Species(
        name="CO2", display_name="Carbon Dioxide", formula="CO₂",
        MW=44.01e-3, phase="gas",
        mu_ref=1.49e-5, mu_T_ref=300, mu_exp=0.78,
        Cp_a=37.11, Cp_b=0.0143, Cp_c=0.0,
        D_ref=1.39e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "NH3": Species(
        name="NH3", display_name="Ammonia", formula="NH₃",
        MW=17.03e-3, phase="gas",
        mu_ref=9.8e-6, mu_T_ref=300, mu_exp=0.90,
        Cp_a=35.06, Cp_b=0.0296, Cp_c=0.0,
        D_ref=1.98e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "SO2": Species(
        name="SO2", display_name="Sulfur Dioxide", formula="SO₂",
        MW=64.07e-3, phase="gas",
        mu_ref=1.26e-5, mu_T_ref=300, mu_exp=0.82,
        Cp_a=39.87, Cp_b=0.0128, Cp_c=0.0,
        D_ref=1.03e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "SO3": Species(
        name="SO3", display_name="Sulfur Trioxide", formula="SO₃",
        MW=80.06e-3, phase="gas",
        mu_ref=1.20e-5, mu_T_ref=300, mu_exp=0.80,
        Cp_a=48.50, Cp_b=0.0150, Cp_c=0.0,
        D_ref=8.5e-6, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "NO": Species(
        name="NO", display_name="Nitric Oxide", formula="NO",
        MW=30.01e-3, phase="gas",
        mu_ref=1.90e-5, mu_T_ref=300, mu_exp=0.69,
        Cp_a=29.34, Cp_b=0.00350, Cp_c=0.0,
        D_ref=1.80e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "NO2": Species(
        name="NO2", display_name="Nitrogen Dioxide", formula="NO₂",
        MW=46.01e-3, phase="gas",
        mu_ref=1.35e-5, mu_T_ref=300, mu_exp=0.79,
        Cp_a=36.97, Cp_b=0.0120, Cp_c=0.0,
        D_ref=1.30e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "HCl": Species(
        name="HCl", display_name="Hydrogen Chloride", formula="HCl",
        MW=36.46e-3, phase="gas",
        mu_ref=1.31e-5, mu_T_ref=300, mu_exp=0.80,
        Cp_a=29.13, Cp_b=0.00341, Cp_c=0.0,
        D_ref=1.55e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "Cl2": Species(
        name="Cl2", display_name="Chlorine", formula="Cl₂",
        MW=70.91e-3, phase="gas",
        mu_ref=1.33e-5, mu_T_ref=300, mu_exp=0.75,
        Cp_a=33.95, Cp_b=0.00560, Cp_c=0.0,
        D_ref=9.4e-6, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    # ── Noble / Inert Gases ───────────────────────────────────────────

    "Ar": Species(
        name="Ar", display_name="Argon", formula="Ar",
        MW=39.95e-3, phase="gas",
        mu_ref=2.23e-5, mu_T_ref=300, mu_exp=0.72,
        Cp_a=20.79, Cp_b=0.0, Cp_c=0.0,
        D_ref=1.50e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    "He": Species(
        name="He", display_name="Helium", formula="He",
        MW=4.003e-3, phase="gas",
        mu_ref=1.96e-5, mu_T_ref=300, mu_exp=0.67,
        Cp_a=20.79, Cp_b=0.0, Cp_c=0.0,
        D_ref=5.80e-5, D_T_ref=300, D_exp=1.75, D_P_ref=101325,
    ),

    # ── Solids (for deposition tracking) ──────────────────────────────

    "C": Species(
        name="C", display_name="Carbon (solid)", formula="C(s)",
        MW=12.01e-3, phase="solid",
    ),

    "S": Species(
        name="S", display_name="Sulfur (solid)", formula="S(s)",
        MW=32.07e-3, phase="solid",
    ),

    "Fe": Species(
        name="Fe", display_name="Iron (solid)", formula="Fe(s)",
        MW=55.85e-3, phase="solid",
    ),
}


def get_species(name: str) -> Species:
    """Get a species from the database by name. Raises KeyError if not found."""
    if name not in SPECIES_DATABASE:
        available = ", ".join(sorted(SPECIES_DATABASE.keys()))
        raise KeyError(f"Species '{name}' not found in database. Available: {available}")
    return SPECIES_DATABASE[name]


def list_gas_species() -> list:
    """Return names of all gas-phase species in the database."""
    return [name for name, sp in SPECIES_DATABASE.items() if sp.phase == "gas"]


def list_solid_species() -> list:
    """Return names of all solid-phase species in the database."""
    return [name for name, sp in SPECIES_DATABASE.items() if sp.phase == "solid"]
