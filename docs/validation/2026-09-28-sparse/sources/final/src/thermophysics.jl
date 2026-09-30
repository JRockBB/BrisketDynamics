# Thermophysical constants and smooth helper functions.
#
# Component specific heats and conductivities follow the Choi–Okos
# correlations in temperature; the meat model mixes them by the local
# composition of each node (water, fat, non-fat solids — ash lumped into
# protein) and exposes one conductivity multiplier to the calibration.
# Ref: Choi, Y. & Okos, M.R. (1986), "Effects of temperature and composition on
# the thermal properties of foods", Food Engineering and Process Applications.

const σ_SB = 5.670374419e-8   # Stefan–Boltzmann constant [W m⁻² K⁻⁴]
const R_V = 461.52            # specific gas constant of water vapor [J kg⁻¹ K⁻¹]
const P_ATM = 101325.0        # atmospheric pressure [Pa]
const EPS_W = 0.622           # molar-mass ratio water/air (absolute humidity ↔ vapor pressure)

const CP_WATER = 4186.0       # liquid water heat capacity [J kg⁻¹ K⁻¹] (spritz)
const RHO_DRY = 320.0         # dry-matter density [kg m⁻³] (ρ/(1+W₀) at W₀ ≈ 2.33 dry basis)

# Choi–Okos (1986) component properties as polynomials in °C (`T` in K):
# specific heat [J kg⁻¹ K⁻¹] and thermal conductivity [W m⁻¹ K⁻¹] of water,
# protein (ash lumped in) and fat. Component densities stay the constants of
# the shrinkage bookkeeping (`MeatConfig`); their drift over 5–95 °C is under
# 3 %.
_tc(T) = T - 273.15
cp_water_co(T) = (c = _tc(T); 1e3 * (4.1762 - 9.0864e-5 * c + 5.4731e-6 * c^2))
cp_protein_co(T) = (c = _tc(T); 1e3 * (2.0082 + 1.2089e-3 * c - 1.3129e-6 * c^2))
cp_fat_co(T) = (c = _tc(T); 1e3 * (1.9842 + 1.4733e-3 * c - 4.8008e-6 * c^2))
k_water_co(T) = (c = _tc(T); 0.57109 + 1.7625e-3 * c - 6.7036e-6 * c^2)
k_protein_co(T) = (c = _tc(T); 0.17881 + 1.1958e-3 * c - 2.7178e-6 * c^2)
k_fat_co(T) = (c = _tc(T); 0.18071 - 2.7604e-4 * c - 1.7749e-7 * c^2)

"""
    p_sat(T::Number)

Saturation vapor pressure of water at temperature `T`, Magnus formula.
Smooth in `T` (valid range ≈ 273–373 K, more than covers cooking
temperatures).

# Arguments
- `T::Number`: Temperature [K].

# Returns
- `p::Number`: Saturation vapor pressure [Pa].
"""
p_sat(T) = 610.94 * exp(17.625 * (T - 273.15) / (T - 30.11))

"""
    logistic(x::Number)

Logistic sigmoid `1 / (1 + e⁻ˣ)`, written via `tanh` for numerical stability
at large |x| (saturates cleanly to exactly 0.0 / 1.0 instead of overflowing).

# Arguments
- `x::Number`: Input.

# Returns
- `y::Number`: Sigmoid value in [0, 1].
"""
logistic(x) = (1 + tanh(x / 2)) / 2

