# Thermophysical constants and smooth helper functions.
#
# Property constants are evaluated once from the Choi–Okos correlations at 60 °C
# for a typical brisket composition (≈70% water, 20% protein, 9% fat, 1% ash,
# wet basis) and treated as fixed; the calibration exposes a single conductivity
# multiplier instead of the full temperature/composition dependence, which a
# single-probe cook log cannot identify.
# Ref: Choi, Y. & Okos, M.R. (1986), "Effects of temperature and composition on
# the thermal properties of foods", Food Engineering and Process Applications.

const σ_SB = 5.670374419e-8   # Stefan–Boltzmann constant [W m⁻² K⁻⁴]
const R_V = 461.52            # specific gas constant of water vapor [J kg⁻¹ K⁻¹]
const P_ATM = 101325.0        # atmospheric pressure [Pa]
const EPS_W = 0.622           # molar-mass ratio water/air (absolute humidity ↔ vapor pressure)

const K_MEAT = 0.45           # baseline thermal conductivity [W m⁻¹ K⁻¹]
const RHO_MEAT = 1060.0       # density [kg m⁻³]
const CP_MEAT = 3400.0        # specific heat capacity [J kg⁻¹ K⁻¹]
const CP_WATER = 4186.0       # liquid water heat capacity [J kg⁻¹ K⁻¹] (spritz)
const RHO_DRY = 320.0         # dry-matter density [kg m⁻³] (ρ/(1+W₀) at W₀ ≈ 2.33 dry basis)

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

"""
    softclamp(x::Number, scale::Number)

Smooth non-negative clamp `scale·softplus(x/scale)`: ≈ `max(0, x)` for
|x| ≫ scale, smooth near 0. Overflow-safe and free of Julia-level branching
(`max`/`abs` are scalar AD primitives), so it is safe under AD
compiled tape and under Enzyme.

# Arguments
- `x::Number`: Value to clamp.
- `scale::Number`: Smoothing scale; sets the width of the soft corner at 0.

# Returns
- `y::Number`: Smoothly clamped value, ≥ 0.
"""
function softclamp(x, scale)
    z = x / scale
    return scale * (max(z, zero(z)) + log1p(exp(-abs(z))))
end
