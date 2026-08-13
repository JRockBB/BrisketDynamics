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

const K_MEAT = 0.45           # baseline thermal conductivity [W m⁻¹ K⁻¹]
const RHO_MEAT = 1060.0       # density [kg m⁻³]
const CP_MEAT = 3400.0        # specific heat capacity [J kg⁻¹ K⁻¹]
const RHO_DRY = 320.0         # dry-matter density [kg m⁻³] (ρ/(1+W₀) at W₀ ≈ 2.33 dry basis)

"""
    p_sat(T)

Saturation vapor pressure of water [Pa] at temperature `T` [K], Magnus formula.
Smooth in `T` (valid range ≈ 273–373 K, more than covers cooking temperatures).
"""
p_sat(T) = 610.94 * exp(17.625 * (T - 273.15) / (T - 30.11))

"""
    logistic(x)

Logistic sigmoid, written as `tanh` for numerical stability at large |x|.
"""
logistic(x) = (1 + tanh(x / 2)) / 2

"""
    softclamp(x, scale)

Smooth non-negative clamp `scale·softplus(x/scale)`: ≈ `max(0, x)` for
|x| ≫ scale, smooth near 0. Overflow-safe and free of Julia-level branching
(`max`/`abs` are scalar AD primitives), so it is safe inside a ReverseDiff
compiled tape.
"""
function softclamp(x, scale)
    z = x / scale
    scale * (max(z, zero(z)) + log1p(exp(-abs(z))))
end
