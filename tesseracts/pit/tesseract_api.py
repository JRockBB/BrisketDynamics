"""Pit Tesseract: lumped-parameter smoker ODE (Julia, ForwardDiff Jacobian VJP).

Wraps BrisketDynamics.jl's `pit_solve` / `pit_vjp`. Deliberately uses a
*different* differentiation strategy than the meat Tesseract: the 2-state
non-stiff ODE is differentiated with a chunked ForwardDiff dense Jacobian and
the VJP endpoint returns Jᵀv̄ — cheap and exact for a small system. The
composed pipeline is therefore a genuine continuous-adjoint + forward-mode
composition meeting only at the Tesseract VJP boundary.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType

REPO_ROOT = Path(__file__).resolve().parents[2]

os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

import filelock

with filelock.FileLock(Path(tempfile.gettempdir()) / "brisket_julia_init.lock", timeout=600):
    import juliacall

    jl = juliacall.newmodule("PitTess")
    jl.seval(f'import Pkg; Pkg.activate(raw"{REPO_ROOT}"; io=devnull)')
    jl.seval("using BrisketDynamics")
    jl.seval(
        """
        function _pit_apply(φ, u_knots, u_tknots, q_knots, q_tknots, tsave,
                            abstol, reltol)
            o = pit_solve(φ, u_knots, u_tknots, q_knots, q_tknots, tsave;
                          abstol=abstol, reltol=reltol)
            return o.Tpit, o.m_fuel
        end
        function _pit_vjp(φ, u_knots, u_tknots, q_knots, q_tknots, tsave,
                          abstol, reltol, cT, cm)
            g = pit_vjp(φ, u_knots, u_tknots, q_knots, q_tknots, tsave,
                        (Tpit=cT, m_fuel=cm); abstol=abstol, reltol=reltol)
            return g.dφ, g.du_knots, g.dqmeat_knots
        end
        """
    )


def _jvec(x) -> "juliacall.VectorValue":
    return juliacall.convert(jl.Vector, np.ascontiguousarray(np.asarray(x, dtype=np.float64)))


class InputSchema(BaseModel):
    phi: Differentiable[Array[(5,), Float64]] = Field(
        description="Pit parameters [C_th J/K, UA W/K, r_max kg/s, dH_eff J/kg, u_leak]"
    )
    u_knots: Differentiable[Array[(None,), Float64]] = Field(
        description="Damper schedule in [0,1] on the uniform grid `u_tknots`"
    )
    u_tknots: Array[(None,), Float64] = Field(description="Damper knot times [s]")
    qmeat_knots: Differentiable[Array[(None,), Float64]] = Field(
        description="Meat heat-load trajectory [W] on the uniform grid `q_tknots`"
    )
    q_tknots: Array[(None,), Float64] = Field(description="Heat-load knot times [s]")
    tsave: Array[(None,), Float64] = Field(description="Output save times [s]")
    abstol: float = Field(default=1e-8, description="ODE solver absolute tolerance")
    reltol: float = Field(default=1e-8, description="ODE solver relative tolerance")


class OutputSchema(BaseModel):
    tpit: Differentiable[Array[(None,), Float64]] = Field(
        description="Pit gas temperature [K] at tsave"
    )
    m_fuel: Differentiable[Array[(None,), Float64]] = Field(
        description="Remaining fuel mass [kg] at tsave"
    )


def apply(inputs: InputSchema) -> OutputSchema:
    T, m = jl._pit_apply(
        _jvec(inputs.phi),
        _jvec(inputs.u_knots),
        _jvec(inputs.u_tknots),
        _jvec(inputs.qmeat_knots),
        _jvec(inputs.q_tknots),
        _jvec(inputs.tsave),
        float(inputs.abstol),
        float(inputs.reltol),
    )
    return OutputSchema(tpit=np.asarray(T), m_fuel=np.asarray(m))


def abstract_eval(abstract_inputs):
    nt = abstract_inputs.tsave.shape[0]
    return {
        "tpit": ShapeDType(shape=(nt,), dtype="float64"),
        "m_fuel": ShapeDType(shape=(nt,), dtype="float64"),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    nt = len(np.asarray(inputs.tsave))

    def cot(name):
        if name in cotangent_vector and name in vjp_outputs:
            return _jvec(cotangent_vector[name])
        return _jvec(np.zeros(nt))

    dphi, du, dq = jl._pit_vjp(
        _jvec(inputs.phi),
        _jvec(inputs.u_knots),
        _jvec(inputs.u_tknots),
        _jvec(inputs.qmeat_knots),
        _jvec(inputs.q_tknots),
        _jvec(inputs.tsave),
        float(inputs.abstol),
        float(inputs.reltol),
        cot("tpit"),
        cot("m_fuel"),
    )
    out = {}
    if "phi" in vjp_inputs:
        out["phi"] = np.asarray(dphi)
    if "u_knots" in vjp_inputs:
        out["u_knots"] = np.asarray(du)
    if "qmeat_knots" in vjp_inputs:
        out["qmeat_knots"] = np.asarray(dq)
    return out
