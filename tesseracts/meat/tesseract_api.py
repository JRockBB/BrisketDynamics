"""Meat Tesseract: 1D heat-moisture brisket PDE (Julia, SciMLSensitivity adjoint).

Wraps BrisketDynamics.jl's `meat_solve` / `meat_vjp`. The VJP endpoint is backed
by a continuous adjoint (InterpolatingAdjoint + compiled ReverseDiff tape) of
the stiff method-of-lines solve — to our knowledge the first public
SciMLSensitivity adjoint exposed through the Tesseract VJP interface.

Julia bridge pattern follows pasteurlabs/mosaic `topopt-jl`: synchronous
module-level init guarded by a cross-process file lock (uvicorn worker
healthchecks SIGKILL slow async inits).
"""

import os
import tempfile
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field

from tesseract_core.runtime import Array, Differentiable, Float64, Int32, ShapeDType

REPO_ROOT = Path(__file__).resolve().parents[2]

os.environ.setdefault("PYTHON_JULIACALL_HANDLE_SIGNALS", "yes")
os.environ.setdefault("PYTHON_JULIACALL_THREADS", "1")

import filelock

with filelock.FileLock(Path(tempfile.gettempdir()) / "brisket_julia_init.lock", timeout=600):
    import juliacall

    jl = juliacall.newmodule("MeatTess")
    jl.seval(f'import Pkg; Pkg.activate(raw"{REPO_ROOT}"; io=devnull)')
    jl.seval("using BrisketDynamics")
    jl.seval(
        """
        function _meat_apply(θ, knots, tknots, tsave, abstol, reltol)
            o = meat_solve(θ, knots, tknots, tsave; abstol=abstol, reltol=reltol)
            return o.Tsurf, o.Tcore, o.j_evap, o.q_meat, o.C_field_final
        end
        function _meat_vjp(θ, knots, tknots, tsave, abstol, reltol,
                           cTs, cTc, cj, cq, cC)
            cot = (Tsurf=cTs, Tcore=cTc, j_evap=cj, q_meat=cq, C_field_final=cC)
            g = meat_vjp(θ, knots, tknots, tsave, cot; abstol=abstol, reltol=reltol)
            return g.dθ, g.dTpit_knots
        end
        """
    )

NX = 40  # spatial nodes (MeatConfig default; c_field_final length)


def _jvec(x) -> "juliacall.VectorValue":
    return juliacall.convert(jl.Vector, np.ascontiguousarray(np.asarray(x, dtype=np.float64)))


class InputSchema(BaseModel):
    theta: Differentiable[Array[(5,), Float64]] = Field(
        description="Meat parameters [D_eff m²/s, k_m m/s, h_c W/m²K, L m, k_mult]"
    )
    tpit_knots: Differentiable[Array[(None,), Float64]] = Field(
        description="Pit temperature trajectory [K] on the uniform knot grid `tknots`"
    )
    tknots: Array[(None,), Float64] = Field(
        description="Uniform time knot grid [s] for tpit_knots"
    )
    tsave: Array[(None,), Float64] = Field(description="Output save times [s]")
    abstol: float = Field(default=1e-6, description="ODE solver absolute tolerance")
    reltol: float = Field(default=1e-6, description="ODE solver relative tolerance")


class OutputSchema(BaseModel):
    tsurf: Differentiable[Array[(None,), Float64]] = Field(
        description="Surface temperature [K] at tsave"
    )
    tcore: Differentiable[Array[(None,), Float64]] = Field(
        description="Center (probe) temperature [K] at tsave"
    )
    j_evap: Differentiable[Array[(None,), Float64]] = Field(
        description="Evaporative mass flux [kg/m²/s] at tsave"
    )
    q_meat: Differentiable[Array[(None,), Float64]] = Field(
        description="Total heat drawn from the pit [W] at tsave (coupling output)"
    )
    c_field_final: Differentiable[Array[(NX,), Float64]] = Field(
        description="Final collagen conversion field (center → surface)"
    )


def apply(inputs: InputSchema) -> OutputSchema:
    ts, tc, j, q, c = jl._meat_apply(
        _jvec(inputs.theta),
        _jvec(inputs.tpit_knots),
        _jvec(inputs.tknots),
        _jvec(inputs.tsave),
        float(inputs.abstol),
        float(inputs.reltol),
    )
    return OutputSchema(
        tsurf=np.asarray(ts),
        tcore=np.asarray(tc),
        j_evap=np.asarray(j),
        q_meat=np.asarray(q),
        c_field_final=np.asarray(c),
    )


def abstract_eval(abstract_inputs):
    nt = abstract_inputs.tsave.shape[0]
    vec = lambda n: ShapeDType(shape=(n,), dtype="float64")
    return {
        "tsurf": vec(nt),
        "tcore": vec(nt),
        "j_evap": vec(nt),
        "q_meat": vec(nt),
        "c_field_final": vec(NX),
    }


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    nt = len(np.asarray(inputs.tsave))

    def cot(name, n):
        if name in cotangent_vector and name in vjp_outputs:
            return _jvec(cotangent_vector[name])
        return _jvec(np.zeros(n))

    dtheta, dknots = jl._meat_vjp(
        _jvec(inputs.theta),
        _jvec(inputs.tpit_knots),
        _jvec(inputs.tknots),
        _jvec(inputs.tsave),
        float(inputs.abstol),
        float(inputs.reltol),
        cot("tsurf", nt),
        cot("tcore", nt),
        cot("j_evap", nt),
        cot("q_meat", nt),
        cot("c_field_final", NX),
    )
    out = {}
    if "theta" in vjp_inputs:
        out["theta"] = np.asarray(dtheta)
    if "tpit_knots" in vjp_inputs:
        out["tpit_knots"] = np.asarray(dknots)
    return out
