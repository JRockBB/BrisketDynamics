"""Minimal Tesseract for the tesseract-jax partial-tangent repro.

Two differentiable inputs of different shapes, one scalar-ish output:
    y = sum(a**2) + sum(3.0 * b)
so the correct gradients are trivially  dy/da = 2a,  dy/db = 3.

No external dependencies beyond numpy/pydantic (pure Python, no solver).
"""

import numpy as np
from pydantic import BaseModel
from tesseract_core.runtime import Array, Differentiable, Float64, ShapeDType


class InputSchema(BaseModel):
    a: Differentiable[Array[(3,), Float64]]
    # non-differentiable input BETWEEN the differentiable ones, as in any
    # real tesseract (grids, tolerances, ...)
    knots: Array[(4,), Float64]
    b: Differentiable[Array[(2,), Float64]]


class OutputSchema(BaseModel):
    y: Differentiable[Array[(), Float64]]
    z: Differentiable[Array[(3,), Float64]]


def apply(inputs: InputSchema) -> OutputSchema:
    a = np.asarray(inputs.a, dtype=np.float64)
    b = np.asarray(inputs.b, dtype=np.float64)
    k = np.asarray(inputs.knots, dtype=np.float64)
    return OutputSchema(y=np.float64(np.sum(a**2) + np.sum(3.0 * b) + k.sum()),
                        z=a * np.sum(b))


def abstract_eval(abstract_inputs):
    return {"y": ShapeDType(shape=(), dtype="float64"),
            "z": ShapeDType(shape=(3,), dtype="float64")}


def vector_jacobian_product(
    inputs: InputSchema,
    vjp_inputs: set[str],
    vjp_outputs: set[str],
    cotangent_vector: dict,
):
    a = np.asarray(inputs.a, dtype=np.float64)
    b = np.asarray(inputs.b, dtype=np.float64)
    ybar = (float(np.asarray(cotangent_vector["y"]))
            if "y" in cotangent_vector else 0.0)
    zbar = (np.asarray(cotangent_vector["z"], dtype=np.float64)
            if "z" in cotangent_vector else np.zeros(3))
    out = {}
    if "a" in vjp_inputs:
        out["a"] = ybar * 2.0 * a + zbar * np.sum(b)
    if "b" in vjp_inputs:
        out["b"] = ybar * 3.0 * np.ones(2) + float(np.dot(zbar, a)) * np.ones(2)
    return out
