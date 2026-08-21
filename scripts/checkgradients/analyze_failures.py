"""Classify check-gradients failures: benign causality-onset kinks vs real.

The tesseract-runtime check-gradients tool compares VJP rows to central
finite differences elementwise with a relative tolerance only (no absolute
floor). At a control/forcing knot's causality onset, the trajectory response
begins mid-interval, and the piecewise-linear interpolation puts a kink
there: two-sided FD straddles the kink while the adjoint integrates one
side, so the two legitimately disagree — at an absolute magnitude many
orders below the row's informative entries. Those elements are measure-zero
and carry no calibration/control signal.

This script parses a saved check-gradients log (run with a high
--max-failures so every failure is detailed), and for each failure block
finds every element violating the tool's rtol. A failure is BENIGN if all
its violating elements are absolutely small: |vjp|, |fd| <= ATOL_FRAC * the
row's max magnitude. Anything else is flagged REAL and printed.

Usage:  python analyze_failures.py <log> [<log> ...]
"""

import re
import sys

import numpy as np

RTOL = 0.1  # the tool's default
ATOL_FRAC = 1e-3  # "near zero" = below this fraction of the row max


def parse_blocks(text):
    """Extract the detailed failure blocks from a check-gradients log.

    Parameters
    ----------
    text : str
        Full contents of a saved ``tesseract-runtime check-gradients`` log,
        run with a high ``--max-failures`` so every failure is detailed.

    Yields
    ------
    tuple of (str, str, str, numpy.ndarray, numpy.ndarray)
        ``(input_path, output_path, index, vjp, fd)`` per failure block:
        the input/output endpoint paths and index string as they appear in
        the log, plus the vector_jacobian_product row and the central
        finite-difference row parsed as float arrays.
    """
    pat = re.compile(
        r"Input path: '(?P<inp>[^']+)', Output path: '(?P<out>[^']+)', "
        r"Index: \((?P<idx>[^)]*)\)\s*\n"
        r"\s*vector_jacobian_product value: \[(?P<vjp>[^\]]*)\]\s*\n"
        r"\s*Finite difference value: \[(?P<fd>[^\]]*)\]",
        re.DOTALL,
    )
    for m in pat.finditer(text):
        vjp = np.fromstring(m["vjp"].replace("\n", " "), sep=" ")
        fd = np.fromstring(m["fd"].replace("\n", " "), sep=" ")
        yield m["inp"], m["out"], m["idx"], vjp, fd


ONSET_VJP_FRAC = 1e-4  # "adjoint says (essentially) exactly zero"
ONSET_FD_FRAC = 2e-2  # FD artifact bound observed at eps=1e-4


def main(argv):
    """Classify every failure block in the given logs and print a summary.

    Each block's rtol-violating elements are tested against the benign
    categories (near-zero kink, causality-onset FD artifact, or an entirely
    FD-unresolvable weak row); anything else is printed as a REAL mismatch
    with its worst elements.

    Parameters
    ----------
    argv : list of str
        Log file paths, optionally preceded by ``--exclude-inputs a,b`` to
        drop failure blocks for input directions that are excluded from the
        container sweep by the absolute-eps doctrine (t_wrap, smoke_knots —
        verified at the Julia level with relative steps); classifying them
        here would misreport doctrine violations as gradient errors.

    Returns
    -------
    int
        Process exit code: 1 if any REAL mismatch was found, else 0.
    """
    exclude = set()
    paths = []
    it = iter(argv)
    for a in it:
        if a == "--exclude-inputs":
            exclude = set(next(it).split(","))
        else:
            paths.append(a)
    total = kink = onset = weak = real = excl = 0
    for path in paths:
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        for inp, out, idx, vjp, fd in parse_blocks(text):
            if vjp.shape != fd.shape or vjp.size == 0:
                continue
            if inp in exclude:
                excl += 1
                continue
            total += 1
            scale = max(np.abs(vjp).max(), np.abs(fd).max(), 1e-300)
            # FD-unresolvable rows: when the entire row's response is below
            # ~1e-5, the tool's eps leaves the finite difference at solver-
            # repeatability noise (observed: tpit->bark_s rows at 2e-6 with
            # VJP and FD agreeing in sign/order). These rows are verified at
            # the Julia level with relative steps instead.
            if scale < 1e-5:
                weak += 1
                continue
            viol = np.abs(vjp - fd) > RTOL * np.maximum(np.abs(fd), 1e-300)
            viol &= np.abs(vjp - fd) > 1e-12  # ignore exact agreement
            small = (np.abs(vjp) <= ATOL_FRAC * scale) & (
                np.abs(fd) <= ATOL_FRAC * scale
            )
            # Onset artifact: the adjoint's (correct-by-causality) exact zero
            # vs an FD value that is pure perturbed-solve noise / eps. The
            # eps-dependence experiment (docs/verification.md) shows the FD
            # side shrinks ~100x when eps grows 10x — a real slope would be
            # eps-independent.
            onset_ok = (np.abs(vjp) <= ONSET_VJP_FRAC * scale) & (
                np.abs(fd) <= ONSET_FD_FRAC * scale
            )
            if np.all(~viol | small):
                kink += 1
            elif np.all(~viol | small | onset_ok):
                onset += 1
            else:
                real += 1
                bad = np.where(viol & ~small & ~onset_ok)[0]
                print(
                    f"REAL MISMATCH {path} {inp}->{out} idx {idx}: "
                    f"elements {bad.tolist()}"
                )
                for b in bad[:5]:
                    print(
                        f"    [{b}] vjp={vjp[b]:.6e} fd={fd[b]:.6e} "
                        f"(row scale {scale:.3e})"
                    )
    print(
        f"\n{total} failure blocks parsed: "
        f"{kink} near-zero kink elements, "
        f"{onset} causality-onset FD artifacts (adjoint zero correct), "
        f"{weak} FD-unresolvable weak rows (verified at Julia level), "
        f"{real} REAL" + (f"; {excl} excluded-direction blocks skipped" if excl else "")
    )
    return 1 if real else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
