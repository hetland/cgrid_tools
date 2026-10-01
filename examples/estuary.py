"""Estuary-style multi-segment domain.

A synthetic estuary boundary: a wider seaward basin tapering into a narrower
upstream channel, with a bend. The boundary has 6 convex corners (each
beta = +1; sum = 4 with two 0 betas as spacers is not used here). This
example is intended to stress-test solver quality on a non-trivial but
fully convex polygon that is not well approximated by a single logical
rectangle.

Runs the transfinite, elliptic, and GRIDGEN_CRDT solvers and reports
scalar orthogonality diagnostics. Includes a radial focus target near the
inlet to cluster resolution in the narrow upstream reach.

Run:

    python examples/estuary.py
"""

from __future__ import annotations

import pprint
import sys

import numpy as np

import cgrid_tools
from cgrid_tools.gridgen import BoundaryDefinition, CRDTOptions, Focus, GridGenerator, SolverMode

# Synthetic estuary: a seaward basin (wide) tapers and bends into a narrower
# upstream channel. Eight vertices, four of which are corners (beta=1); the
# other four are waypoints along the shorelines (beta=0) so sum(beta)=4.
#
# (0,0) *-------------------------------* (10,0)
#        \                              |
#         \                             |
#          \  (*3,-2)               (*7,-2)
#           \_____                 ____/
#                 \               /
#                  (*4,-3) . (*6,-3)
#                   |      |
#                   |______|
#
#  Simpler variant: 6 vertices, 4 corners with beta=1 each, 2 waypoints
#  with beta=0.
BOUNDARY = BoundaryDefinition(
    x=np.array([0.0, 10.0, 8.0, 6.0, 4.0, 2.0]),
    y=np.array([0.0, 0.0, -3.0, -4.5, -4.5, -3.0]),
    beta=np.array([1.0, 1.0, 0.0, 1.0, 1.0, 0.0]),
)

SHAPE = (33, 49)  # (Mp+1, Lp+1)

# Cluster resolution toward the downstream (narrow) end along eta.
FOCUS = Focus()
FOCUS.add_focus(pos=0.85, axis="y", factor=2.5, extent=0.2)

# Loose per-solver gates. This estuary has tapered walls, so transfinite
# blending from opposite sides produces noticeably sheared cells;
# elliptic smoothing reduces but does not eliminate the shear. CRDT
# handles the taper cleanly. Gates are set to document expected behavior,
# not enforce solver parity.
PER_SOLVER_GATE_DEG = {
    SolverMode.TRANSFINITE.value: 35.0,
    SolverMode.ELLIPTIC.value: 30.0,
    SolverMode.GRIDGEN_CRDT.value: 5.0,
}


def _report(name: str, grid) -> bool:
    summary = cgrid_tools.orthogonality_summary(grid.x, grid.y)
    mean_deg = np.degrees(summary["mean_abs"])
    max_deg = np.degrees(summary["max_abs"])
    rms_deg = np.degrees(summary["rms"])
    active = int(np.count_nonzero(grid.mask_rho)) if grid.mask_rho is not None else grid.x.size
    gate_deg = PER_SOLVER_GATE_DEG.get(name, 90.0)
    flag = "" if mean_deg <= gate_deg else "  <-- mean exceeds gate"
    print(
        f"  {name:14s} | mean {mean_deg:6.3f} deg | "
        f"max {max_deg:6.3f} deg | rms {rms_deg:6.3f} deg | "
        f"active {active:5d}{flag}"
    )
    return mean_deg <= gate_deg


def main() -> int:
    print(f"Boundary: {len(BOUNDARY.x) - 1} vertices (closed)")
    print(f"sum(beta) = {BOUNDARY.beta.sum():.1f}")
    print(f"Grid shape: {SHAPE}")
    print("Focus: eta pos=0.85, factor=2.5, extent=0.2")
    print()
    print("Solver         | mean_abs | max_abs  | rms      | active cells")
    print("-" * 72)

    ok = True

    tr = GridGenerator(mode=SolverMode.TRANSFINITE, focus=FOCUS).generate(BOUNDARY, shape=SHAPE)
    ok &= _report("transfinite", tr)

    el = GridGenerator(
        mode=SolverMode.ELLIPTIC,
        focus=FOCUS,
        smoothing_iterations=400,
        elliptic_relaxation=1.3,
        elliptic_tolerance=1.0e-9,
    ).generate(BOUNDARY, shape=SHAPE)
    ok &= _report("elliptic", el)

    crdt = GridGenerator(
        mode=SolverMode.GRIDGEN_CRDT,
        focus=FOCUS,
        crdt_options=CRDTOptions(),
    ).generate(BOUNDARY, shape=SHAPE)
    ok &= _report("gridgen_crdt", crdt)

    print()
    print("Elliptic grid pm/pn range (with focus):")
    print(f"  pm: {el.pm.min():.3f} to {el.pm.max():.3f}")
    print(f"  pn: {el.pn.min():.3f} to {el.pn.max():.3f}")

    print("\nGrid options (provenance):")
    pprint.pprint(crdt.options)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
