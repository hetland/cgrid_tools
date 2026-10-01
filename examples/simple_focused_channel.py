"""Simple focused channel example.

Generates a rectangular channel grid with xi-direction focus applied near the
midpoint, compares transfinite, Laplacian-smoothed, and elliptic solvers, and
prints scalar orthogonality diagnostics for each.

Run:

    python examples/simple_focused_channel.py

No file output. The script exits with a non-zero code if any solver produces
a grid with mean orthogonality error above a loose quality threshold.
"""

from __future__ import annotations

import pprint
import sys

import numpy as np

import cgrid_tools
from cgrid_tools.gridgen import BoundaryDefinition, Focus, GridGenerator, SolverMode

# Rectangular channel domain (Cartesian units are arbitrary; think meters).
BOUNDARY = BoundaryDefinition(
    x=np.array([0.0, 20.0, 20.0, 0.0]),
    y=np.array([0.0, 0.0, 4.0, 4.0]),
    beta=np.array([1.0, 1.0, 1.0, 1.0]),
)

# Request a moderate grid size in ROMS (Mp+1, Lp+1) convention.
SHAPE = (21, 81)  # -> Mp=20 rho rows, Lp=80 rho columns

# Cluster nodes in the middle of the channel along xi (the long axis).
FOCUS = Focus()
FOCUS.add_focus(pos=0.5, axis="x", factor=3.0, extent=0.1)

# Loose quality threshold: flag a regression if mean orthogonality error
# exceeds 5 degrees on any solver. This is not a solver-quality target.
QUALITY_GATE_RAD = np.deg2rad(5.0)


def _report(name: str, grid) -> bool:
    summary = cgrid_tools.orthogonality_summary(grid.x, grid.y)
    mean_deg = np.degrees(summary["mean_abs"])
    max_deg = np.degrees(summary["max_abs"])
    rms_deg = np.degrees(summary["rms"])
    print(
        f"  {name:12s} | mean {mean_deg:6.3f} deg | "
        f"max {max_deg:6.3f} deg | rms {rms_deg:6.3f} deg | "
        f"iters {grid.options.get('iterations', 0)}"
    )
    return summary["mean_abs"] <= QUALITY_GATE_RAD


def main() -> int:
    print(f"Boundary: {len(BOUNDARY.x) - 1} vertices (closed)")
    print(f"Grid shape: {SHAPE} (ROMS Mp+1, Lp+1)")
    print("Focus: xi pos=0.5, factor=3.0, extent=0.1")
    print()
    print("Solver       | mean_abs | max_abs  | rms      | iterations")
    print("-" * 70)

    ok = True
    for mode in (SolverMode.TRANSFINITE, SolverMode.LAPLACIAN, SolverMode.ELLIPTIC):
        grid = GridGenerator(
            mode=mode,
            focus=FOCUS,
            smoothing_iterations=200,
            elliptic_relaxation=1.3,
            elliptic_tolerance=1.0e-8,
        ).generate(BOUNDARY, shape=SHAPE)
        ok &= _report(mode.value, grid)

    # Also report the derived ROMS metric shapes for the final result.
    x_rho, _ = cgrid_tools.rho_points(grid.x, grid.y)
    x_psi, _ = cgrid_tools.psi_points(grid.x, grid.y)
    print()
    print("Final elliptic grid metrics (shapes):")
    print(f"  x_rho : {x_rho.shape}")
    print(f"  x_psi : {x_psi.shape}")
    print(f"  dx    : {grid.dx.shape}  (range {grid.dx.min():.4f} to {grid.dx.max():.4f})")
    print(f"  dy    : {grid.dy.shape}  (range {grid.dy.min():.4f} to {grid.dy.max():.4f})")
    print(f"  pm    : {grid.pm.shape}  (range {grid.pm.min():.4f} to {grid.pm.max():.4f})")

    print("\nGrid options (provenance):")
    pprint.pprint(grid.options)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
