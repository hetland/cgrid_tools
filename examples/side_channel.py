"""Side-channel example with a beta=-1 reentrant corner pair.

Demonstrates grid generation on a domain similar to the side-channel tutorial
from the ``pygridgen`` examples. The boundary forms a vertical main channel
with a short lateral notch on the right flank. The notch is encoded with two
``beta = -1`` corners.

Uses GRIDGEN_CRDT and prints orthogonality and mapping diagnostics.
Four-side solvers reject this nonrectangular logical domain.

Run:

    python examples/side_channel.py
"""

from __future__ import annotations

import pprint
import sys

import numpy as np

import cgrid_tools
from cgrid_tools.gridgen import BoundaryDefinition, CRDTOptions, GridGenerator, SolverMode

# Domain: rectangular bay with an attached side channel on the east wall.
# Two beta=-1 corners mark the entrance and exit of the side channel.
#
# Vertex walk (counter-clockwise):
#   (0.5, 0.5) -> (2.0, 0.5)    south edge of bay
#   (2.0, 0.5) -> (2.0, 1.75)   south entry jog (into side channel)
#   (2.0, 1.75) -> (3.5, 1.75)  south wall of side channel
#   (3.5, 1.75) -> (3.5, 2.25)  east cap of side channel
#   (3.5, 2.25) -> (2.0, 2.25)  north wall of side channel
#   (2.0, 2.25) -> (2.0, 3.5)   north jog (back into bay)
#   (2.0, 3.5) -> (0.5, 3.5)    north edge of bay
#   (0.5, 3.5) -> (0.5, 0.5)    west edge of bay
BOUNDARY = BoundaryDefinition(
    x=np.array([0.5, 2.0, 2.0, 3.5, 3.5, 2.0, 2.0, 0.5]),
    y=np.array([0.5, 0.5, 1.75, 1.75, 2.25, 2.25, 3.5, 3.5]),
    beta=np.array([1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, 1.0]),
)

SHAPE = (33, 17)  # (Mp+1, Lp+1)

QUALITY_GATE_DEG = 3.0


def _report(name: str, grid) -> bool:
    summary = cgrid_tools.orthogonality_summary(grid.x, grid.y)
    mean_deg = np.degrees(summary["mean_abs"])
    max_deg = np.degrees(summary["max_abs"])
    rms_deg = np.degrees(summary["rms"])
    gate_deg = QUALITY_GATE_DEG
    flag = "" if mean_deg <= gate_deg else "  <-- mean exceeds gate"
    print(
        f"  {name:14s} | mean {mean_deg:6.3f} deg | "
        f"max {max_deg:6.3f} deg | rms {rms_deg:6.3f} deg | "
        f"iters {grid.options.get('iterations', 0)}{flag}"
    )
    return mean_deg <= gate_deg


def main() -> int:
    print(f"Boundary: side-channel, {len(BOUNDARY.x) - 1} vertices")
    print(f"Grid shape: {SHAPE}")
    print()
    print("Solver         | mean_abs | max_abs  | rms      | iterations")
    print("-" * 72)

    ok = True

    crdt = GridGenerator(
        mode=SolverMode.GRIDGEN_CRDT,
        crdt_options=CRDTOptions(),
    ).generate(BOUNDARY, shape=SHAPE)
    ok &= _report("gridgen_crdt", crdt)

    # Masked-cell accounting: the side channel excludes the notch block from
    # the rho mask. Report how many cells are active.
    total_cells = crdt.mask_rho.size
    active_cells = int(np.count_nonzero(crdt.mask_rho))
    print()
    print(f"CRDT active rho cells: {active_cells} of {total_cells}")

    print("\nGrid options (provenance):")
    pprint.pprint(crdt.options)

    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
