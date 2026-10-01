from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from cgrid_tools.gridgen import (
    BoundaryDefinition,
    CRDTOptions,
    GridGenerator,
    SolverMode,
    solve_crdt_state,
)

FIXTURES = Path(__file__).parent / "fixtures" / "gridgen_c"


def test_crowded_polygon_sigma_solve_with_higher_quadrature():
    # MINPACK's machine-epsilon finite differences stagnated on this polygon
    # with 16 quadrature nodes even when 10 nodes happened to converge.
    boundary = _boundary_from_xy(FIXTURES / "xy.3")
    state = solve_crdt_state(boundary, CRDTOptions(nnodes=16, precision=1e-12))
    assert state.residual_norm < 1e-8


def _parse_xy(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    xs: list[float] = []
    ys: list[float] = []
    betas: list[float] = []
    start_index: int | None = None
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        token = line.split()
        if len(token) < 2:
            continue
        x = float(token[0])
        y = float(token[1])
        beta = 0.0
        star = False
        if len(token) >= 3:
            marker = token[2]
            if marker.endswith("*"):
                star = True
                marker = marker[:-1]
            if marker:
                beta = float(marker)
        xs.append(x)
        ys.append(y)
        betas.append(beta)
        if star:
            if start_index is not None:
                raise ValueError(f"{path}: multiple '*' start markers")
            start_index = len(xs) - 1
    if start_index is None:
        start_index = 0
    return (
        np.asarray(xs, dtype=float),
        np.asarray(ys, dtype=float),
        np.asarray(betas, dtype=float),
        start_index,
    )


def _parse_prm(path: Path) -> dict[str, str]:
    params: dict[str, str] = {}
    for line in path.read_text().splitlines():
        parts = line.split(None, 1)
        if len(parts) == 2:
            params[parts[0]] = parts[1].strip()
    return params


def _parse_grid(path: Path, shape: tuple[int, int]) -> np.ndarray:
    ny, nx = shape
    text = path.read_text().splitlines()
    header = text[0]
    match = re.match(r"##\s*(\d+)\s*x\s*(\d+)", header)
    if not match:
        raise ValueError(f"{path}: unrecognized header {header!r}")
    body = text[1:]
    if len(body) != ny * nx:
        raise ValueError(f"{path}: body has {len(body)} entries but prm shape {ny}x{nx}={ny * nx}")
    xs = np.full(ny * nx, np.nan)
    ys = np.full(ny * nx, np.nan)
    for i, line in enumerate(body):
        token = line.split()
        if not token or token[0] == "NaN":
            continue
        xs[i] = float(token[0])
        ys[i] = float(token[1])
    x = xs.reshape(ny, nx)
    y = ys.reshape(ny, nx)
    return np.stack([x, y], axis=-1)


def _rotate_to_start(
    x: np.ndarray,
    y: np.ndarray,
    beta: np.ndarray,
    start: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.roll(x, -start)
    y = np.roll(y, -start)
    beta = np.roll(beta, -start)
    return x, y, beta


def _boundary_from_xy(path: Path) -> BoundaryDefinition:
    x, y, beta, start = _parse_xy(path)
    x, y, beta = _rotate_to_start(x, y, beta, start)
    return BoundaryDefinition(x=x, y=y, beta=beta)


def _candidate_layouts(x: np.ndarray, y: np.ndarray) -> list[tuple[np.ndarray, np.ndarray, str]]:
    layouts: list[tuple[np.ndarray, np.ndarray, str]] = []
    for axis_flip, flip_label in ((None, ""), (0, "flipH"), (1, "flipV"), ((0, 1), "flip2")):
        if axis_flip is None:
            xf, yf = x, y
        else:
            xf = np.flip(x, axis=axis_flip)
            yf = np.flip(y, axis=axis_flip)
        for transpose, t_label in ((False, ""), (True, "T")):
            if transpose:
                xt, yt = xf.T, yf.T
            else:
                xt, yt = xf, yf
            label = (flip_label + t_label) or "id"
            layouts.append((xt, yt, label))
    return layouts


def _alignment_error(
    ours: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, float, float, str, np.ndarray, np.ndarray]:
    best: tuple[float, float, float, str, np.ndarray, np.ndarray] | None = None
    for xf, yf, label in _candidate_layouts(ours[..., 0], ours[..., 1]):
        if xf.shape != reference.shape[:2]:
            continue
        both_finite = (
            np.isfinite(xf)
            & np.isfinite(yf)
            & np.isfinite(reference[..., 0])
            & np.isfinite(reference[..., 1])
        )
        if not np.any(both_finite):
            continue
        dx = xf - reference[..., 0]
        dy = yf - reference[..., 1]
        dist = np.hypot(dx, dy)
        dist_finite = dist[both_finite]
        mean = float(np.mean(dist_finite))
        rms = float(np.sqrt(np.mean(dist_finite**2)))
        dmax = float(np.max(dist_finite))
        candidate = (mean, rms, dmax, label, xf, yf)
        if best is None or mean < best[0]:
            best = candidate
    if best is None:
        raise AssertionError("no alignment candidate produced finite overlap with reference")
    return best


def _orthogonality_deg(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    from cgrid_tools import orthogonality_error

    err = orthogonality_error(x, y)
    finite = err[np.isfinite(err)]
    if finite.size == 0:
        return {"mean": float("nan"), "max": float("nan"), "rms": float("nan")}
    return {
        "mean": float(np.degrees(np.mean(np.abs(finite)))),
        "max": float(np.degrees(np.max(np.abs(finite)))),
        "rms": float(np.degrees(np.sqrt(np.mean(finite**2)))),
    }


EXAMPLES = [
    pytest.param(
        "1",
        {
            "vertex_mean": 0.01,
            "vertex_max": 0.05,
            "ortho_mean_abs_diff_deg": 0.5,
            "ortho_max_abs_diff_deg": 1.0,
        },
        id="estuary_single_channel",
    ),
    pytest.param(
        "2",
        {
            "vertex_mean": 0.01,
            "vertex_max": 0.05,
            "ortho_mean_abs_diff_deg": 0.5,
            "ortho_max_abs_diff_deg": 1.0,
        },
        id="estuary_moved_turning_points",
    ),
    pytest.param(
        "3",
        {
            "vertex_mean": 6.0,
            "vertex_max": 15.0,
            "ortho_mean_abs_diff_deg": 6.0,
            "ortho_max_abs_diff_deg": 55.0,
        },
        id="crdt_paper_shape_weaker_parity",
    ),
    pytest.param(
        "4",
        {
            "vertex_mean": 0.005,
            "vertex_max": 0.01,
            "ortho_mean_abs_diff_deg": 0.5,
            "ortho_max_abs_diff_deg": 1.0,
        },
        id="circle_non_right_angles",
    ),
]


@pytest.mark.parametrize("example_id,tolerances", EXAMPLES)
def test_reproduces_gridgen_c_example(example_id: str, tolerances: dict[str, float]) -> None:
    xy_path = FIXTURES / f"xy.{example_id}"
    prm_path = FIXTURES / f"prm.{example_id}"
    grid_path = FIXTURES / f"grid.{example_id}"

    boundary = _boundary_from_xy(xy_path)
    prm = _parse_prm(prm_path)
    nx = int(prm["nx"])
    ny = int(prm["ny"])
    shape = (ny, nx)

    options = CRDTOptions(
        nnodes=int(prm.get("nnodes", "10")),
        precision=float(prm.get("precision", "1.0e-12")),
        newton=bool(int(prm.get("newton", "1"))),
    )

    result = GridGenerator(mode=SolverMode.GRIDGEN_CRDT, crdt_options=options).generate(
        boundary, shape=shape
    )

    ours = np.stack([result.x, result.y], axis=-1)
    reference = _parse_grid(grid_path, shape)

    assert (
        ours.shape == reference.shape
    ), f"example {example_id}: our grid shape {ours.shape} != reference {reference.shape}"

    mean, rms, dmax, label, aligned_x, aligned_y = _alignment_error(ours, reference)

    # Small coordinate error on a tiny surviving subset is not parity. The
    # complex third fixture has a different image mask; bound that difference
    # explicitly. The other fixtures must preserve the exact finite-node mask.
    our_finite = np.isfinite(aligned_x) & np.isfinite(aligned_y)
    ref_finite = np.isfinite(reference).all(axis=-1)
    if example_id == "3":
        overlap = np.count_nonzero(our_finite & ref_finite)
        union = np.count_nonzero(our_finite | ref_finite)
        assert overlap / union >= 0.96
    else:
        np.testing.assert_array_equal(our_finite, ref_finite)

    from cgrid_tools import cell_jacobians

    assert np.all(cell_jacobians(result.x, result.y)[:, result.mask_rho] > 0)

    ref_ortho = _orthogonality_deg(reference[..., 0], reference[..., 1])
    our_ortho = _orthogonality_deg(aligned_x, aligned_y)
    ortho_mean_diff = abs(our_ortho["mean"] - ref_ortho["mean"])
    ortho_max_diff = abs(our_ortho["max"] - ref_ortho["max"])

    diag = (
        f"example {example_id} [{label}]: "
        f"vertex mean={mean:.4g} rms={rms:.4g} max={dmax:.4g}; "
        f"ortho mean (ours/ref)={our_ortho['mean']:.3f}/{ref_ortho['mean']:.3f} deg "
        f"(|diff|={ortho_mean_diff:.3f}); "
        f"ortho max (ours/ref)={our_ortho['max']:.3f}/{ref_ortho['max']:.3f} deg "
        f"(|diff|={ortho_max_diff:.3f})"
    )

    assert mean <= tolerances["vertex_mean"], f"vertex mean too large; {diag}"
    assert dmax <= tolerances["vertex_max"], f"vertex max too large; {diag}"
    assert (
        ortho_mean_diff <= tolerances["ortho_mean_abs_diff_deg"]
    ), f"orthogonality mean drift too large; {diag}"
    assert (
        ortho_max_diff <= tolerances["ortho_max_abs_diff_deg"]
    ), f"orthogonality max drift too large; {diag}"
