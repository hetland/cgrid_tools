from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from numba import njit
from scipy.optimize import least_squares, root
from scipy.special import roots_jacobi

from ._validation import grid_shape, validate_cells
from .boundary import BoundaryDefinition
from .focus import Focus
from .geometry import edge_key as _edge_key
from .geometry import points_in_polygon as _points_in_polygon
from .geometry import signed_area as _signed_area
from .triangulation import prepare_crdt_topology


@njit(cache=True)
def _sc_prod_numba(
    w: complex,
    vertex_index: int,
    prevertices: np.ndarray,
    betas: np.ndarray,
) -> complex:
    total = 0.0 + 0.0j
    for idx in range(len(betas)):
        beta = betas[idx]
        if beta == 0.0:
            continue
        term = 1.0 - w / prevertices[idx]
        term_abs = abs(term)
        if term_abs == 0.0:
            continue
        if idx == vertex_index:
            term /= term_abs
        total += beta * np.log(term)
    return np.exp(total)


@njit(cache=True)
def _sc_sum_numba(
    z0: complex,
    z1: complex,
    vertex_index: int,
    prevertices: np.ndarray,
    betas: np.ndarray,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> complex:
    zh = (z1 - z0) / 2.0
    zc = (z1 + z0) / 2.0
    value = 0.0 + 0.0j
    for idx in range(len(nodes)):
        value += weights[idx] * _sc_prod_numba(
            zc + zh * nodes[idx],
            vertex_index,
            prevertices,
            betas,
        )
    value *= zh
    if abs(zh) != 0.0 and vertex_index >= 0:
        value *= abs(zh) ** betas[vertex_index]
    return value


class CRDTError(RuntimeError):
    """Raised when the CRDT backend cannot generate a grid."""


@dataclass(frozen=True)
class CRDTOptions:
    """CRDT quadrature and solver settings; precision is a residual target."""

    nnodes: int = 10
    precision: float = 1.0e-10
    max_iterations: int = 100
    newton: bool = True
    nppe: int = 3
    image_beta: np.ndarray | None = None
    image_beta_blend: float | None = None

    def __post_init__(self) -> None:
        for name, minimum in (("nnodes", 2), ("max_iterations", 1), ("nppe", 0)):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} must be an integer >= {minimum}")
        if not np.isfinite(self.precision) or self.precision <= 0:
            raise ValueError("precision must be finite and positive")


@dataclass(frozen=True)
class CRDTDiagnostics:
    sigma_residual_norm: float
    pre_alignment_vertex_mismatch: float
    alignment_displacement: float
    post_alignment_vertex_mismatch: float
    physical_corner_angles: np.ndarray
    image_corner_angles: np.ndarray
    corner_angle_ratios: np.ndarray
    corner_angle_warning: np.ndarray
    image_beta_source: str

    @property
    def image_vertex_mismatch(self) -> float:
        return self.post_alignment_vertex_mismatch


@dataclass
class _MappingStats:
    sc_map_failures: int = 0
    ode_fallbacks: int = 0
    continuation_uses: int = 0
    inverse_evaluations: int = 0
    secant_solves: int = 0
    newton_solves: int = 0
    root_solves: int = 0
    least_squares_solves: int = 0
    candidate_attempts: int = 0
    candidate_nodes: int = 0
    location_fallbacks: int = 0
    max_inverse_residual: float = 0.0
    worst_node_index: tuple[int, int] | None = None
    worst_qid: int | None = None
    worst_image_point: complex | None = None


@dataclass(frozen=True)
class _InverseResult:
    w: complex
    method: str
    residual: float

    @property
    def used_ode(self) -> bool:
        return self.method == "ode"


@dataclass(frozen=True)
class _MappedPoint:
    z: complex
    qid: int
    w: complex
    method: str
    inverse_residual: float
    used_continuation: bool


@dataclass(frozen=True)
class CRDTSolverState:
    topology: object
    zs: np.ndarray
    betas: np.ndarray
    user_betas: np.ndarray
    image_marker_betas: np.ndarray
    corner_vertex_ids: np.ndarray
    cis: np.ndarray
    sigmas: np.ndarray
    residual_norm: float
    physical_sc: SchwarzChristoffelTransform | None
    image_sc: SchwarzChristoffelTransform | None
    ws: np.ndarray | None
    physical_a: np.ndarray | None
    physical_b: np.ndarray | None
    image_a: np.ndarray | None
    image_b: np.ndarray | None
    image_quadrilateral_polygons: tuple[np.ndarray, ...] | None
    image_vertices: np.ndarray
    image_polygon: np.ndarray
    image_extents: tuple[float, float, float, float]
    diagnostics: CRDTDiagnostics | None = None


class SchwarzChristoffelTransform:
    def __init__(self, betas: np.ndarray, nnodes: int = 10) -> None:
        self.betas = np.asarray(betas, dtype=float)
        if self.betas.ndim != 1:
            raise ValueError("betas must be one-dimensional")
        if nnodes < 2:
            raise ValueError("nnodes must be at least 2")
        self.nnodes = nnodes
        self._nodes: list[np.ndarray] = []
        self._weights: list[np.ndarray] = []
        for idx, beta in enumerate(self.betas):
            if beta <= -1.0:
                warnings.warn(
                    f"SchwarzChristoffelTransform: beta[{idx}]={beta:g} is <= -1; "
                    "Gauss-Jacobi quadrature is not defined for this exponent. "
                    "Using zero nodes/weights, which will collapse SC-product "
                    "integrals at this vertex to zero.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                nodes = np.zeros(nnodes, dtype=float)
                weights = np.zeros(nnodes, dtype=float)
            else:
                nodes, weights = roots_jacobi(nnodes, 0.0, beta)
            self._nodes.append(np.asarray(nodes, dtype=float))
            self._weights.append(np.asarray(weights, dtype=float))
        nodes, weights = roots_jacobi(nnodes, 0.0, 0.0)
        self._nodes.append(np.asarray(nodes, dtype=float))
        self._weights.append(np.asarray(weights, dtype=float))

    def w2z(
        self,
        w: complex,
        prevertices: np.ndarray,
        *,
        vertex_index: int = -1,
        w0: complex = 0.0 + 0.0j,
        z0: complex = 0.0 + 0.0j,
        w0_vertex_index: int = -1,
        scale: complex = 1.0 + 0.0j,
    ) -> complex:
        prevertices = np.asarray(prevertices, dtype=complex)
        return z0 + scale * self._integrate(
            complex(w),
            vertex_index,
            complex(w0),
            w0_vertex_index,
            prevertices,
        )

    def z2w(
        self,
        z: complex,
        prevertices: np.ndarray,
        *,
        z1: complex,
        w1: complex,
        z2: complex,
        w2: complex,
        z0: complex = 0.0 + 0.0j,
        w0: complex = 0.0 + 0.0j,
        scale: complex = 1.0 + 0.0j,
        tolerance: float = 1.0e-10,
        max_iterations: int = 100,
        use_ode_fallback: bool = True,
    ) -> complex:
        result = self.z2w_result(
            z,
            prevertices,
            z1=z1,
            w1=w1,
            z2=z2,
            w2=w2,
            z0=z0,
            w0=w0,
            scale=scale,
            tolerance=tolerance,
            max_iterations=max_iterations,
            use_ode_fallback=use_ode_fallback,
        )
        return result.w

    def z2w_with_status(
        self,
        z: complex,
        prevertices: np.ndarray,
        *,
        z1: complex,
        w1: complex,
        z2: complex,
        w2: complex,
        z0: complex = 0.0 + 0.0j,
        w0: complex = 0.0 + 0.0j,
        scale: complex = 1.0 + 0.0j,
        tolerance: float = 1.0e-10,
        max_iterations: int = 100,
        use_ode_fallback: bool = True,
    ) -> tuple[complex, bool]:
        result = self.z2w_result(
            z,
            prevertices,
            z1=z1,
            w1=w1,
            z2=z2,
            w2=w2,
            z0=z0,
            w0=w0,
            scale=scale,
            tolerance=tolerance,
            max_iterations=max_iterations,
            use_ode_fallback=use_ode_fallback,
        )
        return result.w, result.used_ode

    def z2w_result(
        self,
        z: complex,
        prevertices: np.ndarray,
        *,
        z1: complex,
        w1: complex,
        z2: complex,
        w2: complex,
        z0: complex = 0.0 + 0.0j,
        w0: complex = 0.0 + 0.0j,
        scale: complex = 1.0 + 0.0j,
        tolerance: float = 1.0e-10,
        max_iterations: int = 100,
        use_ode_fallback: bool = True,
    ) -> _InverseResult:
        prevertices = np.asarray(prevertices, dtype=complex)
        target = complex(z)
        z0 = complex(z0)
        w0 = complex(w0)
        scale = complex(scale)
        za = complex(z1)
        zb = complex(z2)
        wa = complex(w1)
        wb = complex(w2)
        w = wb
        secant_failed = False
        acceptance_tolerance = max(tolerance * 1000.0, 1.0e-6)

        for count in range(min(max_iterations, 20)):
            if abs(wa - wb) <= tolerance:
                break
            if abs(zb - za) <= tolerance:
                break
            w = (wb * (target - za) - wa * (target - zb)) / (zb - za)
            wabs = abs(w)
            if wabs > 2.0:
                secant_failed = True
                break
            wa = wb
            za = zb
            wb = w / wabs if wabs > 1.1 else w
            zb = self.w2z(
                wb,
                prevertices,
                w0=w0,
                z0=z0,
                scale=scale,
            )
            if abs(zb - target) <= tolerance:
                return _InverseResult(w=wb, method="secant", residual=float(abs(zb - target)))
        else:
            secant_failed = True

        if not secant_failed:
            if abs(w) <= 1.0 + tolerance and abs(zb - za) > tolerance:
                candidate = (wb * (target - za) - wa * (target - zb)) / (zb - za)
                if abs(candidate - wb) < tolerance:
                    residual = self._inverse_residual(
                        target,
                        candidate,
                        prevertices,
                        z0=z0,
                        w0=w0,
                        scale=scale,
                    )
                    if residual <= acceptance_tolerance:
                        return _InverseResult(w=candidate, method="secant", residual=residual)
                residual = self._inverse_residual(
                    target,
                    wb,
                    prevertices,
                    z0=z0,
                    w0=w0,
                    scale=scale,
                )
                if residual <= acceptance_tolerance:
                    return _InverseResult(w=wb, method="secant", residual=residual)
            if abs(w) <= 1.0 + tolerance:
                residual = self._inverse_residual(
                    target,
                    wb,
                    prevertices,
                    z0=z0,
                    w0=w0,
                    scale=scale,
                )
                if residual <= acceptance_tolerance:
                    return _InverseResult(w=wb, method="secant", residual=residual)

        starts = [wb, wa, 0.0 + 0.0j]
        starts.extend(0.99 * prevertex for prevertex in prevertices if abs(prevertex) > 0.0)
        starts.extend(0.9 * prevertex for prevertex in prevertices if abs(prevertex) > 0.0)
        starts.extend(0.95 * np.exp(1j * angle) for angle in np.linspace(-np.pi, np.pi, 17))
        for start in starts:
            if abs(start) >= 1.0:
                start = start / abs(start) * 0.9
            try:
                newton_w = self._z2w_newton(
                    target,
                    prevertices,
                    start=start,
                    z0=z0,
                    w0=w0,
                    scale=scale,
                    tolerance=tolerance,
                    max_iterations=max_iterations,
                )
                residual = self._inverse_residual(
                    target,
                    newton_w,
                    prevertices,
                    z0=z0,
                    w0=w0,
                    scale=scale,
                )
                if residual > acceptance_tolerance:
                    continue
                return _InverseResult(
                    w=newton_w,
                    method="newton",
                    residual=residual,
                )
            except CRDTError:
                continue
        try:
            root_w = self._z2w_root_solve(
                target,
                prevertices,
                starts=tuple(starts),
                z0=z0,
                w0=w0,
                scale=scale,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            return _InverseResult(
                w=root_w,
                method="root",
                residual=self._inverse_residual(
                    target,
                    root_w,
                    prevertices,
                    z0=z0,
                    w0=w0,
                    scale=scale,
                ),
            )
        except CRDTError:
            pass
        try:
            least_squares_w = self._z2w_least_squares(
                target,
                prevertices,
                starts=tuple(starts),
                z0=z0,
                w0=w0,
                scale=scale,
                tolerance=tolerance,
                max_iterations=max_iterations,
            )
            return _InverseResult(
                w=least_squares_w,
                method="least_squares",
                residual=self._inverse_residual(
                    target,
                    least_squares_w,
                    prevertices,
                    z0=z0,
                    w0=w0,
                    scale=scale,
                ),
            )
        except CRDTError:
            pass
        if not use_ode_fallback:
            raise CRDTError("SC inverse Newton solve failed; relax precision or simplify geometry")

        ode_w = self._z2w_ode(
            target,
            prevertices,
            z0=z0,
            w0=w0,
            scale=scale,
            tolerance=tolerance,
        )
        return _InverseResult(
            w=ode_w,
            method="ode",
            residual=self._inverse_residual(
                target,
                ode_w,
                prevertices,
                z0=complex(z0),
                w0=complex(w0),
                scale=complex(scale),
            ),
        )

    def _prod(self, w: complex, vertex_index: int, prevertices: np.ndarray) -> complex:
        return complex(_sc_prod_numba(w, vertex_index, prevertices, self.betas))

    def _sum(self, z0: complex, z1: complex, vertex_index: int, prevertices: np.ndarray) -> complex:
        zh = (z1 - z0) / 2.0
        zc = (z1 + z0) / 2.0
        rule_index = len(self.betas) if vertex_index < 0 else vertex_index
        nodes = self._nodes[rule_index]
        weights = self._weights[rule_index]
        return complex(
            _sc_sum_numba(
                z0,
                z1,
                vertex_index,
                prevertices,
                self.betas,
                nodes,
                weights,
            )
        )

    def _integrate_one_sided(
        self,
        z0: complex,
        z1: complex,
        vertex_index: int,
        prevertices: np.ndarray,
    ) -> complex:
        if abs(z0 - z1) == 0.0:
            return 0.0 + 0.0j
        distance = self._nearest_prevertex_distance(z0, vertex_index, prevertices)
        ratio = min(1.0, distance / abs(z0 - z1))
        current = z0 + ratio * (z1 - z0)
        total = self._sum(z0, current, vertex_index, prevertices)
        guard = 0
        while ratio != 1.0 and guard < 100:
            distance = self._nearest_prevertex_distance(current, -1, prevertices)
            ratio = min(1.0, distance / abs(current - z1))
            next_point = current + ratio * (z1 - current)
            total += self._sum(current, next_point, -1, prevertices)
            current = next_point
            guard += 1
        return total

    def _integrate(
        self,
        w: complex,
        vertex_index: int,
        w0: complex,
        w0_vertex_index: int,
        prevertices: np.ndarray,
    ) -> complex:
        if abs(w - w0) == 0.0:
            return 0.0 + 0.0j
        if abs(w) > 1.5 or abs(w0) > 1.5:
            raise CRDTError(
                "SC integration point left the unit disk; relax precision or simplify geometry"
            )
        midpoint = (w + w0) / 2.0
        return self._integrate_one_sided(
            w, midpoint, vertex_index, prevertices
        ) - self._integrate_one_sided(
            w0,
            midpoint,
            w0_vertex_index,
            prevertices,
        )

    def _nearest_prevertex_distance(
        self,
        point: complex,
        excluded_index: int,
        prevertices: np.ndarray,
    ) -> float:
        distances = [
            abs(point - prevertex)
            for idx, prevertex in enumerate(prevertices)
            if idx != excluded_index
        ]
        return float(min(distances)) if distances else 1.0

    def _z2w_ode(
        self,
        z: complex,
        prevertices: np.ndarray,
        *,
        z0: complex,
        w0: complex,
        scale: complex,
        tolerance: float,
    ) -> complex:
        dz = z - z0
        if abs(dz) == 0.0:
            return w0

        def rhs(w: complex) -> complex:
            product = self._prod(w, -1, prevertices)
            if abs(product) <= 1.0e-300:
                product = 1.0e-300 + 0.0j
            return dz / scale / product

        w = complex(w0)
        nsteps = 160
        step = 1.0 / nsteps
        for _ in range(nsteps):
            try:
                k1 = rhs(w)
                k2 = rhs(w + 0.5 * step * k1)
                k3 = rhs(w + 0.5 * step * k2)
                k4 = rhs(w + step * k3)
            except FloatingPointError as error:
                raise CRDTError(
                    "SC inverse ODE solve failed; relax precision or simplify geometry"
                ) from error
            w = w + step * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            if not np.isfinite(w.real) or not np.isfinite(w.imag):
                raise CRDTError("SC inverse ODE solve failed; relax precision or simplify geometry")

        if abs(w) > 1.0:
            w /= abs(w)
        return w

    def _z2w_newton(
        self,
        z: complex,
        prevertices: np.ndarray,
        *,
        start: complex,
        z0: complex,
        w0: complex,
        scale: complex,
        tolerance: float,
        max_iterations: int,
    ) -> complex:
        w = start
        best_w = w
        best_error = np.inf
        for _ in range(max_iterations):
            current = self.w2z(w, prevertices, w0=w0, z0=z0, scale=scale)
            residual = current - z
            error = abs(residual)
            if error < best_error:
                best_error = error
                best_w = w
            if error <= tolerance:
                return w
            derivative = scale * self._prod(w, -1, prevertices)
            if abs(derivative) <= 1.0e-14:
                break
            step = residual / derivative
            damping = 1.0
            accepted = False
            while damping >= 1.0e-3:
                candidate = w - damping * step
                if abs(candidate) < 1.0 - max(1.0e-8, tolerance):
                    w = candidate
                    accepted = True
                    break
                damping *= 0.5
            if not accepted:
                break
        if best_error <= max(tolerance * 100.0, 1.0e-7):
            return best_w
        raise CRDTError("SC inverse Newton solve failed; relax precision or simplify geometry")

    def _z2w_root_solve(
        self,
        z: complex,
        prevertices: np.ndarray,
        *,
        starts: tuple[complex, ...],
        z0: complex,
        w0: complex,
        scale: complex,
        tolerance: float,
        max_iterations: int,
    ) -> complex:
        best_w = 0.0 + 0.0j
        best_error = np.inf

        def residual(values: np.ndarray) -> np.ndarray:
            w = complex(values[0], values[1])
            if abs(w) >= 1.0:
                w = w / abs(w) * (1.0 - 1.0e-9)
            mapped = self.w2z(w, prevertices, w0=w0, z0=z0, scale=scale)
            diff = mapped - z
            return np.asarray([diff.real, diff.imag], dtype=float)

        for start in starts:
            if abs(start) >= 1.0:
                start = start / abs(start) * 0.9
            solution = root(
                residual,
                np.asarray([start.real, start.imag], dtype=float),
                method="hybr",
                options={"maxfev": max(20, max_iterations)},
            )
            candidate = complex(solution.x[0], solution.x[1])
            if abs(candidate) >= 1.0:
                candidate = candidate / abs(candidate) * (1.0 - 1.0e-9)
            error = abs(self.w2z(candidate, prevertices, w0=w0, z0=z0, scale=scale) - z)
            if error < best_error:
                best_error = error
                best_w = candidate
            if solution.success and error <= max(tolerance * 100.0, 1.0e-8):
                return candidate

        if best_error <= max(tolerance * 1000.0, 1.0e-6):
            return best_w
        raise CRDTError("SC inverse root solve failed; relax precision or simplify geometry")

    def _z2w_least_squares(
        self,
        z: complex,
        prevertices: np.ndarray,
        *,
        starts: tuple[complex, ...],
        z0: complex,
        w0: complex,
        scale: complex,
        tolerance: float,
        max_iterations: int,
    ) -> complex:
        best_w = 0.0 + 0.0j
        best_error = np.inf
        radius = 1.0 - 1.0e-9

        def project(w: complex) -> complex:
            wabs = abs(w)
            return w / wabs * radius if wabs >= radius else w

        def residual(values: np.ndarray) -> np.ndarray:
            w = project(complex(values[0], values[1]))
            mapped = self.w2z(w, prevertices, w0=w0, z0=z0, scale=scale)
            diff = mapped - z
            return np.asarray([diff.real, diff.imag], dtype=float)

        for start in starts:
            start = project(start)
            solution = least_squares(
                residual,
                np.asarray([start.real, start.imag], dtype=float),
                bounds=([-radius, -radius], [radius, radius]),
                xtol=tolerance,
                ftol=tolerance,
                gtol=tolerance,
                max_nfev=max(100, max_iterations * 4),
            )
            candidate = project(complex(solution.x[0], solution.x[1]))
            error = self._inverse_residual(
                z,
                candidate,
                prevertices,
                z0=z0,
                w0=w0,
                scale=scale,
            )
            if error < best_error:
                best_error = error
                best_w = candidate
            if solution.success and error <= max(tolerance * 1000.0, 1.0e-6):
                return candidate

        if best_error <= max(tolerance * 1000.0, 1.0e-6):
            return best_w
        raise CRDTError(
            "SC inverse least-squares solve failed; relax precision or simplify geometry"
        )

    def _inverse_residual(
        self,
        z: complex,
        w: complex,
        prevertices: np.ndarray,
        *,
        z0: complex,
        w0: complex,
        scale: complex,
    ) -> float:
        mapped = self.w2z(w, prevertices, w0=w0, z0=z0, scale=scale)
        return float(abs(mapped - z))


def generate_crdt_grid(
    boundary: BoundaryDefinition,
    shape: tuple[int, int],
    focus: Focus | None = None,
    options: CRDTOptions | None = None,
):
    from cgrid_tools.grid import Grid

    from .generator import SolverMode, _generation_options

    if options is None:
        options = CRDTOptions()
    ny, nx = grid_shape(shape)

    state = solve_crdt_state(boundary, options)
    image_polygon = np.column_stack((state.image_polygon.real, state.image_polygon.imag))
    xi = np.linspace(0.0, 1.0, nx)
    eta = np.linspace(0.0, 1.0, ny)
    XI, ETA = np.meshgrid(xi, eta)
    if focus is not None:
        XI, ETA = focus(XI, ETA)

    xmin, xmax, ymin, ymax = state.image_extents
    image_x = xmin + XI * (xmax - xmin)
    image_y = ymin + ETA * (ymax - ymin)
    valid_nodes = _points_in_polygon(image_x, image_y, image_polygon)
    stats = _MappingStats()
    x, y = _map_image_grid_to_physical(image_x, image_y, state, valid_nodes, stats)
    _repair_duplicate_run_nodes(x, y)
    mask = _finite_cell_mask(x, y)
    validate_cells(x, y, mask)
    generation_options = _generation_options(
        boundary=boundary,
        mode=SolverMode.GRIDGEN_CRDT,
        shape=shape,
        focus=focus,
        smoothing_iterations=0,
        smoothing_strength=0.0,
        elliptic_relaxation=0.0,
        elliptic_tolerance=0.0,
        crdt_options=options,
        iterations=0,
    )
    mapping = asdict(stats)
    point = mapping["worst_image_point"]
    mapping["worst_image_point"] = None if point is None else [point.real, point.imag]
    generation_options["diagnostics"] = {
        "sigma_residual_norm": state.residual_norm,
        "converged": state.residual_norm <= options.precision,
        "image_vertex_mismatch": state.diagnostics.image_vertex_mismatch,
        "mapping": mapping,
    }
    lon_vert, lat_vert = boundary.geographic_vertices(x, y)
    return Grid(
        x_vert=x,
        y_vert=y,
        mask_rho=mask,
        lon_vert=lon_vert,
        lat_vert=lat_vert,
        crs=boundary.cartesian_crs,
        options=generation_options,
    )


def solve_crdt_state(
    boundary: BoundaryDefinition,
    options: CRDTOptions | None = None,
) -> CRDTSolverState:
    if options is None:
        options = CRDTOptions()
    topology = prepare_crdt_topology(boundary)
    points = topology.triangulation.points
    user_betas = topology.triangulation.beta
    betas = physical_betas(points)
    image_marker_betas, image_beta_source = _prepare_image_marker_betas(
        boundary,
        options,
        len(points),
        user_betas,
        betas,
    )
    cis = _calculate_cis(points, topology.quadrilaterals)
    physical_sc = SchwarzChristoffelTransform(betas, nnodes=options.nnodes)
    sigmas, residual_norm, ws, physical_a, physical_b = _solve_sigmas(
        topology,
        points,
        physical_sc,
        cis,
        options,
    )
    image_betas = _image_betas(image_marker_betas)
    image_sc = SchwarzChristoffelTransform(image_betas, nnodes=options.nnodes)
    (
        image_vertices,
        image_a,
        image_b,
        raw_image_vertices,
        alignment_displacement,
    ) = _calculate_image_vertices(
        topology,
        image_betas,
        image_sc,
        ws,
        options,
    )
    if image_vertices is None or image_a is None or image_b is None or raw_image_vertices is None:
        raise CRDTError(
            "CRDT image-region construction failed; try simplifying boundary "
            "geometry or relaxing precision"
        )
    pre_alignment_vertex_mismatch = _image_vertex_mismatch(
        topology,
        raw_image_vertices,
        image_sc,
        ws,
        image_a,
        image_b,
    )
    image_vertices, image_a, image_b = _normalise_image_orientation(
        image_vertices,
        image_a,
        image_b,
        points,
    )
    image_quadrilateral_polygons = _map_quadrilateral_polygons(
        topology,
        points[:, 0] + 1j * points[:, 1],
        image_vertices,
        physical_sc,
        image_sc,
        ws,
        physical_a,
        physical_b,
        image_a,
        image_b,
        options,
    )
    corner_vertex_ids = np.flatnonzero(image_marker_betas != 0.0)
    image_polygon = image_vertices[corner_vertex_ids]
    image_extents = _image_extents(np.column_stack((image_polygon.real, image_polygon.imag)))
    corner_diagnostics = _corner_angle_diagnostics(
        betas,
        image_marker_betas,
        corner_vertex_ids,
    )
    diagnostics = CRDTDiagnostics(
        sigma_residual_norm=residual_norm,
        pre_alignment_vertex_mismatch=pre_alignment_vertex_mismatch,
        alignment_displacement=alignment_displacement,
        post_alignment_vertex_mismatch=_image_vertex_mismatch(
            topology,
            image_vertices,
            image_sc,
            ws,
            image_a,
            image_b,
        ),
        physical_corner_angles=corner_diagnostics[0],
        image_corner_angles=corner_diagnostics[1],
        corner_angle_ratios=corner_diagnostics[2],
        corner_angle_warning=corner_diagnostics[3],
        image_beta_source=image_beta_source,
    )
    return CRDTSolverState(
        topology=topology,
        zs=points[:, 0] + 1j * points[:, 1],
        betas=betas,
        user_betas=user_betas,
        image_marker_betas=image_marker_betas,
        corner_vertex_ids=corner_vertex_ids,
        cis=cis,
        sigmas=sigmas,
        residual_norm=residual_norm,
        physical_sc=physical_sc,
        image_sc=image_sc,
        ws=ws,
        physical_a=physical_a,
        physical_b=physical_b,
        image_a=image_a,
        image_b=image_b,
        image_quadrilateral_polygons=image_quadrilateral_polygons,
        image_vertices=image_vertices,
        image_polygon=image_polygon,
        image_extents=image_extents,
        diagnostics=diagnostics,
    )


def _solve_sigmas(
    topology,
    points: np.ndarray,
    transform: SchwarzChristoffelTransform,
    cis: np.ndarray,
    options: CRDTOptions,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    if len(topology.quadrilaterals) == 0:
        raise CRDTError("CRDT topology has no quadrilateral embeddings")
    zs = points[:, 0] + 1j * points[:, 1]

    def residual(sigmas: np.ndarray) -> np.ndarray:
        values = np.empty(len(topology.quadrilaterals), dtype=float)
        for idx, quad in enumerate(topology.quadrilaterals):
            ws = _calculate_fi(topology, sigmas, idx)
            dzetas = _integrate_quad_vertices(transform, quad.vids, ws)
            z0, z1, z2, z3 = dzetas
            cross_ratio = (z3 - z0) * (z1 - z2) / (z2 - z3) / (z0 - z1)
            values[idx] = np.log(abs(cross_ratio)) - cis[idx]
        if not np.all(np.isfinite(values)):
            raise CRDTError("CRDT sigma residual produced non-finite values")
        return values

    initial = cis.copy()
    method = "hybr" if options.newton else "broyden1"
    solver_options = (
        # The residual includes numerical quadrature and crowded prevertices.
        # Treating it as accurate to machine epsilon makes finite-difference
        # Jacobians unstable across NumPy/Numba versions. eps is the residual
        # noise estimate: MINPACK takes differences on its square-root scale.
        {"maxfev": options.max_iterations * max(1, len(initial)), "eps": 1.0e-12}
        if options.newton
        else {"maxiter": options.max_iterations}
    )
    solution = root(
        residual,
        initial,
        method=method,
        tol=max(options.precision, np.finfo(float).eps),
        options=solver_options,
    )
    # A stagnation flag can accompany a good root; a success flag can accompany
    # a poor one. Choose by the measured residual, never by the flag alone.
    sigmas = initial
    residual_norm = float(np.linalg.norm(residual(initial)))
    candidate = np.asarray(solution.x, dtype=float)
    if np.all(np.isfinite(candidate)):
        try:
            candidate_norm = float(np.linalg.norm(residual(candidate)))
        except CRDTError:
            candidate_norm = np.inf
        if candidate_norm < residual_norm:
            sigmas, residual_norm = candidate, candidate_norm
    if not np.isfinite(residual_norm) or residual_norm > max(options.precision * 100.0, 1.0e-8):
        raise CRDTError(
            f"CRDT sigma solve failed (residual={residual_norm:.3e}, target={options.precision:.3e}); try increasing nnodes, relaxing precision, "
            "or simplifying/moving boundary vertices"
        )
    if residual_norm > options.precision:
        warnings.warn(
            "CRDT sigma residual exceeds target precision "
            f"(method={method!r}, message={solution.message!r}); "
            f"residual_norm={residual_norm:.3e} "
            f"(target precision={options.precision:.3e}). "
            "Results may be degraded; consider increasing nnodes or max_iterations.",
            RuntimeWarning,
            stacklevel=2,
        )
    ws, physical_a, physical_b = _calculate_all_dzetas(topology, zs, transform, sigmas)
    return sigmas, residual_norm, ws, physical_a, physical_b


def _calculate_all_dzetas(
    topology,
    zs: np.ndarray,
    transform: SchwarzChristoffelTransform,
    sigmas: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nquads = len(topology.quadrilaterals)
    nvertices = len(zs)
    ws_all = np.zeros((nquads, nvertices), dtype=complex)
    affine_a = np.zeros(nquads, dtype=complex)
    affine_b = np.ones(nquads, dtype=complex)
    for qid, quad in enumerate(topology.quadrilaterals):
        ws = _calculate_fi(topology, sigmas, qid)
        ws_all[qid] = ws
        dzetas = _integrate_quad_vertices(transform, quad.vids, ws)
        z0 = zs[quad.vids[0]]
        z2 = zs[quad.vids[2]]
        if abs(dzetas[2] - dzetas[0]) > 0.0:
            affine_b[qid] = (z2 - z0) / (dzetas[2] - dzetas[0])
            affine_a[qid] = z0 - affine_b[qid] * dzetas[0]
    return ws_all, affine_a, affine_b


def _calculate_affine_constants_for_vertices(
    topology,
    vertices: np.ndarray,
    transform: SchwarzChristoffelTransform,
    ws_all: np.ndarray,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    nquads = len(topology.quadrilaterals)
    affine_a = np.zeros(nquads, dtype=complex)
    affine_b = np.ones(nquads, dtype=complex)
    for qid, quad in enumerate(topology.quadrilaterals):
        dzetas = _integrate_quad_vertices(transform, quad.vids, ws_all[qid])
        z0 = vertices[quad.vids[0]]
        z2 = vertices[quad.vids[2]]
        denom = dzetas[2] - dzetas[0]
        if abs(denom) <= 0.0:
            return None, None
        affine_b[qid] = (z2 - z0) / denom
        affine_a[qid] = z0 - affine_b[qid] * dzetas[0]
    return affine_a, affine_b


def _image_vertex_mismatch(
    topology,
    image_vertices: np.ndarray,
    transform: SchwarzChristoffelTransform,
    ws_all: np.ndarray,
    affine_a: np.ndarray,
    affine_b: np.ndarray,
) -> float:
    mismatch = 0.0
    for quad in topology.quadrilaterals:
        dzetas = _integrate_quad_vertices(transform, quad.vids, ws_all[quad.id])
        mapped = affine_a[quad.id] + affine_b[quad.id] * dzetas
        target = image_vertices[list(quad.vids)]
        mismatch = max(mismatch, float(np.max(np.abs(mapped - target))))
    return mismatch


def _normalise_image_orientation(
    image_vertices: np.ndarray,
    image_a: np.ndarray,
    image_b: np.ndarray,
    physical_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge = image_vertices[1] - image_vertices[0]
    target = _dominant_direction(physical_points[1] - physical_points[0])
    rotation = 1.0 + 0.0j if abs(edge) == 0.0 else target * abs(edge) / edge
    vertices = image_vertices * rotation
    offset = -float(np.min(vertices.real)) - 1j * float(np.min(vertices.imag))
    return vertices + offset, image_a * rotation + offset, image_b * rotation


def _dominant_direction(vector: np.ndarray) -> complex:
    if abs(vector[0]) >= abs(vector[1]):
        return 1.0 + 0.0j if vector[0] >= 0.0 else -1.0 + 0.0j
    return 0.0 + 1.0j if vector[1] >= 0.0 else 0.0 - 1.0j


def _map_quadrilateral_polygons(
    topology,
    zs: np.ndarray,
    image_vertices: np.ndarray,
    physical_sc: SchwarzChristoffelTransform,
    image_sc: SchwarzChristoffelTransform,
    ws_all: np.ndarray,
    physical_a: np.ndarray | None,
    physical_b: np.ndarray | None,
    image_a: np.ndarray | None,
    image_b: np.ndarray | None,
    options: CRDTOptions,
) -> tuple[np.ndarray, ...] | None:
    if (
        options.nppe <= 0
        or physical_a is None
        or physical_b is None
        or image_a is None
        or image_b is None
    ):
        return None

    diagonal_samples: dict[tuple[int, int], tuple[int, np.ndarray]] = {}
    for qid, quad in enumerate(topology.quadrilaterals):
        edge = _edge_key((quad.vids[0], quad.vids[2]))
        samples = _map_diagonal_samples(
            zs,
            physical_sc,
            image_sc,
            ws_all[qid],
            physical_a[qid],
            physical_b[qid],
            image_a[qid],
            image_b[qid],
            quad.vids[0],
            quad.vids[2],
            options,
        )
        if samples is not None:
            diagonal_samples[edge] = (qid, samples)
        else:
            raise CRDTError(
                "CRDT quadrilateral diagonal mapping failed; try increasing nnodes, "
                "relaxing precision, or simplifying boundary geometry"
            )

    polygons: list[np.ndarray] = []
    nvertices = len(zs)
    for quad in topology.quadrilaterals:
        vertices: list[complex] = []
        for edge_idx in range(4):
            vid0 = quad.vids[edge_idx]
            vid1 = quad.vids[(edge_idx + 1) % 4]
            vertices.append(image_vertices[vid0])
            diff = abs(vid0 - vid1)
            if diff == 1 or diff == nvertices - 1:
                continue
            diagonal_entry = diagonal_samples.get(_edge_key((vid0, vid1)))
            if diagonal_entry is None:
                continue
            sample_qid, samples = diagonal_entry
            if vid0 != topology.quadrilaterals[sample_qid].vids[0]:
                samples = samples[::-1]
            vertices.extend(samples)
        polygons.append(np.asarray(vertices, dtype=complex))
    return tuple(polygons)


def _map_diagonal_samples(
    zs: np.ndarray,
    physical_sc: SchwarzChristoffelTransform,
    image_sc: SchwarzChristoffelTransform,
    ws: np.ndarray,
    physical_a: complex,
    physical_b: complex,
    image_a: complex,
    image_b: complex,
    vid0: int,
    vid1: int,
    options: CRDTOptions,
) -> np.ndarray | None:
    samples: list[complex] = []
    for idx in range(options.nppe):
        fraction = (idx + 1.0) / (options.nppe + 1.0)
        z = zs[vid0] * (1.0 - fraction) + zs[vid1] * fraction
        start_vid = vid0 if fraction <= 0.5 else vid1
        try:
            w, _ = physical_sc.z2w_with_status(
                z,
                ws,
                z1=zs[start_vid],
                w1=ws[start_vid],
                z2=physical_a,
                w2=0.0 + 0.0j,
                z0=physical_a,
                w0=0.0 + 0.0j,
                scale=physical_b,
                tolerance=max(options.precision * 10.0, 1.0e-9),
                max_iterations=30,
                use_ode_fallback=True,
            )
            samples.append(
                image_sc.w2z(
                    w,
                    ws,
                    vertex_index=-1,
                    w0=0.0 + 0.0j,
                    z0=image_a,
                    scale=image_b,
                )
            )
        except CRDTError:
            return None
    return np.asarray(samples, dtype=complex)


def _calculate_fi(topology, sigmas: np.ndarray, start_qid: int) -> np.ndarray:
    nvertices = len(topology.triangulation.points)
    ws = np.zeros(nvertices, dtype=complex)
    done = np.zeros(len(topology.quadrilaterals), dtype=bool)
    quad = topology.quadrilaterals[start_qid]
    ro = float(np.exp(sigmas[start_qid]))
    alpha = float(np.arctan(np.sqrt(ro)))
    ws[quad.vids[0]] = np.cos(alpha) + 1j * np.sin(alpha)
    ws[quad.vids[1]] = -np.cos(alpha) + 1j * np.sin(alpha)
    ws[quad.vids[2]] = -np.cos(alpha) - 1j * np.sin(alpha)
    _process_quadrilateral(topology, sigmas, ws, done, start_qid, True)
    return ws


def _process_quadrilateral(
    topology,
    sigmas: np.ndarray,
    ws: np.ndarray,
    done: np.ndarray,
    start_qid: int,
    start_do_second_triangle: bool,
) -> None:
    """Iteratively propagate image-vertex positions across the quadrilateral graph.

    Uses an explicit LIFO stack to avoid Python recursion limits on large
    topologies. Behavior is equivalent to the previous recursive implementation:
    a DFS rooted at ``start_qid`` that sets the remaining vertex of each
    quadrilateral from the three already-known vertices of the adjacent one.
    """
    stack: list[tuple[int, bool]] = [(start_qid, start_do_second_triangle)]
    while stack:
        qid, do_second_triangle = stack.pop()
        if done[qid]:
            continue
        quad = topology.quadrilaterals[qid]
        ro = -float(np.exp(sigmas[qid]))
        v0, v1, v2, v3 = quad.vids
        if do_second_triangle:
            pa, pb, pc = ws[v0], ws[v1], ws[v2]
            # Unit-circle prevertices can be very close but still distinct.
            # np.isclose's relative tolerance would collapse crowded corners.
            if abs(pc - pb) <= 8.0 * np.finfo(float).eps:
                ws[v3] = pc
            else:
                ws[v3] = ((ro * (pb - pa) / (pc - pb)) * pc + pa) / (
                    ro * (pb - pa) / (pc - pb) + 1.0
                )
        else:
            pa, pb, pc = ws[v2], ws[v3], ws[v0]
            if abs(pc - pb) <= 8.0 * np.finfo(float).eps:
                ws[v1] = pc
            else:
                ws[v1] = ((ro * (pb - pa) / (pc - pb)) * pc + pa) / (
                    ro * (pb - pa) / (pc - pb) + 1.0
                )

        done[qid] = True
        for neighbour_id in quad.neighbours:
            if done[neighbour_id]:
                continue
            neighbour = topology.quadrilaterals[neighbour_id]
            next_second = neighbour.tids[0] in quad.tids
            stack.append((neighbour_id, next_second))


def _integrate_quad_vertices(
    transform: SchwarzChristoffelTransform,
    vids: tuple[int, int, int, int],
    ws: np.ndarray,
) -> np.ndarray:
    return np.asarray(
        [
            transform.w2z(
                ws[vid],
                ws,
                vertex_index=vid,
                w0=0.0 + 0.0j,
                z0=0.0 + 0.0j,
                scale=1.0 + 0.0j,
            )
            for vid in vids
        ],
        dtype=complex,
    )


def _prepare_image_marker_betas(
    boundary: BoundaryDefinition,
    options: CRDTOptions,
    expected_vertices: int,
    topology_betas: np.ndarray,
    physical_betas_: np.ndarray,
) -> tuple[np.ndarray, str]:
    if options.image_beta is not None:
        raw_beta = _validate_image_beta(boundary, options.image_beta)
        _, image_beta = _preprocess_marker_betas_like_boundary(boundary.vertices, raw_beta)
        source = "explicit"
    elif options.image_beta_blend is not None:
        blend = float(options.image_beta_blend)
        if blend < 0.0 or blend > 1.0:
            raise ValueError("image_beta_blend must lie within [0, 1]")
        image_beta = (1.0 - blend) * topology_betas + blend * (-2.0 * physical_betas_)
        source = "blend"
    else:
        image_beta = np.asarray(topology_betas, dtype=float)
        source = "topology"
    if len(image_beta) != expected_vertices:
        raise ValueError(
            "image_beta preprocessing did not match CRDT boundary preprocessing; "
            "check duplicate boundary vertices"
        )
    if np.count_nonzero(image_beta != 0.0) < 3:
        raise ValueError("image_beta must mark at least three image-region corners")
    if not np.isclose(float(np.sum(image_beta)), 4.0):
        raise ValueError("sum(image_beta) must equal 4")
    return image_beta, source


def _validate_image_beta(boundary: BoundaryDefinition, image_beta: np.ndarray) -> np.ndarray:
    beta = np.asarray(image_beta, dtype=float)
    if beta.ndim != 1:
        raise ValueError("image_beta must be a one-dimensional array")
    vertex_count = len(boundary.x) - 1
    expected_lengths = {vertex_count, vertex_count + 1}
    if len(beta) not in expected_lengths:
        raise ValueError(
            "image_beta must have one entry per vertex, with an optional trailing closing entry"
        )
    if len(beta) == vertex_count + 1:
        if not np.isclose(beta[-1], 0.0):
            raise ValueError("the closing image_beta entry must be 0")
        beta = beta[:-1]
    if not np.isclose(float(np.sum(beta)), 4.0):
        raise ValueError("sum(image_beta) must equal 4")
    return beta


def _preprocess_marker_betas_like_boundary(
    points: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=float)
    beta = np.asarray(beta, dtype=float).copy()
    if len(points) != len(beta):
        raise ValueError("image_beta and boundary vertices must have matching lengths")

    span = np.ptp(points, axis=0)
    tolerance = max(float(np.max(span)) / 1.0e6, 1.0e-12)
    keep = [0]
    for idx in range(1, len(points)):
        if np.linalg.norm(points[idx] - points[keep[-1]]) > tolerance:
            keep.append(idx)
        else:
            beta[keep[-1]] += beta[idx]
    if len(keep) > 1 and np.linalg.norm(points[keep[0]] - points[keep[-1]]) <= tolerance:
        beta[keep[0]] += beta[keep[-1]]
        keep.pop()

    points = points[keep]
    beta = beta[keep]
    if _signed_area(points) < 0.0:
        points = points[::-1]
        beta = beta[::-1]
    return points, beta


def _image_betas(user_betas: np.ndarray) -> np.ndarray:
    image_betas = np.zeros_like(user_betas, dtype=float)
    marked = user_betas != 0.0
    image_betas[marked] = -0.5 * user_betas[marked]
    return image_betas


def _calculate_image_vertices(
    topology,
    image_betas: np.ndarray,
    image_sc: SchwarzChristoffelTransform,
    ws_all: np.ndarray,
    options: CRDTOptions,
) -> tuple[
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    np.ndarray | None,
    float,
]:
    nquads = len(topology.quadrilaterals)
    nvertices = len(topology.triangulation.points)
    corner_ids = np.flatnonzero(image_betas != 0.0)
    if len(corner_ids) < 3:
        raise CRDTError("CRDT image region needs at least three beta-marked corners")
    image_zs = np.zeros(nvertices, dtype=complex)
    image_a = np.zeros(nquads, dtype=complex)
    image_b = np.ones(nquads, dtype=complex)
    start_qid = _first_quadrilateral(topology, int(corner_ids[0]))
    if start_qid is None:
        return None, None, None, None, np.inf
    start_quad = topology.quadrilaterals[start_qid]
    start_vid = int(corner_ids[0])
    next_vid = (start_vid + 1) % nvertices
    dzetas = _integrate_quad_vertices(image_sc, start_quad.vids, ws_all[start_qid])
    try:
        qvid0 = start_quad.vids.index(start_vid)
        qvid1 = start_quad.vids.index(next_vid)
    except ValueError:
        return None, None, None, None, np.inf
    denom = dzetas[qvid1] - dzetas[qvid0]
    if abs(denom) == 0.0:
        return None, None, None, None, np.inf
    image_b[start_qid] = -1j / denom
    image_a[start_qid] = 1j - image_b[start_qid] * dzetas[qvid0]
    for local_idx, vid in enumerate(start_quad.vids):
        image_zs[vid] = image_a[start_qid] + image_b[start_qid] * dzetas[local_idx]

    done = np.zeros(nquads, dtype=bool)
    done[start_qid] = True
    _propagate_image_vertices(
        topology,
        image_sc,
        ws_all,
        image_zs,
        image_a,
        image_b,
        done,
        start_qid,
    )
    if not np.all(done):
        return None, None, None, None, np.inf
    image_points = np.column_stack((image_zs.real, image_zs.imag))
    if not np.all(np.isfinite(image_points)) or abs(_signed_area(image_points)) <= 1.0e-12:
        return None, None, None, None, np.inf

    raw_image_zs = image_zs.copy()
    aligned = _align_image_vertices(image_zs, image_betas, corner_ids, options.precision)
    if aligned is not None:
        image_zs = aligned
    alignment_displacement = float(np.max(np.abs(image_zs - raw_image_zs)))
    return image_zs, image_a, image_b, raw_image_zs, alignment_displacement


def _first_quadrilateral(topology, start_vid: int) -> int | None:
    next_vid = (start_vid + 1) % len(topology.triangulation.points)
    candidates = [quad.id for quad in topology.quadrilaterals if start_vid in quad.vids]
    for qid in candidates:
        if next_vid in topology.quadrilaterals[qid].vids:
            return qid
    return candidates[0] if candidates else None


def _propagate_image_vertices(
    topology,
    image_sc: SchwarzChristoffelTransform,
    ws_all: np.ndarray,
    image_zs: np.ndarray,
    image_a: np.ndarray,
    image_b: np.ndarray,
    done: np.ndarray,
    start_qid: int,
) -> None:
    """Iteratively fill in image-plane vertex positions across the topology.

    Uses an explicit LIFO stack to avoid Python recursion limits for large
    topologies. Behavior is equivalent to the former recursive DFS.
    """
    stack: list[int] = [start_qid]
    while stack:
        qid = stack.pop()
        quad = topology.quadrilaterals[qid]
        for neighbour_id in quad.neighbours:
            if done[neighbour_id]:
                continue
            neighbour = topology.quadrilaterals[neighbour_id]
            new_vid = next((vid for vid in neighbour.vids if vid not in quad.vids), None)
            if new_vid is None:
                done[neighbour_id] = True
                continue
            dzetas = _integrate_quad_vertices(image_sc, neighbour.vids, ws_all[neighbour_id])
            z0 = image_zs[neighbour.vids[0]]
            z2 = image_zs[neighbour.vids[2]]
            denom = dzetas[2] - dzetas[0]
            if abs(denom) > 0.0:
                image_b[neighbour_id] = (z2 - z0) / denom
                image_a[neighbour_id] = z0 - image_b[neighbour_id] * dzetas[0]
                local_idx = neighbour.vids.index(new_vid)
                image_zs[new_vid] = (
                    image_a[neighbour_id] + image_b[neighbour_id] * dzetas[local_idx]
                )
            done[neighbour_id] = True
            stack.append(neighbour_id)


def _align_image_vertices(
    image_zs: np.ndarray,
    image_betas: np.ndarray,
    corner_ids: np.ndarray,
    precision: float,
) -> np.ndarray | None:
    aligned = image_zs.copy()
    corner_zs = aligned[corner_ids].copy()
    eps5 = precision * 5.0
    for idx, value in enumerate(corner_zs):
        x = value.real
        y = value.imag
        if abs(x) < eps5:
            x = 0.0
        elif abs(1.0 - x) < eps5:
            x = 1.0
        if abs(y) < eps5:
            y = 0.0
        elif abs(1.0 - y) < eps5:
            y = 1.0
        corner_zs[idx] = x + 1j * y

    for idx, vertex_id in enumerate(corner_ids):
        aligned[vertex_id] = corner_zs[idx]

    nvertices = len(aligned)
    beta = 0.0
    for idx, start_vid in enumerate(corner_ids):
        stop_vid = int(corner_ids[(idx + 1) % len(corner_ids)])
        start_vid = int(start_vid)
        start_z = aligned[start_vid]
        if abs(np.fmod(beta, 0.5)) < precision:
            k = int(round(abs(beta / 0.5)))
            steps = (stop_vid + nvertices - start_vid) % nvertices
            next_vid = (start_vid + 1) % nvertices
            for _ in range(steps - 1):
                if k % 2 == 0:
                    aligned[next_vid] = start_z.real + 1j * aligned[next_vid].imag
                else:
                    aligned[next_vid] = aligned[next_vid].real + 1j * start_z.imag
                next_vid = (next_vid + 1) % nvertices
        beta += image_betas[stop_vid]
    return aligned


def physical_betas(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    betas = np.empty(len(points), dtype=float)
    for idx in range(len(points)):
        previous = points[idx - 1]
        current = points[idx]
        next_point = points[(idx + 1) % len(points)]
        incoming = current - previous
        outgoing = next_point - current
        angle = np.angle(-complex(incoming[0], incoming[1]) / complex(outgoing[0], outgoing[1]))
        if angle < 0.0:
            angle += 2.0 * np.pi
        betas[idx] = angle / np.pi - 1.0
    return betas


def _corner_angle_diagnostics(
    physical_betas_: np.ndarray,
    image_marker_betas: np.ndarray,
    corner_vertex_ids: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    physical_angles = (physical_betas_[corner_vertex_ids] + 1.0) * np.pi
    image_sc_betas = _image_betas(image_marker_betas)
    image_angles = (image_sc_betas[corner_vertex_ids] + 1.0) * np.pi
    ratios = np.divide(
        physical_angles,
        image_angles,
        out=np.full_like(physical_angles, np.inf, dtype=float),
        where=np.abs(image_angles) > 0.0,
    )
    warnings = (ratios < 0.75) | (ratios > 1.25)
    return physical_angles, image_angles, ratios, warnings


def _calculate_cis(points: np.ndarray, quadrilaterals) -> np.ndarray:
    z = points[:, 0] + 1j * points[:, 1]
    cis = np.empty(len(quadrilaterals), dtype=float)
    for idx, quad in enumerate(quadrilaterals):
        z1, z2, z3, z4 = z[list(quad.vids)]
        value = (z4 - z1) * (z2 - z3) / (z3 - z4) / (z1 - z2)
        cis[idx] = np.log(abs(value))
    return cis


def _image_extents(image_vertices: np.ndarray) -> tuple[float, float, float, float]:
    xmin = float(np.min(image_vertices[:, 0]))
    xmax = float(np.max(image_vertices[:, 0]))
    ymin = float(np.min(image_vertices[:, 1]))
    ymax = float(np.max(image_vertices[:, 1]))
    return xmin, xmax, ymin, ymax


def _map_image_grid_to_physical(
    image_x: np.ndarray,
    image_y: np.ndarray,
    state: CRDTSolverState,
    valid_nodes: np.ndarray,
    stats: _MappingStats | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.full(image_x.shape, np.nan, dtype=float)
    y = np.full(image_y.shape, np.nan, dtype=float)
    if stats is None:
        stats = _MappingStats()
    continuation: dict[int, tuple[complex, complex]] = {}
    last_qid: int | None = None

    for index in np.argwhere(valid_nodes):
        j, i = int(index[0]), int(index[1])
        image_point = complex(image_x[j, i], image_y[j, i])
        mapped = _map_image_point_with_details(
            image_point,
            state,
            stats,
            continuation,
            node_index=(j, i),
            preferred_qid=last_qid,
        )
        if mapped is None:
            stats.sc_map_failures += 1
            continue
        x[j, i] = mapped.z.real
        y[j, i] = mapped.z.imag
        continuation[mapped.qid] = (image_point, mapped.w)
        last_qid = mapped.qid
    if stats.sc_map_failures:
        raise CRDTError(
            "CRDT point mapping failed for "
            f"{stats.sc_map_failures} image-grid nodes; try increasing nnodes, "
            "relaxing precision, or simplifying boundary geometry"
        )
    return x, y


def _map_image_point_with_sc(
    point: complex,
    state: CRDTSolverState,
    stats: _MappingStats | None = None,
) -> complex | None:
    mapped = _map_image_point_with_details(point, state, stats)
    return mapped.z if mapped is not None else None


def _map_image_point_with_details(
    point: complex,
    state: CRDTSolverState,
    stats: _MappingStats | None = None,
    continuation: dict[int, tuple[complex, complex]] | None = None,
    node_index: tuple[int, int] | None = None,
    preferred_qid: int | None = None,
) -> _MappedPoint | None:
    if (
        state.image_sc is None
        or state.physical_sc is None
        or state.ws is None
        or state.image_a is None
        or state.image_b is None
        or state.physical_a is None
        or state.physical_b is None
    ):
        return None

    if stats is not None:
        stats.candidate_nodes += 1
    candidate_qids = _candidate_image_quadrilaterals(point, state, preferred_qid)
    if not candidate_qids:
        return None
    residual_tolerance = _inverse_residual_tolerance(point, state.image_vertices)
    candidates = _map_candidate_quadrilaterals(
        point,
        state,
        candidate_qids,
        continuation,
        stats,
        residual_tolerance,
    )
    used_location_fallback = False
    if not candidates:
        fallback_qids = [
            quad.id for quad in state.topology.quadrilaterals if quad.id not in set(candidate_qids)
        ]
        candidates = _map_candidate_quadrilaterals(
            point,
            state,
            fallback_qids,
            continuation,
            stats,
            residual_tolerance,
        )
        used_location_fallback = bool(candidates)
    if not candidates:
        return None

    mapped = min(candidates, key=lambda candidate: candidate.inverse_residual)
    if stats is not None:
        _record_mapping_stats(stats, mapped, point, node_index)
        if used_location_fallback:
            stats.location_fallbacks += 1
    return mapped


def _map_candidate_quadrilaterals(
    point: complex,
    state: CRDTSolverState,
    qids: list[int],
    continuation: dict[int, tuple[complex, complex]] | None,
    stats: _MappingStats | None,
    residual_tolerance: float,
) -> list[_MappedPoint]:
    candidates: list[_MappedPoint] = []
    for qid in qids:
        if stats is not None:
            stats.candidate_attempts += 1
        try:
            mapped = _map_image_point_in_quadrilateral(
                point,
                state,
                qid,
                None if continuation is None else continuation.get(qid),
            )
        except CRDTError:
            continue
        if not np.isfinite(mapped.z.real) or not np.isfinite(mapped.z.imag):
            continue
        candidates.append(mapped)
        if mapped.inverse_residual <= residual_tolerance:
            return candidates
    return candidates


def _map_image_point_in_quadrilateral(
    point: complex,
    state: CRDTSolverState,
    qid: int,
    continuation: tuple[complex, complex] | None = None,
) -> _MappedPoint:
    quad = state.topology.quadrilaterals[qid]
    image_vertices = state.image_vertices
    ws = state.ws[qid]
    distances = np.asarray([abs(point - image_vertices[vid]) for vid in quad.vids], dtype=float)
    local_vid = int(np.argmin(distances))
    vid = int(quad.vids[local_vid])
    method = "vertex"
    inverse_residual = 0.0
    used_continuation = False

    try:
        if distances[local_vid] <= 1.0e-10:
            w = ws[vid]
            vertex_index = vid
        else:
            z2 = state.image_a[qid]
            w2 = 0.0 + 0.0j
            if continuation is not None:
                z2, w2 = continuation
                used_continuation = True
            try:
                inverse = state.image_sc.z2w_result(
                    point,
                    ws,
                    z1=image_vertices[vid],
                    w1=ws[vid],
                    z2=z2,
                    w2=w2,
                    z0=state.image_a[qid],
                    w0=0.0 + 0.0j,
                    scale=state.image_b[qid],
                    tolerance=1.0e-9,
                    max_iterations=30,
                    use_ode_fallback=True,
                )
                if inverse.residual > _inverse_residual_tolerance(point, state.image_vertices):
                    raise CRDTError("SC inverse residual exceeds tolerance")
            except CRDTError:
                if (boundary_mapped := _map_boundary_edge_point(point, state, qid)) is not None:
                    return boundary_mapped
                raise
            w = inverse.w
            method = inverse.method
            inverse_residual = inverse.residual
            vertex_index = -1
        mapped = state.physical_sc.w2z(
            w,
            ws,
            vertex_index=vertex_index,
            w0=0.0 + 0.0j,
            z0=state.physical_a[qid],
            scale=state.physical_b[qid],
        )
    except CRDTError:
        raise
    return _MappedPoint(
        z=mapped,
        qid=qid,
        w=w,
        method=method,
        inverse_residual=inverse_residual,
        used_continuation=used_continuation,
    )


def _map_boundary_edge_point(
    point: complex,
    state: CRDTSolverState,
    qid: int,
) -> _MappedPoint | None:
    quad = state.topology.quadrilaterals[qid]
    nvertices = len(state.image_vertices)
    for edge_idx in range(4):
        vid0 = int(quad.vids[edge_idx])
        vid1 = int(quad.vids[(edge_idx + 1) % 4])
        diff = abs(vid0 - vid1)
        if diff != 1 and diff != nvertices - 1:
            continue
        fraction = _segment_fraction_if_on(
            point, state.image_vertices[vid0], state.image_vertices[vid1]
        )
        if fraction is None:
            continue
        z = state.zs[vid0] * (1.0 - fraction) + state.zs[vid1] * fraction
        w = state.ws[qid, vid0] * (1.0 - fraction) + state.ws[qid, vid1] * fraction
        if abs(w) > 0.0:
            w /= abs(w)
        return _MappedPoint(
            z=z,
            qid=qid,
            w=w,
            method="boundary",
            inverse_residual=0.0,
            used_continuation=False,
        )
    return None


def _segment_fraction_if_on(point: complex, start: complex, stop: complex) -> float | None:
    edge = stop - start
    length_squared = float(edge.real * edge.real + edge.imag * edge.imag)
    if length_squared <= 1.0e-28:
        return 0.0 if abs(point - start) <= 1.0e-10 else None
    fraction = float(
        ((point - start).real * edge.real + (point - start).imag * edge.imag) / length_squared
    )
    if fraction < -1.0e-10 or fraction > 1.0 + 1.0e-10:
        return None
    projection = start + fraction * edge
    if abs(projection - point) > max(1.0e-10, 1.0e-10 * abs(edge)):
        return None
    return min(1.0, max(0.0, fraction))


def _inverse_residual_tolerance(point: complex, image_vertices: np.ndarray) -> float:
    span = max(
        float(np.ptp(image_vertices.real)),
        float(np.ptp(image_vertices.imag)),
        1.0,
    )
    return max(1.0e-6 * span, 1.0e-9 * max(abs(point), 1.0))


def _record_mapping_stats(
    stats: _MappingStats,
    mapped: _MappedPoint,
    image_point: complex,
    node_index: tuple[int, int] | None,
) -> None:
    stats.inverse_evaluations += 1
    if mapped.method == "secant":
        stats.secant_solves += 1
    elif mapped.method == "newton":
        stats.newton_solves += 1
    elif mapped.method == "root":
        stats.root_solves += 1
    elif mapped.method == "least_squares":
        stats.least_squares_solves += 1
    elif mapped.method == "ode":
        stats.ode_fallbacks += 1
    if mapped.used_continuation:
        stats.continuation_uses += 1
    if mapped.inverse_residual > stats.max_inverse_residual:
        stats.max_inverse_residual = mapped.inverse_residual
        stats.worst_node_index = node_index
        stats.worst_qid = mapped.qid
        stats.worst_image_point = image_point


def _locate_image_quadrilateral(point: complex, state: CRDTSolverState) -> int | None:
    candidates = _candidate_image_quadrilaterals(point, state)
    return candidates[0] if candidates else None


def _candidate_image_quadrilaterals(
    point: complex,
    state: CRDTSolverState,
    preferred_qid: int | None = None,
) -> list[int]:
    candidates: list[int] = []
    if (
        preferred_qid is not None
        and state.image_quadrilateral_polygons is not None
        and 0 <= preferred_qid < len(state.image_quadrilateral_polygons)
        and _point_in_complex_polygon(point, state.image_quadrilateral_polygons[preferred_qid])
    ):
        candidates.append(preferred_qid)
    if state.image_quadrilateral_polygons is not None:
        for qid, polygon in enumerate(state.image_quadrilateral_polygons):
            if qid == preferred_qid:
                continue
            if _point_in_complex_polygon(point, polygon):
                candidates.append(qid)
        if candidates:
            return candidates
    points = np.column_stack((state.image_vertices.real, state.image_vertices.imag))
    xy = np.asarray([point.real, point.imag], dtype=float)
    for quad in state.topology.quadrilaterals:
        if quad.id == preferred_qid:
            continue
        if _point_in_quad(xy, points[list(quad.vids)]):
            candidates.append(quad.id)
    return candidates


def _point_in_complex_polygon(point: complex, polygon: np.ndarray) -> bool:
    x = np.asarray([[point.real]], dtype=float)
    y = np.asarray([[point.imag]], dtype=float)
    points = np.column_stack((polygon.real, polygon.imag))
    return bool(_points_in_polygon(x, y, points)[0, 0])


def _point_in_quad(point: np.ndarray, quad: np.ndarray) -> bool:
    return bool(
        np.all(_barycentric_weights(point, quad[[0, 1, 2]]) >= -1.0e-10)
        or np.all(_barycentric_weights(point, quad[[0, 2, 3]]) >= -1.0e-10)
    )


def _barycentric_weights(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    matrix = np.asarray(
        [
            [triangle[0, 0], triangle[1, 0], triangle[2, 0]],
            [triangle[0, 1], triangle[1, 1], triangle[2, 1]],
            [1.0, 1.0, 1.0],
        ]
    )
    rhs = np.asarray([point[0], point[1], 1.0])
    try:
        return np.linalg.solve(matrix, rhs)
    except np.linalg.LinAlgError:
        return np.asarray([-np.inf, -np.inf, -np.inf])


def _finite_cell_mask(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    finite_nodes = np.isfinite(x) & np.isfinite(y)
    return (
        finite_nodes[:-1, :-1]
        & finite_nodes[1:, :-1]
        & finite_nodes[:-1, 1:]
        & finite_nodes[1:, 1:]
    )


def _repair_duplicate_run_nodes(x: np.ndarray, y: np.ndarray) -> None:
    finite = np.isfinite(x) & np.isfinite(y)
    for row in range(x.shape[0]):
        _repair_duplicate_runs_1d(x[row], y[row], finite[row])
    for column in range(x.shape[1]):
        _repair_duplicate_runs_1d(x[:, column], y[:, column], finite[:, column])


def _repair_duplicate_runs_1d(
    values_x: np.ndarray, values_y: np.ndarray, finite: np.ndarray
) -> None:
    start: int | None = None
    for idx, is_finite in enumerate(np.append(finite, False)):
        if is_finite and start is None:
            start = idx
        elif not is_finite and start is not None:
            _repair_duplicate_run(values_x, values_y, start, idx)
            start = None


def _repair_duplicate_run(
    values_x: np.ndarray, values_y: np.ndarray, start: int, stop: int
) -> None:
    if stop - start < 3:
        return
    segment = np.column_stack((values_x[start:stop], values_y[start:stop]))
    lengths = np.hypot(np.diff(segment[:, 0]), np.diff(segment[:, 1]))
    positive = lengths[lengths > 1.0e-10]
    if len(positive) == 0:
        return
    threshold = max(1.0e-8, 0.05 * float(np.median(positive)))
    for local_idx, length in enumerate(lengths):
        if length > threshold:
            continue
        idx = start + local_idx
        if idx > start and idx + 1 < stop:
            values_x[idx] = 0.5 * (values_x[idx - 1] + values_x[idx + 1])
            values_y[idx] = 0.5 * (values_y[idx - 1] + values_y[idx + 1])
        elif idx == start and idx + 2 < stop:
            values_x[idx + 1] = 0.5 * (values_x[idx] + values_x[idx + 2])
            values_y[idx + 1] = 0.5 * (values_y[idx] + values_y[idx + 2])
        elif idx + 1 == stop - 1 and idx - 1 >= start:
            values_x[idx] = 0.5 * (values_x[idx - 1] + values_x[idx + 1])
            values_y[idx] = 0.5 * (values_y[idx - 1] + values_y[idx + 1])
