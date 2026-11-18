from dataclasses import dataclass, replace
from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class LinearConstraint:
    """A two-dimensional half-space constraint ``coefficients @ u >= lower``."""

    coefficients: np.ndarray
    lower: float
    name: str
    kind: str

    def __post_init__(self):
        coefficients = np.asarray(self.coefficients, dtype=float).reshape(2)
        object.__setattr__(self, "coefficients", coefficients)

    def residual(self, control: np.ndarray) -> float:
        return float(self.coefficients @ control - self.lower)


@dataclass(frozen=True)
class QPSolution:
    control: np.ndarray
    hard_feasible: bool
    max_violation: float
    objective: float
    normalized_violation: float = 0.0
    fallback_mode: str = "none"
    mandatory_feasible: bool = True


def solve_weighted_projection(
    reference: np.ndarray,
    weights: np.ndarray,
    constraints: Sequence[LinearConstraint],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    tolerance: float = 1e-7,
    fallback_penalty: float = 1000.0,
    fallback_iterations: int = 120,
) -> QPSolution:
    """Solve a strictly convex two-variable QP without an external QP package.

    The hard-feasible optimum is the weighted projection of ``reference`` onto
    a polygon. In two dimensions it must be the reference itself, a projection
    onto one active edge, or an intersection of two active edges, so all such
    candidates can be enumerated exactly.

    If the CBF constraints have an empty intersection with the physical input
    box, a bounded squared-hinge fallback is used. The fallback is explicitly
    marked infeasible by the returned result.
    """

    reference = np.asarray(reference, dtype=float).reshape(2)
    weights = np.asarray(weights, dtype=float).reshape(2)
    lower_bounds = np.asarray(lower_bounds, dtype=float).reshape(2)
    upper_bounds = np.asarray(upper_bounds, dtype=float).reshape(2)
    if np.any(weights <= 0):
        raise ValueError("QP weights must be positive")
    if np.any(lower_bounds > upper_bounds):
        raise ValueError("lower bounds must not exceed upper bounds")

    all_constraints = list(constraints) + _box_constraints(lower_bounds, upper_bounds)
    sqrt_w = np.sqrt(weights)
    inv_sqrt_w = 1.0 / sqrt_w
    transformed = []
    for item in all_constraints:
        normal = item.coefficients * inv_sqrt_w
        rhs = float(item.lower - item.coefficients @ reference)
        transformed.append((normal, rhs))

    candidates: List[np.ndarray] = [np.zeros(2, dtype=float)]
    for normal, rhs in transformed:
        norm_sq = float(normal @ normal)
        if norm_sq > 1e-14:
            candidates.append(normal * rhs / norm_sq)

    for first, second in combinations(transformed, 2):
        matrix = np.vstack([first[0], second[0]])
        if abs(float(np.linalg.det(matrix))) <= 1e-12:
            continue
        candidates.append(np.linalg.solve(matrix, np.asarray([first[1], second[1]], dtype=float)))

    feasible_controls = []
    for candidate in candidates:
        control = reference + inv_sqrt_w * candidate
        if _is_feasible(control, all_constraints, tolerance):
            objective = float(np.sum(weights * np.square(control - reference)))
            feasible_controls.append((objective, control))

    if feasible_controls:
        objective, control = min(feasible_controls, key=lambda item: item[0])
        return QPSolution(
            control=np.asarray(control, dtype=float),
            hard_feasible=True,
            max_violation=0.0,
            objective=objective,
        )

    control = _penalty_fallback(
        reference=reference,
        weights=weights,
        constraints=constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        penalty=fallback_penalty,
        iterations=fallback_iterations,
    )
    violations = [max(0.0, -item.residual(control)) for item in constraints]
    objective = float(np.sum(weights * np.square(control - reference)))
    return QPSolution(
        control=control,
        hard_feasible=False,
        max_violation=max(violations, default=0.0),
        objective=objective,
        normalized_violation=_max_normalized_violation(
            constraints, control, lower_bounds, upper_bounds
        ),
        fallback_mode="unprioritized_penalty",
    )


def solve_prioritized_projection(
    reference: np.ndarray,
    weights: np.ndarray,
    constraints: Sequence[LinearConstraint],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    mandatory_kinds: Sequence[str] = ("road",),
    tolerance: float = 1e-7,
    fallback_penalty: float = 1000.0,
    fallback_iterations: int = 120,
    precondition: bool = False,
) -> QPSolution:
    """Project onto all constraints, then relax only non-mandatory constraints.

    Road constraints remain hard whenever their intersection with the physical
    input box is non-empty. Vehicle violations are normalized before the
    fallback penalty is applied so a large polynomial gradient cannot dominate
    merely because of its numerical scale.
    """

    if precondition:
        # z=sqrt(W)u makes the tracking Hessian isotropic. Both the constraints
        # and input box are transformed, so this does not change the QP or its
        # normalized soft penalty. It avoids slow fallback convergence when
        # lateral acceleration weighting produces W_kappa proportional to v^4.
        metric = np.asarray(weights, dtype=float).reshape(2)
        if np.any(metric <= 0.) or not np.all(np.isfinite(metric)):
            raise ValueError('QP weights must be positive and finite')
        root = np.sqrt(metric)
        transformed = [LinearConstraint(c.coefficients/root, c.lower, c.name, c.kind)
                       for c in constraints]
        result = solve_prioritized_projection(
            np.asarray(reference)*root, np.ones(2), transformed,
            np.asarray(lower_bounds)*root, np.asarray(upper_bounds)*root,
            mandatory_kinds, tolerance, fallback_penalty, fallback_iterations,
            precondition=False,
        )
        return replace(result, control=result.control/root)

    full_solution = solve_weighted_projection(
        reference=reference,
        weights=weights,
        constraints=constraints,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        tolerance=tolerance,
        fallback_penalty=fallback_penalty,
        fallback_iterations=fallback_iterations,
    )
    if full_solution.hard_feasible:
        return full_solution

    mandatory_kind_set = set(mandatory_kinds)
    mandatory = [item for item in constraints if item.kind in mandatory_kind_set]
    soft = [item for item in constraints if item.kind not in mandatory_kind_set]
    mandatory_projection = solve_weighted_projection(
        reference=reference,
        weights=weights,
        constraints=mandatory,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        tolerance=tolerance,
        fallback_penalty=fallback_penalty,
        fallback_iterations=fallback_iterations,
    )

    if not mandatory_projection.hard_feasible:
        control = mandatory_projection.control
        violations = [max(0.0, -item.residual(control)) for item in constraints]
        return QPSolution(
            control=control,
            hard_feasible=False,
            max_violation=max(violations, default=0.0),
            objective=float(np.sum(weights * np.square(control - reference))),
            normalized_violation=_max_normalized_violation(
                constraints, control, lower_bounds, upper_bounds
            ),
            fallback_mode="mandatory_infeasible",
            mandatory_feasible=False,
        )

    control = mandatory_projection.control
    if soft:
        normals, rhs = _normalized_constraint_system(soft, lower_bounds, upper_bounds)
        lipschitz = 2.0 * float(np.max(weights)) + 2.0 * fallback_penalty * float(
            np.sum(np.square(normals))
        )
        step = 1.0 / max(lipschitz, 1e-9)
        for _ in range(fallback_iterations):
            deficits = np.maximum(0.0, rhs - normals @ control)
            gradient = (
                2.0 * weights * (control - reference)
                - 2.0 * fallback_penalty * (normals.T @ deficits)
            )
            trial = control - step * gradient
            projected = solve_weighted_projection(
                reference=trial,
                weights=np.ones(2, dtype=float),
                constraints=mandatory,
                lower_bounds=lower_bounds,
                upper_bounds=upper_bounds,
                tolerance=tolerance,
                fallback_penalty=fallback_penalty,
                fallback_iterations=fallback_iterations,
            )
            next_control = projected.control
            if np.linalg.norm(next_control - control) <= 1e-10:
                control = next_control
                break
            control = next_control

    all_constraints = list(constraints) + _box_constraints(lower_bounds, upper_bounds)
    hard_feasible = _is_feasible(control, all_constraints, tolerance)
    violations = [max(0.0, -item.residual(control)) for item in constraints]
    return QPSolution(
        control=control,
        hard_feasible=hard_feasible,
        max_violation=max(violations, default=0.0),
        objective=float(np.sum(weights * np.square(control - reference))),
        normalized_violation=_max_normalized_violation(
            constraints, control, lower_bounds, upper_bounds
        ),
        fallback_mode="none" if hard_feasible else "relaxed_vehicle",
        mandatory_feasible=True,
    )


def _normalized_constraint_system(
    constraints: Sequence[LinearConstraint],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    if not constraints:
        return np.empty((0, 2), dtype=float), np.empty(0, dtype=float)
    normals = np.vstack([item.coefficients for item in constraints])
    rhs = np.asarray([item.lower for item in constraints], dtype=float)
    spans = np.maximum(np.asarray(upper_bounds) - np.asarray(lower_bounds), 1e-6)
    scales = np.maximum(1.0, np.abs(rhs))
    scales = np.maximum(scales, np.sum(np.abs(normals) * spans, axis=1))
    return normals / scales[:, None], rhs / scales


def _max_normalized_violation(
    constraints: Sequence[LinearConstraint],
    control: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> float:
    normals, rhs = _normalized_constraint_system(constraints, lower_bounds, upper_bounds)
    if not len(rhs):
        return 0.0
    return float(np.max(np.maximum(0.0, rhs - normals @ control)))

def _box_constraints(lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> List[LinearConstraint]:
    return [
        LinearConstraint(np.asarray([1.0, 0.0]), float(lower_bounds[0]), "acceleration_min", "physical"),
        LinearConstraint(np.asarray([-1.0, 0.0]), float(-upper_bounds[0]), "acceleration_max", "physical"),
        LinearConstraint(np.asarray([0.0, 1.0]), float(lower_bounds[1]), "curvature_min", "physical"),
        LinearConstraint(np.asarray([0.0, -1.0]), float(-upper_bounds[1]), "curvature_max", "physical"),
    ]


def _is_feasible(
    control: np.ndarray,
    constraints: Iterable[LinearConstraint],
    tolerance: float,
) -> bool:
    return all(item.residual(control) >= -tolerance for item in constraints)


def _penalty_fallback(
    reference: np.ndarray,
    weights: np.ndarray,
    constraints: Sequence[LinearConstraint],
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    penalty: float,
    iterations: int,
) -> np.ndarray:
    """Minimize a convex squared-hinge relaxation inside the physical box."""

    control = np.clip(reference, lower_bounds, upper_bounds)
    if not constraints:
        return control

    normals = np.vstack([item.coefficients for item in constraints])
    rhs = np.asarray([item.lower for item in constraints], dtype=float)
    lipschitz = 2.0 * float(np.max(weights)) + 2.0 * penalty * float(np.sum(np.square(normals)))
    step = 1.0 / max(lipschitz, 1e-9)

    for _ in range(iterations):
        deficits = np.maximum(0.0, rhs - normals @ control)
        gradient = 2.0 * weights * (control - reference) - 2.0 * penalty * (normals.T @ deficits)
        next_control = np.clip(control - step * gradient, lower_bounds, upper_bounds)
        if np.linalg.norm(next_control - control) <= 1e-10:
            control = next_control
            break
        control = next_control
    return control
