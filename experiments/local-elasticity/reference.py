"""Independent Float64 CPU reference for the bounded local-elasticity experiment.

This is an audit implementation, never a production simulation dependency. It uses
NumPy dense solves/eigh, independently assembled RKPM moments and physical energy.
FreeForm equations: https://research.nvidia.com/labs/sil/projects/freeform/assets/main.pdf
The six-mode canary includes an exact constant mode; this differs from the paper's
mode-count convention. Point-cloud quadrature weights are an assumed measure, not
evidence of a recovered solid or calibrated material properties. A refined basis
using this same quadrature/energy is a convergence reference, not FEM ground truth.

Run ``python reference.py --self-test``. To audit JavaScript values, write JSON:
  {"basisCases":[{"input":{positions,volumes,young,poisson},"actual":basis}],
   "mechanicsCases":[{"input":{restPositions,volumes,weights,gradients,density,
       young,poisson},"states":[{"q":[],"energy":0,"gradient":[],
       "positions":[],"deformationGradients":[]}],"steps":[{
       "qBefore":[],"velocityBefore":[],"qAfter":[],"dt":0.016,
       "gravity":[0,-9.81,0],"pins":[],"drag":null,"floorHeight":null}]}]}
Then ``python reference.py --report path.json --out new-audit.json``. Output is
created exclusively: old reports are never overwritten. Actual basis must include
centers/radii/origin/scale/coefficients/eigenvalues, optionally mass/stiffness and
evaluated weights/gradients. Arrays are flat, row-major. q is H x 3 x 4, with the
last dimension [x,y,z,1]. The reference does not infer a stable basis from signs of
individual eigenvectors; it checks residuals, mass orthogonality and subspaces.
"""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def array(value, shape=None, name="array"):
    result = np.asarray(value, dtype=np.float64)
    if shape is not None:
        result = result.reshape(shape)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must be finite")
    return result


def lame(young, poisson):
    young, poisson = float(young), float(poisson)
    if not (np.isfinite(young) and young > 0 and -1 < poisson < .5):
        raise ValueError("Require young > 0 and -1 < poisson < .5")
    mu = young / (2 * (1 + poisson))
    lam = young * poisson / ((1 + poisson) * (1 - 2 * poisson))
    return lam, mu


def rkpm(positions, centers, radii, origin=None, scale=None):
    """Affine-reproducing Gaussian RKPM values and *world-space* gradients.

    Moment solves are exact dense solves, with no hidden ridge or radius tuning.
    Explicit radii let this independently check JS even if center selection differs.
    """
    x = array(positions, (-1, 3), "positions")
    c = array(centers, (-1, 3), "centers")
    r = array(radii, (len(c),), "radii")
    if len(c) < 4 or np.any(r <= 0):
        raise ValueError("Need at least four centers and positive radii")
    origin = (c.min(0) + c.max(0)) / 2 if origin is None else array(origin, (3,))
    scale = float(np.ptp(c, axis=0).max() if scale is None else scale)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")
    pc = np.column_stack((np.ones(len(c)), (c - origin) / scale))
    px = np.column_stack((np.ones(len(x)), (x - origin) / scale))
    delta = x[:, None, :] - c[None, :, :]
    raw = np.exp(-np.sum(delta * delta, axis=2) / (r * r))
    draw = -2 * delta * raw[:, :, None] / (r[None, :, None] ** 2)
    moment = np.einsum("nk,ka,kb->nab", raw, pc, pc)
    correction = np.linalg.solve(moment, px[..., None])[..., 0]
    dc_rhs = np.broadcast_to(np.vstack((np.zeros(3), np.eye(3) / scale)),
                             (len(x), 4, 3)).copy()
    dmoment = np.einsum("nkj,ka,kb->najb", draw, pc, pc)
    dc_rhs -= np.einsum("najb,nb->naj", dmoment, correction)
    dcorrection = np.linalg.solve(moment, dc_rhs)
    p_dot_c = correction @ pc.T
    phi = raw * p_dot_c
    dphi = draw * p_dot_c[:, :, None] + raw[:, :, None] * np.einsum(
        "ka,naj->nkj", pc, dcorrection)
    return phi, dphi


def assemble_basis(positions, volumes, centers, radii, young=1000, poisson=.3,
                   origin=None, scale=None):
    phi, dphi = rkpm(positions, centers, radii, origin, scale)
    v = array(volumes, (len(phi),), "volumes")
    if np.any(v <= 0):
        raise ValueError("quadrature volumes must be positive")
    lam, mu = lame(young, poisson)
    mass = np.einsum("nk,n,nl->kl", phi, v, phi)
    stiffness = (lam + 4 * mu) * np.einsum("nkj,n,nlj->kl", dphi, v, dphi)
    return {"phi": phi, "dphi": dphi, "mass": mass, "stiffness": stiffness}


def generalized_eigh(stiffness, mass):
    """Independent Cholesky whitening followed by LAPACK symmetric dense eigh."""
    mass, stiffness = array(mass), array(stiffness)
    lower = np.linalg.cholesky(mass)
    left = np.linalg.solve(lower, stiffness)
    whitened = np.linalg.solve(lower, left.T).T
    values, vectors = np.linalg.eigh((whitened + whitened.T) / 2)
    coefficients = np.linalg.solve(lower.T, vectors)
    return values, coefficients


def evaluate_basis(positions, basis):
    centers = array(basis["centers"], (-1, 3))
    phi, dphi = rkpm(positions, centers, basis["radii"], basis.get("origin"),
                     basis.get("scale"))
    coefficients = array(basis["coefficients"], (len(centers), -1))
    return phi @ coefficients, np.einsum("nkj,kh->nhj", dphi, coefficients)


def cofactor(f):
    """Cofactor remains defined at singular F; no inverse/determinant division."""
    return np.stack((np.cross(f[..., :, 1], f[..., :, 2]),
                     np.cross(f[..., :, 2], f[..., :, 0]),
                     np.cross(f[..., :, 0], f[..., :, 1])), axis=-1)


def neo_hookean(f, young, poisson):
    """Stable Neo-Hookean density, first Piola stress, and determinant."""
    f = array(f)
    lam, mu = lame(young, poisson)
    bulk = lam + mu
    gamma = 1 + mu / bulk
    jacobian = np.linalg.det(f)
    density = .5 * (mu * (np.sum(f * f, axis=(-1, -2)) - 3) + bulk *
                    ((jacobian - gamma) ** 2 - (1 - gamma) ** 2))
    stress = mu * f + (bulk * (jacobian - gamma))[..., None, None] * cofactor(f)
    return density, stress, jacobian


class Mechanics:
    """External affine-coordinate model with explicit physical mass and energy."""

    def __init__(self, config):
        self.x = array(config["restPositions"], (-1, 3), "restPositions")
        self.n = len(self.x)
        self.volumes = array(config["volumes"], (self.n,), "volumes")
        self.weights = array(config["weights"], (self.n, -1), "weights")
        self.h = self.weights.shape[1]
        self.gradients = array(config["gradients"], (self.n, self.h, 3), "gradients")
        self.density = float(config.get("density", 1))
        self.young, self.poisson = config.get("young", 1000), config.get("poisson", .3)
        lame(self.young, self.poisson)
        if self.density <= 0 or np.any(self.volumes <= 0):
            raise ValueError("Positive mass density and volumes required")
        self.mass = self.density * self.volumes
        homogeneous = np.column_stack((self.x, np.ones(self.n)))
        self.b = self.weights[:, :, None] * homogeneous[:, None, :]
        self.g = self.gradients[:, :, None, :] * homogeneous[:, None, :, None]
        self.g[:, :, :3, :] += self.weights[:, :, None, None] * np.eye(3)
        self.scalar_gram = np.einsum("nha,n,nkb->hakb", self.b, self.mass, self.b)
        # q flat order: handle, output coordinate, affine coefficient.
        self.mass_matrix = np.einsum("hakb,cd->hcakdb", self.scalar_gram, np.eye(3)).reshape(
            (12 * self.h, 12 * self.h))

    def state(self, q):
        q = array(q, (self.h, 3, 4), "q")
        position = self.x + np.einsum("nha,hca->nc", self.b, q)
        f = np.eye(3) + np.einsum("nhaj,hca->ncj", self.g, q)
        density, stress, jacobian = neo_hookean(f, self.young, self.poisson)
        gradient = np.einsum("n,ncj,nhaj->hca", self.volumes, stress, self.g)
        return {"energy": float(self.volumes @ density), "gradient": gradient.ravel(),
                "positions": position, "deformationGradients": f,
                "determinants": jacobian}

    def objective(self, q, q_before, velocity_before, dt, gravity=(0, -9.81, 0), drag=None):
        q = array(q, (12 * self.h,))
        q_before = array(q_before, q.shape)
        velocity_before = array(velocity_before, q.shape)
        dt = float(dt)
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("Positive finite timestep required")
        gravity = array(gravity, (3,))
        state = self.state(q)
        error = q - q_before - dt * velocity_before
        inertial_gradient = self.mass_matrix @ error / dt ** 2
        inertia = .5 * float(error @ inertial_gradient)
        gravity_energy = -float(np.sum(self.mass[:, None] * state["positions"] * gravity))
        gravity_gradient = -np.einsum("n,nha,c->hca", self.mass, self.b, gravity).ravel()
        gradient = state["gradient"] + inertial_gradient + gravity_gradient
        spring_energy = 0.
        if drag is not None:
            i = int(drag["index"])
            stiffness = float(drag["stiffness"])
            if not 0 <= i < self.n or not np.isfinite(stiffness) or stiffness < 0:
                raise ValueError("Invalid drag point/stiffness")
            difference = state["positions"][i] - array(drag["target"], (3,))
            spring_energy = .5 * stiffness * float(difference @ difference)
            gradient += (stiffness * np.einsum("ha,c->hca", self.b[i], difference)).ravel()
        return {"value": state["energy"] + inertia + gravity_energy + spring_energy,
                "gradient": gradient, "elastic": state["energy"], "inertia": inertia,
                "gravity": gravity_energy, "spring": spring_energy}


def difference(actual, expected):
    actual, expected = array(actual), array(expected)
    if actual.size != expected.size:
        raise ValueError(f"Array length mismatch: {actual.size} versus {expected.size}")
    d = actual.ravel() - expected.ravel()
    return {"maxAbs": float(np.max(np.abs(d), initial=0)),
            "relativeL2": float(np.linalg.norm(d) / max(1., np.linalg.norm(expected)))}


def require_close(actual, expected, label, atol=1e-8, rtol=1e-7):
    delta = difference(actual, expected)
    a, b = array(actual).ravel(), array(expected).ravel()
    if not np.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{label} mismatch: {delta}")
    return delta


def audit_basis(case):
    inp, actual = case["input"], case["actual"]
    x = array(inp["positions"], (-1, 3))
    v = array(inp["volumes"], (len(x),))
    centers = array(actual["centers"], (-1, 3))
    matrices = assemble_basis(x, v, centers, actual["radii"], inp.get("young", 1000),
        inp.get("poisson", .3), actual.get("origin"), actual.get("scale"))
    phi, dphi, mass, stiffness = (matrices[k] for k in ("phi", "dphi", "mass", "stiffness"))
    coeff = array(actual["coefficients"], (len(centers), -1))
    modes = coeff.shape[1]
    if not 2 <= modes < len(centers):
        raise ValueError("Audit expects nonconstant modes followed by one constant mode")
    values, vectors = generalized_eigh(stiffness, mass)
    supplied_values = array(actual["eigenvalues"], (modes - 1,))
    details = {"points": len(x), "centers": len(centers), "modesIncludingConstant": modes,
        "massCondition": float(np.linalg.cond(mass)), "oracleEigenvalues": values[:modes + 1].tolist()}
    details["partitionUnity"] = require_close(phi.sum(1), np.ones(len(x)), "RKPM partition")
    details["affineReproduction"] = require_close(phi @ centers, x, "RKPM affine reproduction")
    details["gradientUnity"] = require_close(dphi.sum(1), np.zeros((len(x), 3)), "gradient partition")
    details["affineGradient"] = require_close(np.einsum("nkj,ka->naj", dphi, centers),
        np.broadcast_to(np.eye(3), (len(x), 3, 3)), "affine gradient")
    details["constantCoefficients"] = require_close(coeff[:, -1], np.ones(len(centers)), "constant coefficients")
    details["selectedEigenvalues"] = require_close(supplied_values, values[1:modes],
        "smallest nonconstant eigenvalues", atol=1e-6, rtol=2e-6)
    expected_gram = np.diag([1.] * (modes - 1) + [float(v.sum())])
    details["massOrthogonality"] = require_close(coeff.T @ mass @ coeff, expected_gram,
        "basis mass orthogonality", atol=2e-7, rtol=2e-7)
    residual = stiffness @ coeff[:, :-1] - (mass @ coeff[:, :-1]) * supplied_values
    denominator = np.linalg.norm(stiffness @ coeff[:, :-1], axis=0) + np.abs(supplied_values) * np.linalg.norm(mass @ coeff[:, :-1], axis=0)
    relative = np.linalg.norm(residual, axis=0) / np.maximum(1e-12, denominator)
    details["generalizedResiduals"] = relative.tolist()
    if relative.max(initial=0) > 2e-6:
        raise AssertionError(f"Generalized eigen residual too large: {relative}")
    overlap = vectors[:, 1:modes].T @ mass @ coeff[:, :-1]
    singular = np.linalg.svd(overlap, compute_uv=False)
    details["subspaceSingularValues"] = singular.tolist()
    # A truncation through a repeated eigenvalue need not select the same subspace.
    separated = values[modes] - values[modes - 1] > 1e-5 * max(1., abs(values[modes]))
    details["subspaceGateApplicable"] = bool(separated)
    if separated and np.min(singular) < 1 - 1e-5:
        raise AssertionError("Selected eigenspace disagrees with NumPy reference")
    for name, expected in (("mass", mass), ("stiffness", stiffness),
                            ("massMatrix", mass), ("stiffnessMatrix", stiffness),
                            ("weights", phi @ coeff),
                            ("gradients", np.einsum("nkj,kh->nhj", dphi, coeff))):
        if name in actual:
            details[name] = require_close(actual[name], expected, name)
    return details


def audit_mechanics(case):
    model = Mechanics(case["input"])
    output = {"points": model.n, "handles": model.h, "states": [], "steps": []}
    if "massMatrix" in case:
        # Public JS massMatrix stores the scalar 4H Gram matrix. Full external
        # q mass is its three-coordinate expansion with h,c,a ordering.
        output["massMatrix"] = require_close(case["massMatrix"],
            model.scalar_gram.reshape((4 * model.h, 4 * model.h)), "scalar mass matrix")
    for item in case.get("states", []):
        expected = model.state(item["q"])
        checked = {}
        for key in ("energy", "gradient", "positions", "deformationGradients", "determinants"):
            if key in item:
                checked[key] = require_close(item[key], expected[key], key, atol=2e-7, rtol=2e-6)
        if not checked:
            raise ValueError("A state must retain at least one independently checkable output")
        if "hessian" in item and item["hessian"] is not None:
            q = array(item["q"], (12 * model.h,))
            columns = []
            for index in range(len(q)):
                offset = np.zeros_like(q)
                offset[index] = 1e-6
                columns.append((model.state(q + offset)["gradient"] -
                                model.state(q - offset)["gradient"]) / 2e-6)
            fd_hessian = np.column_stack(columns)
            checked["hessian"] = require_close(item["hessian"], fd_hessian,
                "elastic Hessian vs independent stress differences", atol=2e-5, rtol=2e-5)
        checked["minDeterminant"] = float(expected["determinants"].min())
        output["states"].append(checked)
    for step in case.get("steps", []):
        args = (step["qBefore"], step["velocityBefore"], step["dt"],
                step.get("gravity", [0, -9.81, 0]), step.get("drag"))
        before = model.objective(step["qBefore"], *args)
        after = model.objective(step["qAfter"], *args)
        positions = model.state(step["qAfter"])["positions"]
        # Audits the externally visible objective and feasibility, not equivalence
        # of every Newton direction or proof of a global/local physical optimum.
        pins = step.get("pins", [])
        pin_error = max((float(np.linalg.norm(positions[int(pin["index"])] -
                    array(pin["target"], (3,)))) for pin in pins), default=0.)
        floor = step.get("floorHeight")
        floor_violation = 0. if floor is None else max(0., float(floor) - float(positions[:, 1].min()))
        result = {"beforeObjective": before["value"], "afterObjective": after["value"],
            "objectiveChange": after["value"] - before["value"],
            "externalGradientNorm": float(np.linalg.norm(after["gradient"])),
            "maxPinError": pin_error, "maxFloorViolation": floor_violation}
        result["stationarity"] = constrained_stationarity(model, after["gradient"], positions,
            pins, floor, case["input"].get("massRelativeTolerance", 1e-10))
        if step.get("diagnostics", {}).get("converged") and result["stationarity"]["projectedGradientNorm"] > 3e-6:
            raise AssertionError(f"Claimed convergence has a nonstationary feasible direction: {result}")
        if "velocityAfter" in step:
            result["velocity"] = require_close(step["velocityAfter"],
                (array(step["qAfter"]) - array(step["qBefore"])) / float(step["dt"]),
                "external finite step velocity", atol=2e-9)
        if pin_error > 2e-6 or floor_violation > 2e-6:
            raise AssertionError(f"Step constraint violation: {result}")
        # Moving pins may make the old state infeasible; only compare when feasible.
        old_position = model.state(step["qBefore"])["positions"]
        old_pin_error = max((float(np.linalg.norm(old_position[int(pin["index"])] -
                       array(pin["target"], (3,)))) for pin in pins), default=0.)
        old_floor_ok = floor is None or float(old_position[:, 1].min()) >= float(floor) - 2e-6
        result["objectiveDescentGateApplicable"] = old_pin_error <= 2e-6 and old_floor_ok
        if result["objectiveDescentGateApplicable"] and result["objectiveChange"] > 1e-7 * max(1., abs(before["value"])):
            raise AssertionError(f"Feasible step increased incremental potential: {result}")
        for key, expected in (("beforeObjective", before["value"]), ("afterObjective", after["value"])):
            if key in step:
                result[key + "Check"] = require_close(step[key], expected, key)
        output["steps"].append(result)
    if not output["states"] and not output["steps"]:
        raise ValueError("Empty mechanics case")
    return output


def constrained_stationarity(model, gradient, positions, pins, floor, mass_tolerance):
    """Fresh NumPy mass whitening and SVD constraint nullspace; no JS solver reuse.

    Raw external gradients include pin reactions, so their norm is not a valid
    stationarity test. This computes the free gradient in unit-mass coordinates.
    It is a first-order diagnostic, not a proof of positive curvature/uniqueness.
    """
    scalar_size = 4 * model.h
    mass = model.scalar_gram.reshape((scalar_size, scalar_size))
    values, eigenvectors = np.linalg.eigh(mass)
    keep = values > values[-1] * mass_tolerance
    transform = eigenvectors[:, keep] / np.sqrt(values[keep])
    rank = int(keep.sum())
    mapping = np.zeros((model.h, 3, 4, 3, rank))
    for coordinate in range(3):
        mapping[:, coordinate, :, coordinate, :] = transform.reshape((model.h, 4, rank))
    mapping = mapping.reshape((12 * model.h, 3 * rank))
    rows = []
    def row(index, coordinate):
        external = np.zeros((model.h, 3, 4))
        external[:, coordinate, :] = model.b[index]
        return external.ravel() @ mapping
    for pin in pins:
        rows.extend(row(int(pin["index"]), coordinate) for coordinate in range(3))
    contact_count = 0
    if floor is not None:
        for index in np.flatnonzero(positions[:, 1] <= float(floor) + 1e-7):
            rows.append(row(index, 1))
            contact_count += 1
    reduced_gradient = mapping.T @ gradient
    if rows:
        matrix = np.vstack(rows)
        lengths = np.linalg.norm(matrix, axis=1)
        matrix = matrix[lengths > 1e-14] / lengths[lengths > 1e-14, None]
        _, singular, vt = np.linalg.svd(matrix, full_matrices=True)
        row_rank = int(np.sum(singular > max(1e-10, singular[0] * 1e-8)))
        projected = vt[row_rank:] @ reduced_gradient
    else:
        row_rank = 0
        projected = reduced_gradient
    return {"scalarMassRank": rank, "constraintRank": row_rank,
        "nearActiveFloorSamples": contact_count,
        "projectedGradientNorm": float(np.linalg.norm(projected))}


def audit_gpu_covariance(record):
    """Independent matrix contractions for retained GPU center/covariance data."""
    groups = {group["id"]: group["input"] for group in record["probe"]["groups"]}
    cells = record["probe"]["cells"]
    if len(cells) != 8 or set(groups) != {"analytic-96", "spot-128", "analytic-97-tail"}:
        raise ValueError("Incomplete fixed covariance cohort")
    output = []
    for cell in cells:
        source = groups[cell["group"]]
        f32 = lambda values, shape: array(values, shape).astype(np.float32).astype(np.float64)
        x = f32(source["positions"], (-1, 3))
        count, handles = len(x), int(source["modeCount"])
        weights = f32(source["weights"], (count, handles))
        gradients = f32(source["gradients"], (count, handles, 3))
        q = f32(cell["q"], (handles, 3, 4))
        u = np.einsum("hij,nj->nhi", q[:, :, :3], x) + q[None, :, :, 3]
        centers = x + np.einsum("nh,nhi->ni", weights, u)
        jacobians = np.eye(3) + np.einsum("nh,hij->nij", weights, q[:, :, :3]) + np.einsum("nhi,nhj->nij", u, gradients)
        packed = f32(source["covariance"], (count, 6))
        covariance = packed[:, [0, 1, 2, 1, 3, 4, 2, 4, 5]].reshape((count, 3, 3))
        transformed = jacobians @ covariance @ np.swapaxes(jacobians, 1, 2)
        result_packed = transformed.reshape((count, 9))[:, [0, 1, 2, 4, 5, 8]]
        checked = {"group": cell["group"], "state": cell["state"],
            "gpuCenters": require_close(cell["actual"]["centers"], centers, "GPU centers", 2e-6, 2e-6),
            "gpuCovariance": require_close(cell["actual"]["covariance"], result_packed, "GPU covariance", 1e-9, 4e-5),
            "retainedCPUCenters": require_close(cell["expected"]["centers"], centers, "retained CPU centers", 1e-12, 1e-12),
            "retainedCPUCovariance": require_close(cell["expected"]["covariance"], result_packed, "retained CPU covariance", 1e-12, 1e-12),
            "retainedCPUJacobian": require_close(cell["expected"]["jacobians"], jacobians, "retained full Jacobian", 1e-12, 1e-12)}
        output.append(checked)
    return output


def numeric_gradient(function, x, h=1e-6):
    x = array(x).copy()
    result = np.empty_like(x)
    for k in range(x.size):
        plus, minus = x.copy(), x.copy()
        plus.flat[k] += h
        minus.flat[k] -= h
        result.flat[k] = (function(plus) - function(minus)) / (2 * h)
    return result


def self_test():
    started = time.perf_counter()
    rng = np.random.default_rng(20260909)
    centers = rng.uniform(-1, 1, (18, 3))
    x = rng.uniform(-.8, .8, (72, 3))
    radii = np.full(len(centers), .9)
    volumes = np.full(len(x), 1 / len(x))
    origin, scale = np.zeros(3), 2.
    phi, dphi = rkpm(x, centers, radii, origin, scale)
    require_close(phi @ centers, x, "affine positions", atol=1e-10)
    require_close(np.einsum("nkj,ka->naj", dphi, centers),
                  np.broadcast_to(np.eye(3), (len(x), 3, 3)), "affine gradients", atol=1e-10)
    grad_error = 0.
    for j in range(3):
        offset = np.zeros(3)
        offset[j] = 1e-6
        fd = (rkpm(x + offset, centers, radii, origin, scale)[0] -
              rkpm(x - offset, centers, radii, origin, scale)[0]) / 2e-6
        grad_error = max(grad_error, require_close(dphi[:, :, j], fd, "RKPM FD", atol=3e-8)["maxAbs"])
    mats = assemble_basis(x, volumes, centers, radii, origin=origin, scale=scale)
    eig, coeff = generalized_eigh(mats["stiffness"], mats["mass"])
    h = 4
    coeff = np.column_stack((coeff[:, 1:h], np.ones(len(centers))))
    basis = dict(centers=centers, radii=radii, origin=origin, scale=scale,
                 coefficients=coeff, eigenvalues=eig[1:h], mass=mats["mass"], stiffness=mats["stiffness"])
    basis_audit = audit_basis({"input": dict(positions=x, volumes=volumes), "actual": basis})
    w, dw = evaluate_basis(x, basis)
    config = dict(restPositions=x, volumes=volumes, weights=w, gradients=dw,
                  density=2., young=1000., poisson=.3)
    model = Mechanics(config)
    q0 = np.zeros(12 * h)
    rest = model.state(q0)
    require_close(rest["energy"], 0., "rest energy", atol=1e-10)
    require_close(rest["gradient"], np.zeros_like(q0), "rest force", atol=1e-9)
    rotation = np.array([[np.cos(.6), -np.sin(.6), 0], [np.sin(.6), np.cos(.6), 0], [0, 0, 1.]])
    rigid = np.zeros((h, 3, 4))
    rigid[-1, :, :3] = rotation - np.eye(3)
    rigid[-1, :, 3] = [.2, -.4, .1]
    rigid_state = model.state(rigid)
    require_close(rigid_state["energy"], 0., "rigid energy", atol=2e-10)
    require_close(rigid_state["gradient"], np.zeros_like(q0), "rigid force", atol=2e-8)
    q = rng.normal(0, .004, q0.size)
    analytic = model.state(q)["gradient"]
    fd = numeric_gradient(lambda z: model.state(z)["energy"], q)
    force_error = require_close(analytic, fd, "elastic force FD", atol=2e-6, rtol=2e-6)
    drag = {"index": 7, "target": [.1, .3, -.2], "stiffness": 17.}
    velocity = rng.normal(0, .01, q0.size)
    obj = lambda z: model.objective(z, q0, velocity, .025, [0, -1, .2], drag)
    objective_error = require_close(obj(q)["gradient"], numeric_gradient(lambda z: obj(z)["value"], q),
                                    "full objective FD", atol=3e-6, rtol=2e-6)
    # Inverted and singular F exercise cofactor-based stress without inverse(F).
    for f in (np.diag([-.3, 1.1, .8]), np.diag([0., 1., 1.]), np.diag([1.1, .8, 1.2])):
        energy, stress, _ = neo_hookean(f, 1000, .3)
        numerical = numeric_gradient(lambda z: float(neo_hookean(z, 1000, .3)[0]), f)
        require_close(stress, numerical, "singular/inverted stress", atol=2e-6)
    # Independent torch autograd Jacobian of F->energy (CPU only) checks the
    # analytic stress and Hessian-vector product at a nontrivial deformation.
    import torch
    torch.set_num_threads(1)
    tf = torch.tensor(np.eye(3) + rng.normal(0, .06, (3, 3)), dtype=torch.float64, requires_grad=True)
    lam, mu = lame(1000, .3)
    gamma = 1 + mu / (lam + mu)
    def torch_energy(z):
        return .5 * (mu * ((z*z).sum() - 3) + (lam+mu) * ((torch.linalg.det(z)-gamma)**2 - (1-gamma)**2))
    tg = torch.autograd.grad(torch_energy(tf), tf, create_graph=True)[0]
    torch_error = require_close(tg.detach().numpy(), neo_hookean(tf.detach().numpy(), 1000, .3)[1],
                                "torch stress", atol=1e-10)
    direction = rng.normal(size=(3, 3))
    thv = torch.autograd.grad((tg * torch.tensor(direction)).sum(), tf)[0].detach().numpy()
    fn = tf.detach().numpy()
    nhv = (neo_hookean(fn + 1e-6*direction, 1000, .3)[1] - neo_hookean(fn - 1e-6*direction, 1000, .3)[1]) / 2e-6
    hv_error = require_close(thv, nhv, "torch Hessian-vector", atol=3e-6)
    # Wrong basis mode and swapped force components must not silently audit.
    rejected = 0
    wrong_basis = dict(basis, coefficients=coeff.copy())
    wrong_basis["coefficients"][0, 0] += .1
    try:
        audit_basis({"input": dict(positions=x, volumes=volumes), "actual": wrong_basis})
    except AssertionError:
        rejected += 1
    bad = dict(q=q.tolist(), energy=model.state(q)["energy"], gradient=(analytic + .1).tolist())
    try:
        audit_mechanics({"input": config, "states": [bad]})
    except AssertionError:
        rejected += 1
    if rejected != 2:
        raise AssertionError("Mutation checks failed")
    return {"passed": True, "RKPMGradientMaxAbs": grad_error, "basis": basis_audit,
        "elasticGradient": force_error, "objectiveGradient": objective_error,
        "torchStress": torch_error, "torchHessianVector": hv_error,
        "rejectedMutations": rejected, "seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    result = {"kind": "local-elasticity-independent-cpu-audit-v1"}
    if args.self_test:
        result["selfTest"] = self_test()
    if args.report:
        raw = args.report.read_bytes()
        report_name = args.report
        if args.report.suffix == ".gz":
            raw = gzip.decompress(raw)
            report_name = args.report.with_suffix("")
        record = json.loads(raw)
        result["inputSHA256"] = hashlib.sha256(raw).hexdigest()
        sidecar = report_name.with_suffix(".sha256")
        if sidecar.exists() and sidecar.read_text(encoding="utf-8").split()[0] != result["inputSHA256"]:
            raise ValueError("Raw report SHA256 sidecar mismatch")
        source_audit = []
        for relative, snapshot in record.get("sources", {}).items():
            if not isinstance(relative, str) or ":" in relative or relative.startswith(("/", "\\")) or ".." in relative.split("/"):
                raise ValueError("Source snapshot path must be workspace-relative")
            actual_sha = hashlib.sha256(snapshot["text"].encode("utf-8")).hexdigest()
            if actual_sha != snapshot["sha256"]:
                raise ValueError(f"Source snapshot SHA256 mismatch: {relative}")
            source_audit.append({"path": relative, "sha256": actual_sha})
        result["sourceSnapshots"] = source_audit
        if record.get("kind") == "local-elasticity-cpu-canary-v1":
            if record.get("status") != "completed":
                raise ValueError("Canary did not complete")
            for key in ("basisCases", "mechanicsCases"):
                if [item.get("asset") for item in record.get(key, [])] != ["plant", "spot"]:
                    raise ValueError("Expected both fixed assets in declared order")
            for item in record["mechanicsCases"]:
                if len(item.get("steps", [])) != 4 or len(item.get("states", [])) != 6:
                    raise ValueError("Incomplete fixed four-step/six-state canary")
        result["basisCases"] = [audit_basis(item) for item in record.get("basisCases", [])]
        result["mechanicsCases"] = [audit_mechanics(item) for item in record.get("mechanicsCases", [])]
        if record.get("probe", {}).get("cells") and record.get("probe", {}).get("groups"):
            result["gpuCovarianceCases"] = audit_gpu_covariance(record)
        if not result["basisCases"] and not result["mechanicsCases"] and not result.get("gpuCovarianceCases"):
            raise ValueError("No basisCases or mechanicsCases to audit")
    if not args.self_test and not args.report:
        parser.error("Provide --self-test and/or --report")
    result["passed"] = True
    encoded = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(encoded)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, AssertionError, np.linalg.LinAlgError) as error:
        print(json.dumps({"passed": False, "error": str(error)}), file=sys.stderr)
        sys.exit(1)
