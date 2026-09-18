"""Scenario ellipsoid, terminal predecessor closure, and online MPSC SOCP.

Run python analysis.py to regenerate the certificate and test public queries.
"""
import json
from pathlib import Path
import warnings

import cvxpy as cp
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent


def ellipsoid_residuals(data, P, tau):
    A, B, K = (np.asarray(data[key], float) for key in
               ('A', 'B', 'error_feedback_gain'))
    AK = A + B @ K
    worst = -np.inf
    for w in np.asarray(data['disturbance_scenarios']):
        cross = AK.T @ P @ w
        block = np.block([[AK.T @ P @ AK - tau * P, cross[:, None]],
                          [cross[None, :], np.array([[w @ P @ w + tau - 1]])]])
        worst = max(worst, np.linalg.eigvalsh(block)[-1])
    eig = np.linalg.eigvalsh(P)
    return {'lmi': float(worst), 'min_eigenvalue': float(eig[0]),
            'max_eigenvalue': float(eig[-1])}


def design_ellipsoid(data):
    A, B, K = (np.asarray(data[key], float) for key in
               ('A', 'B', 'error_feedback_gain'))
    AK, n = A + B @ K, len(A)
    settings = data['scenario_design']
    lo, hi = settings['tau_bounds']
    scale = 1000.0
    Q = cp.Variable((n, n), symmetric=True)
    tau = cp.Parameter(nonneg=True)
    constraints = [Q >> settings['p_floor'] / scale * np.eye(n),
                   np.eye(n) - Q * (scale / settings['p_ceiling']) >> 0]
    for w in np.asarray(data['disturbance_scenarios']):
        w = w[:, None]
        cross = np.sqrt(scale) * AK.T @ Q @ w
        block = cp.bmat([[AK.T @ Q @ AK - tau * Q, cross],
                         [cross.T, scale * w.T @ Q @ w + tau - 1]])
        constraints.append(block << -1e-9 * np.eye(n + 1))
    problem = cp.Problem(cp.Maximize(cp.log_det(Q)), constraints)
    results = {}

    def solve(t):
        t = float(t)
        if t in results:
            return results[t][0]
        tau.value = t
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', UserWarning)
                problem.solve(solver='CLARABEL', tol_gap_abs=1e-9,
                              tol_gap_rel=1e-9, tol_feas=1e-10, max_iter=150)
            if Q.value is None or problem.status not in ('optimal', 'optimal_inaccurate'):
                return 1e6
            P = scale * (Q.value + Q.value.T) / 2
            # Tiny enlargement makes active disturbance inequalities strict.
            P *= 1 - 2e-7
            residual = ellipsoid_residuals(data, P, t)
            if (not np.isfinite(P).all() or residual['lmi'] > 1e-7 or
                    residual['min_eigenvalue'] < settings['p_floor'] or
                    residual['max_eigenvalue'] > settings['p_ceiling']):
                return 1e6
            score = -np.linalg.slogdet(P)[1]
            results[t] = (score, P)
            return score
        except cp.error.SolverError:
            return 1e6

    grid = np.linspace(lo, hi, 89)
    values = np.array([solve(t) for t in grid])
    if not results:
        raise RuntimeError('No numerically certified scenario ellipsoid found')
    # Refine grid-local minima, retaining every endpoint and grid design.
    for i in range(len(grid)):
        if values[i] < 1e6 and (i == 0 or values[i] <= values[i - 1]) and (
                i == len(grid) - 1 or values[i] <= values[i + 1]):
            minimize_scalar(solve, bounds=(grid[max(0, i - 1)],
                                           grid[min(len(grid) - 1, i + 1)]),
                            method='bounded', options={'xatol': 2e-7})
    best = min(results, key=lambda t: results[t][0])
    return results[best][1], best


def support(H, P):
    return np.sqrt(np.maximum(0, np.einsum('ij,ji->i', H, np.linalg.solve(P, H.T))))


def terminal_closure(data, hx, hu):
    A, B = np.asarray(data['A']), np.asarray(data['B'])
    Hx = np.asarray(data['state_polytope']['H'])
    Hu = np.asarray(data['input_polytope']['H'])
    library = data['terminal_candidate_library']
    states, controls = np.asarray(library['states']), np.asarray(library['controls'])
    core = np.asarray(data['terminal_backup']['core_polytope']['vertices'])
    accepted = np.any(np.max(np.abs(states[:, None, :] - core[None, :, :]), axis=2)
                      <= 1e-9, axis=1)
    tol = 2e-7
    admissible = ((states @ Hx.T <= hx + tol).all(axis=1) &
                  (controls @ Hu.T <= hu + tol).all(axis=1))
    if not admissible[accepted].all():
        raise RuntimeError('Core records violate tightened constraints')
    if not all(np.any(np.max(np.abs(states[accepted] - v), axis=1) <= 1e-9) for v in core):
        raise RuntimeError('Core vertices missing from candidate library')
    successors = states @ A.T + controls @ B.T
    batches = [int(accepted.sum())]
    while True:
        hull = ConvexHull(states[accepted])
        H, h = hull.equations[:, :-1], -hull.equations[:, -1]
        add = ~accepted & admissible & (successors @ H.T <= h + tol).all(axis=1)
        if not add.any():
            break
        accepted |= add
        batches.append(int(add.sum()))
    if np.max(successors[accepted] @ H.T - h) > tol:
        raise RuntimeError('Terminal closure is not invariant')
    # Qhull triangulates coplanar facets; retain one copy of each plane.
    _, idx = np.unique(np.round(np.column_stack((H, h)), 10), axis=0, return_index=True)
    idx.sort()
    return H[idx], h[idx], accepted, batches


def build_certificate(data):
    P, tau = design_ellipsoid(data)
    Hx, hx = (np.asarray(data['state_polytope'][k], float) for k in ('H', 'h'))
    Hu, hu = (np.asarray(data['input_polytope'][k], float) for k in ('H', 'h'))
    sx = support(Hx, P)
    su = support(Hu @ np.asarray(data['error_feedback_gain']), P)
    Hf, hf, accepted, batches = terminal_closure(data, hx - sx, hu - su)
    print('Ellipsoid:', {'tau': tau, 'logdet': np.linalg.slogdet(P)[1],
                         **ellipsoid_residuals(data, P, tau)}, flush=True)
    print('Terminal batches:', batches, 'accepted:', int(accepted.sum()),
          'facets:', len(hf), flush=True)
    return {'schema_version': 2, 'error_ellipsoid': {'P': P.tolist(), 'tau': tau},
            'support_values': {'state': sx.tolist(), 'input': su.tolist()},
            'tightened_polytopes': {
                'state': {'H': Hx.tolist(), 'h': (hx - sx).tolist()},
                'input': {'H': Hu.tolist(), 'h': (hu - su).tolist()}},
            'terminal_polytope': {'H': Hf.tolist(), 'h': hf.tolist()}}


class SafetyFilter:
    def __init__(self, task_data, config=None):
        if isinstance(task_data, (str, Path)):
            task_data = json.loads(Path(task_data).read_text())
        self.data = task_data
        config = config or {}
        certificate = config.get('certificate')
        if certificate is None:
            certificate = json.loads((ROOT / 'certificate.json').read_text())
        self.A, self.B, self.K = (np.asarray(task_data[key], float) for key in
                                ('A', 'B', 'error_feedback_gain'))
        self.P = np.asarray(certificate['error_ellipsoid']['P'])
        self.Hx, self.hx = (np.asarray(certificate['tightened_polytopes']['state'][k])
                            for k in ('H', 'h'))
        self.Hu, self.hu = (np.asarray(certificate['tightened_polytopes']['input'][k])
                            for k in ('H', 'h'))
        self.Hf, self.hf = (np.asarray(certificate['terminal_polytope'][k]) for k in ('H', 'h'))
        n, m, N = self.A.shape[0], self.B.shape[1], int(task_data['horizon'])
        self.x, self.learning = cp.Parameter(n), cp.Parameter(m)
        self.z, self.v = cp.Variable((n, N + 1)), cp.Variable((m, N))
        self.action = self.v[:, 0] + self.K @ (self.x - self.z[:, 0])
        constraints = [self.z[:, 1:] == self.A @ self.z[:, :-1] + self.B @ self.v,
                       self.Hx @ self.z[:, :-1] <= self.hx[:, None],
                       self.Hu @ self.v <= self.hu[:, None],
                       cp.norm(np.linalg.cholesky(self.P).T @ (self.x - self.z[:, 0])) <= 1,
                       self.Hf @ self.z[:, -1] <= self.hf]
        reg = task_data['solver_regularization']
        objective = cp.quad_form(self.action - self.learning, np.asarray(task_data['action_metric']))
        objective += reg['rho_z'] * cp.sum_squares(self.z[:, 0] - self.x)
        objective += reg['rho_v'] * cp.sum_squares(self.v)
        self.problem = cp.Problem(cp.Minimize(objective), constraints)
        self.last_solution = None

    def residual(self, x, z, v):
        e = x - z[:, 0]
        return max(float(np.max(np.abs(z[:, 1:] - self.A @ z[:, :-1] - self.B @ v))),
                   float(np.max(self.Hx @ z[:, :-1] - self.hx[:, None])),
                   float(np.max(self.Hu @ v - self.hu[:, None])),
                   float(np.max(self.Hf @ z[:, -1] - self.hf)),
                   float(e @ self.P @ e - 1))

    def filter_control(self, x, u_learning, t=0, memory=None):
        x = np.asarray(x, float).reshape(self.A.shape[0])
        self.x.value = x
        self.learning.value = np.asarray(u_learning, float).reshape(self.B.shape[1])
        for solver in ('CLARABEL', 'CLARABEL', 'SCS'):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', UserWarning)
                    if solver == 'CLARABEL':
                        self.problem.solve(solver=solver, warm_start=True, max_iter=200,
                                           tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
                    else:
                        self.problem.solve(solver=solver, eps=1e-7, max_iters=100000)
                if self.z.value is not None and self.v.value is not None:
                    z, v = self.z.value.copy(), self.v.value.copy()
                    action = v[:, 0] + self.K @ (x - z[:, 0])
                    if np.isfinite(action).all() and self.residual(x, z, v) <= 2e-6:
                        self.last_solution = {'z': z, 'v': v, 'residual': self.residual(x, z, v)}
                        return action.tolist()
            except cp.error.SolverError:
                continue
        raise RuntimeError('No certified horizon plan exists for the supplied state')


def make_filter(task_data, config=None):
    return SafetyFilter(task_data, config)


if __name__ == '__main__':
    data = json.loads((ROOT / 'data/system.json').read_text())
    certificate = build_certificate(data)
    (ROOT / 'certificate.json').write_text(json.dumps(certificate, indent=2, allow_nan=False) + '\n')
    controller = make_filter(data)
    cases = json.loads((ROOT / 'data/public_cases.json').read_text())['cases']
    for case in cases:
        action = controller.filter_control(case['x'], case['u_learning'])
        print(case['id'], action, 'residual:', controller.last_solution['residual'], flush=True)
