"""
Numerical self-tests for the two hand-derived gradient implementations that
every result in this repository depends on:

  1. The VQC parameter-shift rule (`VQC.forward_and_param_shift_grad_batch`)
     -- checked against central finite differences.
  2. The manual MLP backpropagation (`MLP.forward` / `MLP.backward`)
     -- checked against central finite differences.

These are the *same* assertions that run automatically near the top of
`src/analysis.py` every time it executes; they are duplicated here as
standalone, fast (<5s), CI-friendly pytest functions so the numerical
correctness of the pipeline's two core gradient paths can be verified in
isolation, without running any part of the (multi-minute) training loops.

Implementation note: we import the class definitions directly out of
`src/analysis.py` by executing only the source code that appears *before*
the first experiment begins (everything up to and including the line
`print("[training] orchestration helpers ready")`). This reuses the exact
production implementation -- nothing here is a reimplementation that could
drift out of sync with `src/analysis.py`.
"""
import math
import pathlib

import numpy as np
import pytest

_SRC_PATH = pathlib.Path(__file__).resolve().parents[1] / "src" / "analysis.py"
_SAFE_MARKER = 'print("[training] orchestration helpers ready")'


def _load_pipeline_namespace() -> dict:
    """Execute src/analysis.py up to (and including) the safe marker line,
    returning the resulting namespace (contains VQC, MLP, and friends)
    without running any experiment / training loop."""
    import sys
    import types

    source = _SRC_PATH.read_text()
    idx = source.find(_SAFE_MARKER)
    assert idx != -1, (
        "Safe marker not found in src/analysis.py -- has the file been "
        "restructured? Update _SAFE_MARKER in this test file to match."
    )
    truncated_source = source[: idx + len(_SAFE_MARKER)]

    # Register a real (if synthetic) module in sys.modules first: the
    # dataclasses module (used by Config) resolves field type annotations
    # via sys.modules[cls.__module__], which fails with a bare exec()
    # namespace that isn't registered anywhere.
    module_name = "qacldn_test_import"
    module = types.ModuleType(module_name)
    module.__dict__["__name__"] = module_name
    sys.modules[module_name] = module
    exec(compile(truncated_source, str(_SRC_PATH), "exec"), module.__dict__)
    return module.__dict__


@pytest.fixture(scope="module")
def ns() -> dict:
    return _load_pipeline_namespace()


def test_parameter_shift_matches_finite_differences(ns):
    """The VQC's parameter-shift gradient must match a central finite
    difference to near machine precision for single-qubit-rotation
    generators (this is an *exact* analytic identity, not an
    approximation -- any large discrepancy indicates a bug)."""
    VQC = ns["VQC"]
    rng = np.random.default_rng(0)

    vqc = VQC(n_qubits=4, n_layers=2, entangled=True)
    theta = rng.normal(0, 1, vqc.n_params)
    s_batch = rng.uniform(0, 1, (5, 16))

    _, grad_shift = vqc.forward_and_param_shift_grad_batch(s_batch, theta)

    eps = 1e-5
    flat = theta.reshape(-1).copy()
    i_check = 3
    flat[i_check] += eps
    o_plus = vqc.forward_batch(s_batch, flat.reshape(vqc.L, -1))
    flat[i_check] -= 2 * eps
    o_minus = vqc.forward_batch(s_batch, flat.reshape(vqc.L, -1))
    grad_fd = (o_plus - o_minus) / (2 * eps)

    max_err = np.max(np.abs(grad_fd - grad_shift[i_check]))
    assert max_err < 1e-4, f"parameter-shift rule deviates from finite differences by {max_err:.2e}"


def test_mlp_backward_matches_finite_differences(ns):
    """The manually-derived MLP backward pass must match a central finite
    difference of the same scalar loss to near machine precision."""
    MLP = ns["MLP"]
    rng = np.random.default_rng(3)

    net = MLP(12, 10, 5, rng)
    x = rng.normal(size=12)
    out, cache = net.forward(x)
    dLdout = rng.normal(size=5)
    grads = net.backward(dLdout, cache)
    flat0 = net.get_flat().copy()

    def loss_fn(flat):
        net.set_flat(flat)
        out, _ = net.forward(x)
        return float(out @ dLdout)

    eps = 1e-5
    idxs = np.random.default_rng(4).choice(len(flat0), size=12, replace=False)
    fd_grad = np.zeros_like(flat0)
    for i in idxs:
        fp = flat0.copy(); fp[i] += eps
        fm = flat0.copy(); fm[i] -= eps
        fd_grad[i] = (loss_fn(fp) - loss_fn(fm)) / (2 * eps)
    net.set_flat(flat0)

    analytic_flat = np.concatenate(
        [grads["W1"].ravel(), grads["b1"].ravel(), grads["W2"].ravel(), grads["b2"].ravel()]
    )
    max_err = np.max(np.abs(fd_grad[idxs] - analytic_flat[idxs]))
    assert max_err < 1e-4, f"manual MLP backward pass deviates from finite differences by {max_err:.2e}"


def test_vqc_parameter_count_formula(ns):
    """P(n, L) = 2nL (VQC rotations) + |A|(n+1) (actor head) + (n+1)
    (critic head) must exactly match the agent's actual parameter count,
    for every (n, L) combination used in the qubit-sweep experiments."""
    QuantumActorCriticAgent = ns["QuantumActorCriticAgent"]
    n_actions = 16

    for n_qubits, n_layers in [(2, 2), (4, 2), (6, 2), (8, 2)]:
        agent = QuantumActorCriticAgent(n_qubits, n_layers, seed=0)
        expected = 2 * n_qubits * n_layers + n_actions * (n_qubits + 1) + (n_qubits + 1)
        assert agent.n_total_params == expected, (
            f"n={n_qubits}, L={n_layers}: expected P={expected}, got {agent.n_total_params}"
        )


def test_hover_command_is_dynamic_equilibrium(ns):
    """At the closed-form hover command u_hover = 1/sqrt(thrust_to_weight_max),
    total thrust must equal weight (m*g) to within numerical tolerance --
    i.e. Eq. u_hover in docs/METHODOLOGY.md really is a fixed point of the
    translational dynamics along the vertical axis."""
    CFG = ns["CFG"]
    build_allocation_matrix = ns["build_allocation_matrix"]

    M = build_allocation_matrix(CFG)
    u_hover = 1.0 / math.sqrt(CFG.thrust_to_weight_max)
    u2 = np.array([u_hover, u_hover, u_hover, u_hover]) ** 2
    total_thrust = (M @ u2)[0]
    weight = CFG.mass * CFG.g

    assert abs(total_thrust - weight) / weight < 1e-9, (
        f"hover command does not balance weight: thrust={total_thrust:.6f} N, weight={weight:.6f} N"
    )
