"""
Legendre-Gauss-Lobatto (LGL) Collocation and Pseudospectral Tools.
Generates collocation nodes, quadrature weights, differentiation matrices,
and interpolation routines for trajectory optimization.
"""

import numpy as np
from scipy.special import eval_legendre
from scipy.interpolate import interp1d


def lgl_nodes_weights_diff(N, tol=1e-15, max_iter=30):
    """
    Compute Legendre-Gauss-Lobatto (LGL) nodes, weights, and differentiation matrix.
    N: polynomial degree (number of nodes is N + 1).
    
    Nodes tau_k are roots of (1 - x^2) * P'_N(x) = 0 computed using Newton-Raphson
    iteration from Chebyshev-Lobatto initial guesses for full double-precision stability.
    
    Returns:
        tau: array of shape (N + 1,) in [-1, 1]
        w:   quadrature weights array of shape (N + 1,)
        D:   differentiation matrix of shape (N + 1, N + 1)
    """
    # Chebyshev-Lobatto initial guess
    tau = -np.cos(np.pi * np.arange(N + 1) / N)

    # Newton-Raphson for interior roots of P'_N(x) = 0
    for k in range(1, N):
        xk = tau[k]
        for _ in range(max_iter):
            LN = eval_legendre(N, xk)
            LN_m1 = eval_legendre(N - 1, xk)
            # P'_N(x)
            d_LN = N * (LN_m1 - xk * LN) / (1.0 - xk ** 2)
            # P''_N(x)
            d2_LN = (2.0 * xk * d_LN - N * (N + 1.0) * LN) / (1.0 - xk ** 2)
            dx = d_LN / d2_LN
            xk = xk - dx
            if abs(dx) < tol:
                break
        tau[k] = xk

    # Enforce exact boundary values and symmetry
    tau[0] = -1.0
    tau[N] = 1.0
    tau = 0.5 * (tau - tau[::-1])

    # Quadrature weights: w_i = 2 / (N * (N + 1) * [P_N(tau_i)]^2)
    LN = eval_legendre(N, tau)
    w = 2.0 / (N * (N + 1.0) * (LN ** 2))

    # Differentiation matrix D:
    # D_ij = LN[i] / (LN[j] * (tau[i] - tau[j]))  for i != j
    # D_00 = - N * (N + 1) / 4
    # D_NN = N * (N + 1) / 4
    # D_ii = 0 for 0 < i < N
    D = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            if i != j:
                D[i, j] = LN[i] / (LN[j] * (tau[i] - tau[j]))
            elif i == 0 and j == 0:
                D[i, j] = -0.25 * N * (N + 1.0)
            elif i == N and j == N:
                D[i, j] = 0.25 * N * (N + 1.0)
            else:
                D[i, j] = 0.0

    return tau, w, D


def interpolate_trajectory(t_nodes, values_nodes, t_eval):
    """
    Piecewise-linear interpolation, shared by optimized controls and exports.
    Every interpolated component stays inside the adjacent endpoint values.
    Continuous physical states are obtained by forward integration in continuous.py.
    
    t_nodes:      shape (N+1,)
    values_nodes: shape (N+1,) or (N+1, m)
    t_eval:       shape (K,)
    """
    return interp1d(t_nodes, values_nodes, axis=0, kind="linear",
                    bounds_error=True, assume_sorted=True)(t_eval)
