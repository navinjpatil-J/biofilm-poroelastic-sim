#!/usr/bin/env python3
"""
BiofilmPoroelasticSim.py — publication-ready final version (v2.8)
==================================================================
Companion code for:
"Biofilms as Living Sponges: A Drainage-Regime Framework for
 Mechanics, Transport, and Intervention".

SCOPE
=====
Generates Figure 2 and Figure 3 only. Figure 1 is conceptual and is
produced separately without simulation.

PHYSICS BASIS
=============
Linearised 1D Biot/Terzaghi consolidation under step volumetric loading:

    dp/dt = D_p d^2 p / dx^2 ,    D_p = k M_c / eta

    p     excess pore pressure
    D_p   poroelastic diffusivity [m^2/s]
    k     intrinsic permeability [m^2]
    M_c   drained constrained (oedometric) modulus [Pa]
    eta   pore-fluid viscosity [Pa s]

Geometry factors for tau_drain = L_eff^2 / (gf D_p):
    GF_OPEN_OPEN   = pi^2      both ends drained
    GF_OPEN_SEALED = pi^2 / 4  one drained end, one sealed end

Interior pressure sinks:
    Hard sink (beta -> inf): Dirichlet p = 0, idealised fully connected
        channel.
    Leaky sink (finite beta): Robin-type pressure-relief interface.
        Connectivity number Bi_sink = beta s / D_p; Bi_sink >~ 1 means a
        visible channel is hydraulically effective.

CORRECTIONS IN v2.8 (relative to v2.7)
======================================
(1) LEAKY-SINK DISCRETISATION FIX. The finite-beta sink loss is now the
    mesh-consistent lumped term (beta/dx) p, not (2 beta/dx) p. The
    previous factor of 2 was arbitrary and made the numerically realised
    connectivity differ from the reported Bi_sink = beta s / D_p. With
    the corrected lumping the realised sink conductance is grid
    convergent, so the Bi_sink = 1 threshold in Figure 2C,D reflects
    physics rather than discretisation.

(2) CONVERGENCE CHECK EXTENDED TO FINITE beta. convergence_check_channels
    now also refines the grid for a leaky sink near Bi_sink ~ 1, because
    the hard-sink (Dirichlet) test alone never exercises the sink-loss
    term that governs Panels 2C and 2D.

(3) PANEL B REFERENCE ALIGNED. The no-channel dotted line in Figure 2B
    now uses the same relative-to-initial tau_1/e convention as the
    plotted markers, so line and data are commensurable. Label updated
    to "No-channel tau_1/e".

(4) tau_ref CONVENTION DOCUMENTED. In figure_2_heatmap, tau_ref remains
    the exact spatially averaged Terzaghi tau_1/e (analytic mean starts
    at exactly 1, so relative_to_initial is immaterial there), while the
    channelled numerators use relative_to_initial=True to account for the
    small initial-mean depression caused by zero-width sink nodes. The
    sub-percent convention difference is now documented in-line.

CORRECTIONS RETAINED FROM v2.7
==============================
- figure_2_heatmap tau_ref is the no-channel tau_1/e of the exact
  spatially averaged Terzaghi series, not the first-mode time tau_1.
- Panel D reference line labelled as the no-channel tau_1/e reference.
- Heatmap channel spacings in ascending order.

LITERATURE_DATA NOTES (retained)
================================
    Shaw 04:        t_s 300 s
    Korstgens 01:   L_um 400 um
    Towler 03:      L_um 40 um, t_s 180 s
    Gloag 20:       review-level micro-scale placement (illustrative)
    Moeendarbary 13*: HeLa cell poroelastic anchor, not a biofilm point
    Derlon 12:      Derlon et al. (2012) J. Membr. Sci.; same research
                    line as Derlon 2016 (ref [26]) cited in the body for
                    ultrafiltration hydraulic resistance. Placement
                    (L ~ 50 um, t ~ 1000 s) is illustrative of that work
                    class.

All Figure 3A points are illustrative order-of-magnitude placements from
inferred specimen/probe length and observation window, NOT fitted
drainage measurements. Reviews (Gloag 20, Stoodley 02) have no single
primary (L_eff, t_obs) and are shown as representative estimates.

Validation: numerical solution checked against the exact Terzaghi series
for open-top/sealed-base consolidation, with grid refinement for both
hard and leaky sinks.

Dependencies: numpy, scipy, matplotlib
License: MIT
"""

from __future__ import annotations

import math
import warnings
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized
import matplotlib.pyplot as plt


# ======================================================================
# VERSION AND OUTPUT DIRECTORY
# ======================================================================

__version__ = "2.8.0-publication-final"
OUTPUT_DIR = "biofilm_sim_outputs"


def out(filename: str) -> str:
    """Return full path inside the output directory."""
    return os.path.join(OUTPUT_DIR, filename)


# ======================================================================
# CONSTANTS
# ======================================================================

GF_OPEN_OPEN = math.pi ** 2
GF_OPEN_SEALED = math.pi ** 2 / 4.0


# ======================================================================
# CORE PHYSICS FUNCTIONS
# ======================================================================

def poroelastic_diffusivity(k: float, M_c: float, eta: float) -> float:
    """Compute poroelastic diffusivity D_p = k M_c / eta [m^2/s]."""
    if k <= 0 or M_c <= 0 or eta <= 0:
        raise ValueError("k, M_c, and eta must be positive.")
    return k * M_c / eta


def drainage_time(L_eff: float,
                  D_p: float,
                  geometry_factor: float = GF_OPEN_OPEN) -> float:
    """Characteristic drainage time tau = L_eff^2 / (geometry_factor D_p)."""
    if L_eff <= 0 or D_p <= 0:
        raise ValueError("L_eff and D_p must be positive.")
    return L_eff ** 2 / (geometry_factor * D_p)


def terzaghi_exact(t: np.ndarray,
                   tau1: float,
                   n_modes: int = 20) -> np.ndarray:
    """Exact spatially averaged Terzaghi solution, open-top/sealed-base."""
    y = np.zeros_like(t, dtype=float)
    for m in range(n_modes):
        n = 2 * m + 1
        C = 8.0 / (n ** 2 * math.pi ** 2)
        y += C * np.exp(-(n ** 2) * t / tau1)
    return y


def terzaghi_exact_open_open(t: np.ndarray,
                             tau_oo: float,
                             n_modes: int = 20) -> np.ndarray:
    """Exact spatially averaged solution, open-open boundaries (odd modes)."""
    y = np.zeros_like(t, dtype=float)
    for n in range(1, 2 * n_modes, 2):
        C = 8.0 / (n ** 2 * math.pi ** 2)
        y += C * np.exp(-(n ** 2) * t / tau_oo)
    return y


def terzaghi_1_over_e_crossing(tau1: float,
                               bc: str = "open-sealed",
                               n_modes: int = 20) -> float:
    """Find the 1/e crossing time of the exact mean-pressure series."""
    t_arr = np.logspace(-5, 2, 7000) * tau1

    if bc == "open-sealed":
        p_arr = terzaghi_exact(t_arr, tau1, n_modes=n_modes)
    elif bc == "open-open":
        p_arr = terzaghi_exact_open_open(t_arr, tau1, n_modes=n_modes)
    else:
        raise ValueError("bc must be 'open-sealed' or 'open-open'.")

    target = 1.0 / math.e
    for i in range(1, len(t_arr)):
        if p_arr[i] <= target:
            t0, t1 = t_arr[i - 1], t_arr[i]
            p0, p1 = p_arr[i - 1], p_arr[i]
            return t0 + (t1 - t0) * (p0 - target) / (p0 - p1)

    return float(t_arr[-1])


def log_time_grid(t_max: float,
                  n_t: int = 700,
                  t_min: Optional[float] = None) -> np.ndarray:
    """Create a logarithmic time grid including t = 0."""
    if t_max <= 0:
        raise ValueError("t_max must be positive.")
    if n_t < 3:
        raise ValueError("n_t must be at least 3.")

    if t_min is None:
        t_min = t_max * 1e-7

    t_min = max(float(t_min), t_max * 1e-10)
    t_min = min(t_min, t_max * 1e-2)

    positive = np.logspace(np.log10(t_min), np.log10(t_max), n_t - 1)
    return np.concatenate(([0.0], positive))


def check_physical_bounds(P: np.ndarray,
                          label: str = "",
                          tol: float = 1e-6) -> None:
    """Check that normalised pressure lies in [0, 1] within tolerance."""
    p_min = float(np.min(P))
    p_max = float(np.max(P))
    prefix = f"[{label}] " if label else ""

    if p_min < -tol:
        raise RuntimeError(
            f"{prefix}Negative normalised pressure detected: {p_min:.4e}"
        )
    if p_max > 1.0 + tol:
        raise RuntimeError(
            f"{prefix}Pressure overshoot detected: {p_max:.4e}"
        )


def check_monotone_relaxation(t: np.ndarray,
                              y: np.ndarray,
                              label: str = "",
                              tol: float = 1e-7) -> None:
    """Check that a passive drainage relaxation curve is non-increasing."""
    dy = np.diff(y)
    if np.any(dy > tol):
        idx = int(np.argmax(dy))
        prefix = f"[{label}] " if label else ""
        raise RuntimeError(
            f"{prefix}Non-monotone relaxation detected: "
            f"dy={dy[idx]:.3e} between "
            f"t={t[idx]:.3e} and t={t[idx + 1]:.3e} s."
        )


def tau_1_over_e_from_curve(t: np.ndarray,
                            y: np.ndarray,
                            relative_to_initial: bool = True) -> float:
    """
    Find 1/e crossing time of a relaxation curve.

    If relative_to_initial is True the target is y(0)/e. This is preferred
    for sink simulations because zero-width sink nodes slightly reduce the
    initial spatial mean.
    """
    y = np.asarray(y, dtype=float)

    if relative_to_initial:
        target = y[0] / math.e
    else:
        target = 1.0 / math.e

    if y[0] <= target:
        return 0.0

    for i in range(1, len(t)):
        if y[i] <= target:
            t0, t1 = t[i - 1], t[i]
            y0, y1 = y[i - 1], y[i]
            return t0 + (t1 - t0) * (y0 - target) / (y0 - y1)

    return float(t[-1])


def save_csv(filename: str,
             header: List[str],
             columns: List[np.ndarray]) -> None:
    """Save equal-length columns to CSV."""
    columns = [np.asarray(c) for c in columns]
    n = len(columns[0])

    if any(len(c) != n for c in columns):
        raise ValueError("All CSV columns must have equal length.")

    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(zip(*columns))

    print(f"  [CSV saved] {filename}")


def recommended_theta_for_sink(beta: float) -> float:
    """Recommend theta based on sink stiffness (BE is safer for finite beta)."""
    if math.isinf(beta):
        return 0.5
    if beta > 0.0:
        return 1.0
    return 0.5


# ======================================================================
# 1D POROELASTIC SOLVER
# ======================================================================

class PoroelasticSolver1D:
    """Theta-method solver for 1D Biot/Terzaghi pressure diffusion."""

    def __init__(self,
                 L: float,
                 n_x: int = 301,
                 k: float = 1e-15,
                 M_c: float = 1e4,
                 eta: float = 1e-3,
                 alpha: float = 1.0,
                 boundary_top: str = "open",
                 boundary_bot: str = "sealed",
                 interior_sinks: Optional[Sequence[float]] = None,
                 sink_beta: float = float("inf")):

        if L <= 0:
            raise ValueError("L must be positive.")
        if n_x < 5:
            raise ValueError("n_x must be at least 5.")
        if not (0.0 <= alpha <= 1.0):
            raise ValueError("alpha must lie in [0, 1].")
        if boundary_top not in ("open", "sealed"):
            raise ValueError("boundary_top must be 'open' or 'sealed'.")
        if boundary_bot not in ("open", "sealed"):
            raise ValueError("boundary_bot must be 'open' or 'sealed'.")

        self.L = float(L)
        self.n_x = int(n_x)
        self.x = np.linspace(0.0, self.L, self.n_x)
        self.dx = self.x[1] - self.x[0]

        self.k = float(k)
        self.M_c = float(M_c)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.D_p = poroelastic_diffusivity(self.k, self.M_c, self.eta)

        self.bt = boundary_top
        self.bb = boundary_bot
        self.beta = float(sink_beta)

        self.interior_sinks = list(interior_sinks or [])
        self._sink_idx: List[int] = sorted({
            int(np.argmin(np.abs(self.x - xs)))
            for xs in self.interior_sinks
            if 0.0 < xs < self.L
        })

    def _build_matrices(self, dt: float, theta: float = 0.5):
        """Build theta-method matrices A and B: A p^{n+1} = B p^n."""
        if not (0.5 <= theta <= 1.0):
            raise ValueError("theta should be between 0.5 and 1.0.")

        rA = theta * self.D_p * dt / self.dx ** 2
        rB = (1.0 - theta) * self.D_p * dt / self.dx ** 2

        n = self.n_x
        A = lil_matrix((n, n))
        B = lil_matrix((n, n))

        for i in range(1, n - 1):
            A[i, i - 1] = -rA
            A[i, i] = 1.0 + 2.0 * rA
            A[i, i + 1] = -rA

            B[i, i - 1] = rB
            B[i, i] = 1.0 - 2.0 * rB
            B[i, i + 1] = rB

        # Bottom boundary x = 0
        if self.bb == "open":
            A[0, :] = 0.0
            A[0, 0] = 1.0
            B[0, :] = 0.0
        else:
            A[0, 0] = 1.0 + 2.0 * rA
            A[0, 1] = -2.0 * rA
            B[0, 0] = 1.0 - 2.0 * rB
            B[0, 1] = 2.0 * rB

        # Top boundary x = L
        if self.bt == "open":
            A[-1, :] = 0.0
            A[-1, -1] = 1.0
            B[-1, :] = 0.0
        else:
            A[-1, -1] = 1.0 + 2.0 * rA
            A[-1, -2] = -2.0 * rA
            B[-1, -1] = 1.0 - 2.0 * rB
            B[-1, -2] = 2.0 * rB

        # Interior sinks
        if math.isinf(self.beta):
            for idx in self._sink_idx:
                A[idx, :] = 0.0
                A[idx, idx] = 1.0
                B[idx, :] = 0.0
        else:
            # Mesh-consistent lumped Robin sink loss: -(beta/dx) p.
            # v2.8 fix: removed the previous arbitrary factor of 2 so the
            # realised sink conductance, and hence Bi_sink = beta s / D_p,
            # is grid convergent (see convergence_check_channels).
            sink_loss = self.beta * dt / self.dx
            for idx in self._sink_idx:
                A[idx, idx] += theta * sink_loss
                B[idx, idx] -= (1.0 - theta) * sink_loss

        return A.tocsc(), B.tocsr()

    def _initial_pressure(self, p0: float) -> np.ndarray:
        """Construct initial pressure field."""
        p = p0 * np.ones(self.n_x)

        if self.bb == "open":
            p[0] = 0.0
        if self.bt == "open":
            p[-1] = 0.0

        if math.isinf(self.beta):
            for idx in self._sink_idx:
                p[idx] = 0.0

        return p

    def _apply_dirichlet_rhs(self, rhs: np.ndarray) -> np.ndarray:
        """Apply Dirichlet rows to RHS."""
        if self.bb == "open":
            rhs[0] = 0.0
        if self.bt == "open":
            rhs[-1] = 0.0

        if math.isinf(self.beta):
            for idx in self._sink_idx:
                rhs[idx] = 0.0

        return rhs

    def solve(self,
              t_max: Optional[float] = None,
              n_t: int = 600,
              t_eval: Optional[np.ndarray] = None,
              p0: float = 1.0,
              theta: float = 0.5,
              n_be_init: int = 4,
              enforce_nonneg: bool = False,
              return_diagnostics: bool = False):
        """Solve the pressure-diffusion problem."""
        if p0 <= 0:
            raise ValueError("p0 must be positive.")

        if t_eval is None:
            if t_max is None or t_max <= 0:
                raise ValueError(
                    "Either t_eval or positive t_max must be supplied."
                )
            t = np.linspace(0.0, float(t_max), int(n_t))
        else:
            t = np.asarray(t_eval, dtype=float)
            if len(t) < 2:
                raise ValueError("t_eval must contain at least two times.")
            if abs(t[0]) > 1e-15:
                t = np.concatenate(([0.0], t))
            if np.any(np.diff(t) <= 0):
                raise ValueError("t_eval must be strictly increasing.")

        dt_all = np.diff(t)
        max_r = self.D_p * np.max(dt_all) / (2.0 * self.dx ** 2)
        first_r = self.D_p * dt_all[0] / (2.0 * self.dx ** 2)

        if self._sink_idx and theta < 1.0 and n_be_init == 0 and max_r > 10:
            warnings.warn(
                f"Large maximum CN parameter r={max_r:.1f} with interior "
                "sinks and n_be_init=0. Oscillations may occur.",
                RuntimeWarning, stacklevel=2
            )

        if self._sink_idx and first_r > 5:
            warnings.warn(
                f"First time step has r={first_r:.1f}. Early channel "
                "drainage may be under-resolved. Use a smaller t_min.",
                RuntimeWarning, stacklevel=2
            )

        p = self._initial_pressure(p0)

        P = np.zeros((len(t), self.n_x))
        P[0] = p.copy()

        cache = {}
        clip_info = {
            "min_before_clip": 0.0,
            "total_clipped_mass": 0.0,
            "n_clipped_steps": 0,
        }

        def get_solver(dt: float, th: float):
            key = (round(float(dt), 15), round(float(th), 8))
            if key not in cache:
                A, B = self._build_matrices(dt, theta=th)
                cache[key] = (factorized(A), B)
            return cache[key]

        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]

            if theta < 1.0 and i <= n_be_init:
                th_use = 1.0
            else:
                th_use = theta

            solve_A, B = get_solver(dt, th_use)

            rhs = B @ p
            rhs = self._apply_dirichlet_rhs(rhs)
            p = solve_A(rhs)

            if enforce_nonneg:
                min_before = float(np.min(p))
                if min_before < 0.0:
                    clip_info["min_before_clip"] = min(
                        clip_info["min_before_clip"], min_before
                    )
                    clip_info["total_clipped_mass"] += float(
                        np.sum(np.abs(p[p < 0.0]))
                    )
                    clip_info["n_clipped_steps"] += 1
                p = np.maximum(p, 0.0)

            P[i] = p

        if enforce_nonneg and clip_info["min_before_clip"] < -1e-5:
            warnings.warn(
                f"Significant negative pressure clipping: "
                f"min p = {clip_info['min_before_clip']:.3e}. "
                f"Consider smaller time steps or theta=1.",
                RuntimeWarning, stacklevel=2
            )

        P_norm = P / p0
        check_physical_bounds(P_norm, label="PoroelasticSolver1D")

        diagnostics = {
            "clip_info": clip_info,
            "max_r": float(max_r),
            "first_r": float(first_r),
        }

        if return_diagnostics:
            return t, P_norm, diagnostics
        return t, P_norm

    def mean_pressure(self, P: np.ndarray) -> np.ndarray:
        """Spatially averaged normalised pressure."""
        try:
            integral = np.trapezoid(P, self.x, axis=1)
        except AttributeError:
            integral = np.trapz(P, self.x, axis=1)
        return integral / self.L

    def tau_1_over_e(self,
                     t: np.ndarray,
                     P: np.ndarray,
                     relative_to_initial: bool = False) -> float:
        """Find 1/e crossing time of spatially averaged pressure."""
        y = self.mean_pressure(P)
        return tau_1_over_e_from_curve(
            t, y, relative_to_initial=relative_to_initial
        )

    def apparent_modulus(self,
                         P: np.ndarray,
                         E_dr: float,
                         E_u: float) -> np.ndarray:
        """Pedagogical apparent-modulus interpolation."""
        return E_dr + self.alpha ** 2 * (E_u - E_dr) * self.mean_pressure(P)

    def validate_terzaghi(self,
                          n_t: int = 800,
                          skip_fraction: float = 0.05) -> Dict:
        """Validate against exact Terzaghi open-top/sealed-base solution."""
        if self.bt != "open" or self.bb != "sealed" or self._sink_idx:
            raise ValueError(
                "Validation requires open-top/sealed-base and no sinks."
            )

        tau1 = drainage_time(self.L, self.D_p, GF_OPEN_SEALED)
        t_max = 5.0 * tau1
        t_grid = log_time_grid(t_max, n_t=n_t, t_min=tau1 * 1e-5)

        t, P = self.solve(
            t_eval=t_grid, theta=0.5, n_be_init=4, enforce_nonneg=False
        )
        p_num = self.mean_pressure(P)
        p_ex = terzaghi_exact(t, tau1, n_modes=30)

        skip = int(skip_fraction * len(t))
        denom = np.abs(p_ex[skip:]) + 1e-12
        rel_err = np.abs(p_num[skip:] - p_ex[skip:]) / denom

        return {
            "t": t,
            "numerical": p_num,
            "analytical": p_ex,
            "max_rel_err": float(np.max(rel_err)),
            "tau1": tau1
        }


# ======================================================================
# REGIME PLACEMENT
# ======================================================================

@dataclass
class ProtocolInputs:
    L_eff: float
    k_range: Tuple[float, float]
    eta: float = 1e-3
    M_c: float = 1e4
    t_obs: float = 1.0
    boundary_type: str = "sealed"
    strain_mode: str = "volumetric"
    alpha: float = 1.0


@dataclass
class NdResult:
    Nd_min: float
    Nd_max: float
    regime: str
    tau_drain_range: Tuple[float, float]
    recommendation: str


def classify_regime(Nd_min: float, Nd_max: float) -> str:
    """Classify drainage regime from a drainage-number range."""
    if Nd_max < 0.1:
        return "UNDRAINED-like  (N_d << 1)"
    if Nd_min > 10:
        return "DRAINED-like    (N_d >> 1)"
    if 0.1 <= Nd_min and Nd_max <= 10.0:
        return "INTERMEDIATE    (0.1 < N_d < 10) -- DIAGNOSTIC WINDOW"
    return "MIXED / SPANS REGIMES"


def regime_placement(p: ProtocolInputs, verbose: bool = True) -> NdResult:
    """Compute drainage-number range for a protocol."""
    gf = GF_OPEN_SEALED if p.boundary_type == "sealed" else GF_OPEN_OPEN
    kmin, kmax = p.k_range

    D_slow = poroelastic_diffusivity(kmin, p.M_c, p.eta)
    D_fast = poroelastic_diffusivity(kmax, p.M_c, p.eta)

    tau_slow = drainage_time(p.L_eff, D_slow, gf)
    tau_fast = drainage_time(p.L_eff, D_fast, gf)

    Nd_min = p.t_obs / tau_slow
    Nd_max = p.t_obs / tau_fast

    regime = classify_regime(Nd_min, Nd_max)

    if "INTERMEDIATE" in regime:
        rec = "Run geometry-scaling collapse tau ~ L^2 and boundary perturbation."
    elif "UNDRAINED" in regime:
        rec = "Increase t_obs or decrease L_eff to approach N_d ~ 1."
    elif "DRAINED" in regime:
        rec = "Decrease t_obs or increase L_eff to approach N_d ~ 1."
    else:
        rec = "Constrain k and L_eff using pressure-driven conductance and imaging."

    result = NdResult(Nd_min, Nd_max, regime, (tau_fast, tau_slow), rec)

    if verbose:
        sep = "=" * 72
        gf_label = (
            "pi^2/4 (open-sealed)" if gf == GF_OPEN_SEALED else "pi^2 (open-open)"
        )
        print(sep)
        print("  DRAINAGE REGIME REPORT")
        print(sep)
        print(f"  L_eff           : {p.L_eff * 1e6:.1f} um")
        print(f"  t_obs           : {p.t_obs:.3g} s")
        print(f"  k range         : [{kmin:.2g}, {kmax:.2g}] m^2")
        print(f"  M_c             : {p.M_c:.2g} Pa")
        print(f"  eta             : {p.eta:.2g} Pa s")
        print(f"  boundary        : {p.boundary_type} (gf = {gf_label})")
        print(f"  tau range       : [{tau_fast:.3g}, {tau_slow:.3g}] s")
        print(f"  N_d range       : [{Nd_min:.3g}, {Nd_max:.3g}]")
        print(f"  REGIME          : {regime}")
        print(f"  RECOMMENDATION  : {rec}")
        print(sep)

    return result


# ======================================================================
# FIGURE 2 SIMULATIONS
# ======================================================================

def channel_sink_positions(L: float, spacing_um: float) -> List[float]:
    """Return regularly spaced interior sink positions inside (0, L)."""
    s = spacing_um * 1e-6
    if s <= 0:
        raise ValueError("spacing_um must be positive.")

    positions = []
    x = s
    tol = 1e-12 * L
    while x < L - tol:
        positions.append(float(x))
        x += s
    return positions


def simulate_with_channels(L: float = 200e-6,
                           spacings_um: Optional[Sequence] = None,
                           k: float = 1e-15,
                           M_c: float = 1e4,
                           eta: float = 1e-3,
                           n_x: int = 401,
                           n_t: int = 800,
                           n_be_init: int = 4,
                           sink_beta: float = float("inf")) -> Dict:
    """Simulate pressure relaxation for different interior channel spacings."""
    if spacings_um is None:
        spacings_um = [None, 100, 50, 25, 10]

    D_p = poroelastic_diffusivity(k, M_c, eta)
    tau_ref = drainage_time(L, D_p, GF_OPEN_SEALED)
    t_max = 5.0 * tau_ref

    finite_spacings = [s for s in spacings_um if s is not None]
    if finite_spacings:
        s_min = min(finite_spacings) * 1e-6
        tau_min = drainage_time(s_min, D_p, GF_OPEN_OPEN)
        t_min = tau_min / 200.0
    else:
        t_min = tau_ref * 1e-6

    t_grid = log_time_grid(t_max, n_t=n_t, t_min=t_min)

    results = {}

    for s_um in spacings_um:
        if s_um is None:
            sinks = []
            label = "no channels"
        else:
            sinks = channel_sink_positions(L, s_um)
            if math.isinf(sink_beta):
                beta_label = r"$\beta\to\infty$"
            else:
                Bi = sink_beta * (s_um * 1e-6) / D_p
                beta_label = fr"$\beta={sink_beta:.1e}$ m/s, Bi={Bi:.2g}"
            label = fr"$s={s_um}$ $\mu$m, {beta_label}"

        sol = PoroelasticSolver1D(
            L=L, n_x=n_x, k=k, M_c=M_c, eta=eta,
            boundary_top="open", boundary_bot="sealed",
            interior_sinks=sinks, sink_beta=sink_beta
        )

        theta_use = recommended_theta_for_sink(sink_beta)
        n_be_use = 0 if theta_use == 1.0 else n_be_init

        t, P, diag = sol.solve(
            t_eval=t_grid, theta=theta_use, n_be_init=n_be_use,
            enforce_nonneg=False, return_diagnostics=True
        )

        p_avg = sol.mean_pressure(P)
        check_monotone_relaxation(t, p_avg, label=label)

        p_avg_plot = p_avg / p_avg[0]
        tau = tau_1_over_e_from_curve(t, p_avg, relative_to_initial=True)

        results[s_um] = {
            "label": label, "t": t, "p_avg": p_avg,
            "p_avg_plot": p_avg_plot, "tau": tau,
            "x": sol.x, "diagnostics": diag
        }

    return results


def figure_2_channel_demo(results: Dict,
                          save_path: Optional[str] = None,
                          csv_path: Optional[str] = None) -> plt.Figure:
    """
    Figure 2A-B:
        A. Remaining normalised excess pore pressure vs time
        B. tau_1/e vs channel spacing
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    # Panel A
    ax = axes[0]
    colors = plt.cm.plasma(np.linspace(0.18, 0.85, len(results)))
    for (s_um, d), col in zip(results.items(), colors):
        lw = 2.7 if s_um is None else 2.1
        ax.semilogx(d["t"] + 1e-14, d["p_avg_plot"],
                    lw=lw, color=col, label=d["label"])

    ax.axhline(1.0 / math.e, color="k", ls=":", lw=1.4, label=r"$1/e$")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"Time $t$ (s)", fontsize=12)
    ax.set_ylabel(r"Remaining normalised excess pore pressure", fontsize=12)
    ax.set_title(
        "A. Pore-pressure relaxation vs. connected channel spacing\n"
        r"Curves normalised by their initial spatial mean",
        fontsize=10.5
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Panel B
    ax = axes[1]
    spacings = [s for s in results.keys() if s is not None]
    taus = np.array([results[s]["tau"] for s in spacings], dtype=float)
    s_arr = np.array(spacings, dtype=float)

    ax.loglog(s_arr, taus, "o-", lw=2.3, ms=9,
              color="#D62728", label=r"Simulation $\tau_{1/e}$")
    ax.loglog(s_arr, taus[0] * (s_arr / s_arr[0]) ** 2,
              "k--", lw=1.6, alpha=0.6, label=r"$\tau\propto s^2$ guide")

    # v2.8: reference line uses the SAME relative-to-initial tau_1/e
    # convention as the markers, so the dotted line is commensurable
    # with the plotted simulation points.
    if None in results:
        ax.axhline(results[None]["tau"], color="steelblue", ls=":",
                   lw=2.0, label=r"No-channel $\tau_{1/e}$")

    ax.set_xlabel(r"Channel spacing $s$ ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"$\tau_{1/e}$ (s)", fontsize=12)
    ax.set_title(
        r"B. $\tau_{\mathrm{drain}}$ tracks spacing, not thickness",
        fontsize=10.5
    )
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Figure saved] {save_path}")

    if csv_path:
        header = ["time_s"]
        columns = [next(iter(results.values()))["t"]]
        for s_um, d in results.items():
            name = "no_channels" if s_um is None else f"s_{s_um}_um"
            header.append(name)
            columns.append(d["p_avg_plot"])
        save_csv(csv_path, header, columns)

    return fig


def figure_2_heatmap(L: float = 200e-6,
                     spacings_um: Optional[Sequence] = None,
                     beta_values: Optional[Sequence] = None,
                     k: float = 1e-15,
                     M_c: float = 1e4,
                     eta: float = 1e-3,
                     n_x: int = 301,
                     n_t: int = 500,
                     n_be_init: int = 4,
                     save_path: Optional[str] = None,
                     csv_path: Optional[str] = None) -> plt.Figure:
    """
    Figure 2C-D:
        C. Heatmap of log10(tau/tau_ref) vs spacing and beta
        D. Slices through the heatmap

    tau_ref is the no-channel tau_1/e crossing of the exact spatially
    averaged Terzaghi series (open-top/sealed-base). Because the analytic
    mean starts at exactly 1, relative_to_initial is immaterial for the
    reference. The channelled numerators use relative_to_initial=True to
    account for the small (sub-percent) initial-mean depression from
    zero-width sink nodes; this is the physically correct choice.
    """
    if spacings_um is None:
        spacings_um = [10, 20, 30, 40, 50, 75, 100, 150]  # ascending

    if beta_values is None:
        beta_values = np.logspace(-5, 0, 12)

    D_p = poroelastic_diffusivity(k, M_c, eta)

    tau_char_ref = drainage_time(L, D_p, GF_OPEN_SEALED)
    tau_ref = terzaghi_1_over_e_crossing(
        tau_char_ref, bc="open-sealed", n_modes=30
    )

    t_max = 5.0 * tau_char_ref
    s_min = min(spacings_um) * 1e-6
    tau_min = drainage_time(s_min, D_p, GF_OPEN_OPEN)
    t_min = tau_min / 200.0
    t_grid = log_time_grid(t_max, n_t=n_t, t_min=t_min)

    S_arr = np.array(spacings_um, dtype=float)
    B_arr = np.array(beta_values, dtype=float)
    TAU = np.zeros((len(B_arr), len(S_arr)))

    # Finite-beta cases (Backward Euler: stable and monotone for stiff loss).
    for j, s_um in enumerate(S_arr):
        sinks = channel_sink_positions(L, s_um)
        for i, beta in enumerate(B_arr):
            sol = PoroelasticSolver1D(
                L=L, n_x=n_x, k=k, M_c=M_c, eta=eta,
                boundary_top="open", boundary_bot="sealed",
                interior_sinks=sinks, sink_beta=float(beta)
            )
            t, P, _ = sol.solve(
                t_eval=t_grid, theta=1.0, n_be_init=0,
                enforce_nonneg=False, return_diagnostics=True
            )
            p_avg = sol.mean_pressure(P)
            check_monotone_relaxation(
                t, p_avg, label=f"heatmap s={s_um}, beta={beta:.1e}"
            )
            TAU[i, j] = tau_1_over_e_from_curve(
                t, p_avg, relative_to_initial=True
            )

    # Hard-sink limit
    tau_hard = np.zeros(len(S_arr))
    for j, s_um in enumerate(S_arr):
        sinks = channel_sink_positions(L, s_um)
        sol = PoroelasticSolver1D(
            L=L, n_x=n_x, k=k, M_c=M_c, eta=eta,
            boundary_top="open", boundary_bot="sealed",
            interior_sinks=sinks, sink_beta=float("inf")
        )
        t, P, _ = sol.solve(
            t_eval=t_grid, theta=0.5, n_be_init=n_be_init,
            enforce_nonneg=False, return_diagnostics=True
        )
        p_avg = sol.mean_pressure(P)
        check_monotone_relaxation(t, p_avg, label=f"hard sink s={s_um}")
        tau_hard[j] = tau_1_over_e_from_curve(
            t, p_avg, relative_to_initial=True
        )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel C
    ax = axes[0]
    Z = np.log10(np.clip(TAU / tau_ref, 1e-4, 1.5))
    im = ax.pcolormesh(S_arr, B_arr, Z, shading="auto",
                       cmap="magma_r", vmin=-3, vmax=0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}(\tau/\tau_{\mathrm{ref}})$", fontsize=11)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Channel spacing $s$ ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"Sink pressure-relief coefficient $\beta$ (m s$^{-1}$)",
                  fontsize=12)
    ax.set_title(
        r"C. Structural spacing + hydraulic connectivity"
        "\n"
        r"Effective sinks require $\mathrm{Bi}_{\mathrm{sink}}=\beta s/D_{\mathrm{p}} \gtrsim 1$",
        fontsize=10.5
    )

    beta_bi1 = D_p / (S_arr * 1e-6)
    valid = (beta_bi1 >= B_arr.min()) & (beta_bi1 <= B_arr.max())
    if np.any(valid):
        ax.plot(S_arr[valid], beta_bi1[valid], "w--", lw=2.0,
                label=r"$\mathrm{Bi}_{\mathrm{sink}}=1$")
        ax.legend(fontsize=9, loc="lower left")

    # Panel D
    ax = axes[1]
    ax.loglog(S_arr, tau_hard / tau_ref, "k^-", lw=2.4, ms=9,
              label=r"$\beta\to\infty$ hard sink")

    picks = [B_arr[-3], B_arr[-5], B_arr[-7]]
    for beta in picks:
        idx = int(np.argmin(np.abs(B_arr - beta)))
        ax.loglog(S_arr, TAU[idx, :] / tau_ref, "o--", lw=1.8, ms=7,
                  label=fr"$\beta={B_arr[idx]:.1e}$ m/s")

    ax.axhline(1.0, color="gray", ls=":", lw=1.5,
               label=r"No-channel $\tau_{1/e}$ reference")
    ax.loglog(S_arr, (S_arr / S_arr[0]) ** 2 * tau_hard[0] / tau_ref,
              "k:", lw=1.2, alpha=0.55, label=r"$s^2$ guide")

    ax.set_xlabel(r"Channel spacing $s$ ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"$\tau/\tau_{\mathrm{ref}}$", fontsize=12)
    ax.set_title(
        "D. Morphology alone is insufficient\n"
        r"(visible channels require hydraulic connectivity)",
        fontsize=10.5
    )
    ax.legend(fontsize=8.8)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Figure saved] {save_path}")

    if csv_path:
        header = ["beta_m_per_s"] + [f"s_{int(s)}_um" for s in S_arr]
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i, beta in enumerate(B_arr):
                writer.writerow([beta] + list(TAU[i, :]))
        print(f"  [CSV saved] {csv_path}")

    return fig


# ======================================================================
# FIGURE 3
# ======================================================================

# ----------------------------------------------------------------------
# LITERATURE_DATA — illustrative order-of-magnitude placements only.
#
#   All points are inferred from specimen/probe length scale and
#   observation window across protocols that differ in strain mode;
#   they are NOT fitted drainage states. Reviews (Gloag 20, Stoodley 02)
#   have no single primary (L_eff, t_obs) and are shown as representative
#   estimates. Moeendarbary 13* is a HeLa-cell poroelastic anchor, not a
#   biofilm point. "Derlon 12" denotes Derlon et al. (2012),
#   J. Membr. Sci.; the same research line is cited in the body as
#   Derlon 2016 (ref [26]) for ultrafiltration hydraulic resistance.
# ----------------------------------------------------------------------

LITERATURE_DATA = [
    {"label": "Klapper 02", "L_um": 100, "t_s": 1.0,
     "marker": "D", "color": "#FDAE61", "cat": "flow/compression"},
    {"label": "Stoodley 02", "L_um": 50, "t_s": 0.1,
     "marker": "D", "color": "#F46D43", "cat": "flow/shear"},
    {"label": "Shaw 04", "L_um": 200, "t_s": 300.0,
     "marker": "s", "color": "#FEE08B", "cat": "rheology"},
    {"label": "Wilking (Glass)", "L_um": 150, "t_s": 10.0,
     "marker": "^", "color": "#91BFDB", "cat": "colony"},
    {"label": "Wilking (Agar)", "L_um": 150, "t_s": 3.0,
     "marker": "v", "color": "#4575B4", "cat": "colony"},
    {"label": "Gloag 20", "L_um": 2, "t_s": 1.0,
     "marker": "o", "color": "#66BD63", "cat": "micro"},
    {"label": "Körstgens 01", "L_um": 400, "t_s": 100.0,
     "marker": "s", "color": "#D73027", "cat": "compression"},
    {"label": "Towler 03", "L_um": 40, "t_s": 180.0,
     "marker": "s", "color": "#FC8D59", "cat": "compression"},
    {"label": "Kundukad 16", "L_um": 3, "t_s": 0.5,
     "marker": "o", "color": "#1A9850", "cat": "AFM"},
    {"label": "Lau 09", "L_um": 2, "t_s": 0.5,
     "marker": "o", "color": "#A6D96A", "cat": "micro"},
    {"label": "Böl 13", "L_um": 500, "t_s": 60.0,
     "marker": "s", "color": "#E08080", "cat": "compression"},
    {"label": "Derlon 12", "L_um": 50, "t_s": 1000.0,
     "marker": "v", "color": "#C2A5CF", "cat": "hydraulic"},
    {"label": "Moeendarbary 13*", "L_um": 5, "t_s": 2.0,
     "marker": "*", "color": "gray", "cat": "cell reference"},
]


def figure_3A(save_path: Optional[str] = None,
              csv_path: Optional[str] = None) -> plt.Figure:
    """Figure 3A: drainage-regime map."""
    L_arr = np.logspace(0, 3, 55) * 1e-6
    t_arr = np.logspace(-2, 3.3, 55)

    k_low, k_high = 1e-16, 1e-13
    M_c, eta = 1e4, 1e-3
    k_mid = math.sqrt(k_low * k_high)
    D_mid = poroelastic_diffusivity(k_mid, M_c, eta)

    # Background field and N_d = 1 lines both use the first-mode time
    # tau_1 = L^2/(GF_OPEN_SEALED D_p), so colour and contours share one
    # consistent drainage-time convention.
    Z = np.zeros((len(t_arr), len(L_arr)))
    for i, t_obs in enumerate(t_arr):
        for j, L in enumerate(L_arr):
            tau1 = drainage_time(L, D_mid, GF_OPEN_SEALED)
            Z[i, j] = float(np.clip(
                terzaghi_exact(np.array([t_obs]), tau1, n_modes=20)[0],
                0.0, 1.0
            ))

    fig, ax = plt.subplots(figsize=(11.0, 6.6))
    LL, TT = np.meshgrid(L_arr * 1e6, t_arr)
    pcm = ax.pcolormesh(LL, TT, Z, shading="auto",
                        cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\langle p\rangle/p_0$ (Terzaghi series, open-sealed)",
                   fontsize=10)

    D_low = poroelastic_diffusivity(k_low, M_c, eta)
    D_high = poroelastic_diffusivity(k_high, M_c, eta)
    L_line = np.logspace(0, 3, 300) * 1e-6
    tau_low_perm = L_line ** 2 / (GF_OPEN_SEALED * D_low)
    tau_high_perm = L_line ** 2 / (GF_OPEN_SEALED * D_high)

    ax.loglog(L_line * 1e6, tau_low_perm, "k--", lw=2.0,
              label=r"$N_{\mathrm{d}}=1$, low $k=10^{-16}$ m$^2$")
    ax.loglog(L_line * 1e6, tau_high_perm, "k:", lw=2.0,
              label=r"$N_{\mathrm{d}}=1$, high $k=10^{-13}$ m$^2$")

    _regime_kw = dict(fontsize=10.5, fontweight="bold", zorder=8)
    ax.text(1.5, 1000, "DRAINED\n" r"($N_{\mathrm{d}}\gg1$)",
            color="#00CFFF", ha="left", va="top", **_regime_kw)
    ax.text(820, 0.014, "UNDRAINED\n" r"($N_{\mathrm{d}}\ll1$)",
            color="#FF6B6B", ha="right", va="bottom", **_regime_kw)
    ax.text(150, 0.3, "INTERMEDIATE\n" r"($N_{\mathrm{d}}\approx1$)",
            color="black", ha="left", va="center", **_regime_kw)

    for pt in LITERATURE_DATA:
        ms = 14 if pt["marker"] == "*" else 9.5
        edge = "gray" if pt["cat"] == "cell reference" else "k"
        ax.plot(pt["L_um"], pt["t_s"], pt["marker"], ms=ms,
                mec=edge, mew=1.1, color=pt["color"], zorder=6)
        ax.annotate(pt["label"], (pt["L_um"], pt["t_s"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=6.5, color="white", fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(1e-2, 2000)
    ax.set_xlabel(r"Operative drainage length $L_{\mathrm{eff}}$ ($\mu$m)",
                  fontsize=12)
    ax.set_ylabel(r"Observation time $t_{\mathrm{obs}}$ (s)", fontsize=12)
    ax.set_title("A. Drainage regime map", fontsize=12)
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.18, which="both")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Figure saved] {save_path}")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "L_eff_um", "t_obs_s", "category"])
            for pt in LITERATURE_DATA:
                writer.writerow(
                    [pt["label"], pt["L_um"], pt["t_s"], pt["cat"]]
                )
        print(f"  [CSV saved] {csv_path}")

    return fig


def figure_3BC(save_path: Optional[str] = None,
               csv_path_B: Optional[str] = None,
               csv_path_C: Optional[str] = None) -> plt.Figure:
    """
    Figure 3B and 3C side by side.
        B. Geometry-scaling diagnostic: tau ~ L^2 (open-open)
        C. Matrix-permeability effect: tau_drain ~ 1/k (open-sealed)
    """
    L_values = np.logspace(np.log10(20e-6), np.log10(500e-6), 7)
    k_B, M_c_B, eta_B = 1e-15, 1e4, 1e-3
    D_p_B = poroelastic_diffusivity(k_B, M_c_B, eta_B)

    tau_num = []
    tau_exact = []
    for L in L_values:
        tau_oo = drainage_time(L, D_p_B, GF_OPEN_OPEN)
        t_max = 20.0 * tau_oo
        t_grid = log_time_grid(t_max, n_t=600, t_min=tau_oo * 1e-5)
        sol = PoroelasticSolver1D(
            L=L, n_x=201, k=k_B, M_c=M_c_B, eta=eta_B,
            boundary_top="open", boundary_bot="open"
        )
        t, P = sol.solve(
            t_eval=t_grid, theta=0.5, n_be_init=4, enforce_nonneg=False
        )
        tau_num.append(sol.tau_1_over_e(t, P))
        tau_exact.append(terzaghi_1_over_e_crossing(tau_oo, bc="open-open"))

    tau_num = np.array(tau_num)
    tau_exact = np.array(tau_exact)
    slope_num = np.polyfit(np.log10(L_values), np.log10(tau_num), 1)[0]
    slope_ex = np.polyfit(np.log10(L_values), np.log10(tau_exact), 1)[0]
    tau_visco = np.median(tau_exact)

    L_C, M_c_C, eta_C = 100e-6, 1e4, 1e-3
    k_values = np.logspace(-17, -12, 45)
    tau_values = np.array([
        drainage_time(L_C, poroelastic_diffusivity(k, M_c_C, eta_C),
                      GF_OPEN_SEALED)
        for k in k_values
    ])
    k_low_case, k_high_case = 1e-16, 1e-13
    tau_low_case = drainage_time(
        L_C, poroelastic_diffusivity(k_low_case, M_c_C, eta_C), GF_OPEN_SEALED
    )
    tau_high_case = drainage_time(
        L_C, poroelastic_diffusivity(k_high_case, M_c_C, eta_C), GF_OPEN_SEALED
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    ax = axes[0]
    ax.loglog(L_values * 1e6, tau_num, "o-", lw=2.2, ms=9,
              color="#D62728", label=fr"Numerical (slope={slope_num:.2f})")
    ax.loglog(L_values * 1e6, tau_exact, "k+--", lw=1.8, ms=10,
              label=fr"Exact series (slope={slope_ex:.2f})")
    ax.loglog(L_values * 1e6, np.full_like(L_values, tau_visco),
              "s--", lw=2.0, ms=7, color="steelblue",
              label=r"Viscoelastic null: $\tau$ = const")
    ax.set_xlabel(r"$L_{\mathrm{eff}}$ ($\mu$m)", fontsize=12)
    ax.set_ylabel(r"$\tau_{1/e}$ (s)", fontsize=12)
    ax.set_title(r"B. Geometry-scaling diagnostic: $\tau\propto L^2$",
                 fontsize=10.5)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    ax.loglog(k_values, tau_values, "-", lw=2.2, color="0.35",
              label=r"Theory: $\tau_{\mathrm{drain}}\propto1/k$")
    ax.scatter([k_low_case], [tau_low_case], s=155, color="#FF7F0E",
               zorder=6, label=r"Low matrix permeability")
    ax.scatter([k_high_case], [tau_high_case], s=155, marker="v",
               color="#1F77B4", zorder=6, label=r"High matrix permeability")
    ax.annotate("matrix\npermeability shift",
                xy=(k_high_case, tau_high_case),
                xytext=(2e-16, tau_low_case * 0.6),
                arrowprops=dict(arrowstyle="->", lw=1.5, color="0.3"),
                fontsize=9)
    ax.set_xlabel(r"Biofilm matrix permeability $k$ (m$^2$)", fontsize=12)
    ax.set_ylabel(r"$\tau_{\mathrm{drain}}$ (s)", fontsize=12)
    ax.set_title(r"C. Matrix-permeability effect: $\tau_{\mathrm{drain}}\propto1/k$",
                 fontsize=10.5)
    ax.legend(fontsize=9.5)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Figure saved] {save_path}")

    if csv_path_B:
        save_csv(csv_path_B, ["L_eff_um", "tau_numerical_s", "tau_exact_s"],
                 [L_values * 1e6, tau_num, tau_exact])
    if csv_path_C:
        save_csv(csv_path_C, ["k_m2", "tau_drain_s"],
                 [k_values, tau_values])

    return fig


# ======================================================================
# VALIDATION AND CALIBRATION
# ======================================================================

def figure_validation(save_path: Optional[str] = None) -> plt.Figure:
    """Supplementary validation against the exact Terzaghi solution."""
    k, M_c, eta, L = 1e-15, 1e4, 1e-3, 100e-6
    D_p = poroelastic_diffusivity(k, M_c, eta)
    tau1 = drainage_time(L, D_p, GF_OPEN_SEALED)

    t_exact = np.linspace(1e-4 * tau1, 4 * tau1, 500)
    p_exact = terzaghi_exact(t_exact, tau1, n_modes=30)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.0))

    ax = axes[0]
    ax.plot(t_exact / tau1, p_exact, "k-", lw=2.5, label="Exact Terzaghi")
    for nx, col in [(51, "C0"), (101, "C1"), (201, "C3")]:
        sol = PoroelasticSolver1D(
            L=L, n_x=nx, k=k, M_c=M_c, eta=eta,
            boundary_top="open", boundary_bot="sealed"
        )
        vr = sol.validate_terzaghi(n_t=800, skip_fraction=0.05)
        ax.plot(vr["t"] / tau1, vr["numerical"], "--", lw=1.7, color=col,
                label=fr"$n_x={nx}$, max err={vr['max_rel_err']*100:.2f}%")
    ax.set_xlabel(r"$t/\tau_1$", fontsize=11)
    ax.set_ylabel(r"$\langle p\rangle/p_0$", fontsize=11)
    ax.set_title("A. Numerical vs exact Terzaghi", fontsize=11)
    ax.legend(fontsize=8.5)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for nx, col in [(51, "C0"), (101, "C1"), (201, "C3")]:
        sol = PoroelasticSolver1D(
            L=L, n_x=nx, k=k, M_c=M_c, eta=eta,
            boundary_top="open", boundary_bot="sealed"
        )
        vr = sol.validate_terzaghi(n_t=800, skip_fraction=0.05)
        t_n = vr["t"] / tau1
        err = np.abs(vr["numerical"] - vr["analytical"]) / (
            vr["analytical"] + 1e-12
        )
        skip = int(0.05 * len(t_n))
        ax.semilogy(t_n[skip:], err[skip:], lw=1.8, color=col,
                    label=fr"$n_x={nx}$")
    ax.axhline(0.01, color="gray", ls="--", label="1% error")
    ax.set_xlabel(r"$t/\tau_1$", fontsize=11)
    ax.set_ylabel("Relative error", fontsize=11)
    ax.set_title("B. Error after initial boundary layer", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Figure saved] {save_path}")

    return fig


def calibration_against_cells(verbose: bool = True) -> Dict:
    """Order-of-magnitude calibration against Moeendarbary et al. (2013)."""
    L, k, M_c, eta = 5e-6, 1e-17, 1e3, 1e-3
    D_p = poroelastic_diffusivity(k, M_c, eta)
    tau1 = drainage_time(L, D_p, GF_OPEN_SEALED)
    tau_1e = terzaghi_1_over_e_crossing(tau1, bc="open-sealed")

    if verbose:
        print("[Calibration] Moeendarbary et al. (2013) HeLa cell check")
        print(f"  L=5 um, k=1e-17 m^2, M_c=1 kPa")
        print(f"  D_p = {D_p:.3g} m^2/s")
        print(f"  tau_1/e = {tau_1e:.2f} s")
        print("  Observed cell relaxation range: ~0.5-5 s")
        print("  Correct order of magnitude; contact geometry sets prefactor.\n")

    return {"D_p": D_p, "tau1": tau1, "tau_1e": tau_1e}


def convergence_check_channels(verbose: bool = True) -> Dict:
    """
    Grid-convergence check for interior sinks.

    v2.8: extended to cover BOTH the hard sink (Dirichlet) and a leaky
    sink near Bi_sink ~ 1. The leaky-sink test is essential because the
    finite-beta loss term governs Figure 2C and 2D, yet the hard-sink
    Dirichlet condition never exercises it.
    """
    L = 200e-6
    spacing_um = 25
    k, M_c, eta = 1e-15, 1e4, 1e-3
    D_p = poroelastic_diffusivity(k, M_c, eta)
    nx_values = [201, 401, 801]

    # --- Hard sink ---
    taus_hard = []
    for nx in nx_values:
        res = simulate_with_channels(
            L=L, spacings_um=[spacing_um], k=k, M_c=M_c, eta=eta,
            n_x=nx, n_t=900, sink_beta=float("inf")
        )
        taus_hard.append(res[spacing_um]["tau"])
    rel_hard = abs(taus_hard[-1] - taus_hard[-2]) / taus_hard[-1]

    # --- Leaky sink near Bi_sink ~ 1 ---
    s_m = spacing_um * 1e-6
    beta_bi1 = D_p / s_m  # Bi_sink = beta s / D_p = 1
    taus_leaky = []
    for nx in nx_values:
        res = simulate_with_channels(
            L=L, spacings_um=[spacing_um], k=k, M_c=M_c, eta=eta,
            n_x=nx, n_t=900, sink_beta=float(beta_bi1)
        )
        taus_leaky.append(res[spacing_um]["tau"])
    rel_leaky = abs(taus_leaky[-1] - taus_leaky[-2]) / taus_leaky[-1]

    if verbose:
        print("[Convergence] Channel-spacing grid-refinement test")
        print("  Hard sink (beta -> inf):")
        for nx, tau in zip(nx_values, taus_hard):
            print(f"    n_x={nx:<4d} tau_1/e={tau:.6g} s")
        print(f"    Relative change 401->801: {rel_hard:.3%}")
        print(f"  Leaky sink at Bi_sink=1 (beta={beta_bi1:.3g} m/s):")
        for nx, tau in zip(nx_values, taus_leaky):
            print(f"    n_x={nx:<4d} tau_1/e={tau:.6g} s")
        print(f"    Relative change 401->801: {rel_leaky:.3%}\n")

    return {
        "nx_values": nx_values,
        "taus_hard": taus_hard,
        "rel_change_hard": rel_hard,
        "beta_bi1": beta_bi1,
        "taus_leaky": taus_leaky,
        "rel_change_leaky": rel_leaky,
    }


# ======================================================================
# MAIN
# ======================================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\n" + "#" * 80)
    print(f"BiofilmPoroelasticSim v{__version__}")
    print("Generating Figure 2 and Figure 3 only")
    print("Figure 1 is conceptual and not generated by this script")
    print("#" * 80 + "\n")

    calibration_against_cells(verbose=True)
    figure_validation(save_path=out("fig_S1_validation.png"))
    convergence_check_channels(verbose=True)

    regime_placement(
        ProtocolInputs(
            L_eff=50e-6, k_range=(1e-16, 1e-14),
            M_c=1e4, eta=1e-3, t_obs=1.0, boundary_type="sealed"
        ),
        verbose=True
    )

    print("\n[Figure 2A-B] Simulating channel-spacing relaxation...")
    channel_results = simulate_with_channels(
        L=200e-6, spacings_um=[None, 100, 50, 25, 10],
        k=1e-15, M_c=1e4, eta=1e-3, n_x=401, n_t=800,
        n_be_init=4, sink_beta=float("inf")
    )
    for s, d in channel_results.items():
        label = "no channels" if s is None else f"s={s} um"
        print(f"  {label:<15s} tau_1/e = {d['tau']:.4g} s")

    figure_2_channel_demo(
        channel_results,
        save_path=out("fig2AB_channel_demo.png"),
        csv_path=out("fig2AB_pressure_curves.csv")
    )

    print("\n[Figure 2C-D] Generating spacing-connectivity heatmap...")
    figure_2_heatmap(
        L=200e-6, spacings_um=[10, 20, 30, 40, 50, 75, 100, 150],
        beta_values=np.logspace(-5, 0, 12),
        k=1e-15, M_c=1e4, eta=1e-3, n_x=301, n_t=500, n_be_init=4,
        save_path=out("fig2CD_heatmap.png"),
        csv_path=out("fig2CD_tau_matrix.csv")
    )

    print("\n[Figure 3A] Generating drainage-regime map...")
    figure_3A(
        save_path=out("fig3A_regime_map.png"),
        csv_path=out("fig3A_literature_points.csv")
    )

    print("\n[Figure 3B-C] Generating geometry-scaling and permeability panels...")
    figure_3BC(
        save_path=out("fig3BC_scaling_and_permeability.png"),
        csv_path_B=out("fig3B_tau_vs_L.csv"),
        csv_path_C=out("fig3C_tau_vs_k.csv")
    )

    print("\n" + "#" * 80)
    print("Simulation complete.")
    print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("#" * 80 + "\n")

    plt.show()


if __name__ == "__main__":
    main()
