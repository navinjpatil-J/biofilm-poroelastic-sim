#!/usr/bin/env python3
"""
biofilm_poroelastic_sim.py
==========================

Poroelastic drainage model and figure code for the review article

    "Interfacial drainage regime as a coordinate linking biofilm mechanics
     and transport"

Version 3.1.0. Released under the MIT License (see LICENSE).


Contents
--------
Running this script regenerates Figures 2-5 and Supplementary Figure S1 of
the article, together with the source data for each figure:

    Figure 2   Interior channels (hard sinks) and leaky pressure-relief
               interfaces shorten drainage (1D solver).
    Figure 3   (a) Drainage-regime map with literature placements;
               (b) ensemble test of reported moduli against a drainage
               proxy; (c), (d) tau ~ L_eff^2 and tau ~ 1/k scaling checks.
    Figure 4   Mandel-Cryer benchmark: interior pressure overshoot versus
               monotone 1D drainage.
    Figure 5   Model recovery: a single relaxation curve cannot identify
               the mechanism, a length series can, and conductance data
               separate k from M_c.
    Figure S1  Solver validation against the exact Terzaghi series.

Figures 1 and 6 are produced by the companion scripts
figure1_drainage_boundary_access.py and
figure6_clinical_active_architecture.py.


How to run
----------
    pip install -r requirements.txt
    python biofilm_poroelastic_sim.py

No input files or command-line options are needed. Outputs are written to
biofilm_sim_outputs/ next to this script, one file per figure with all
panels assembled:

    Figure_2 ... Figure_5 (.png, .tiff)     500 dpi, 3,740 px wide
    Figure_S1_Validation (.png, .tiff)      1000 dpi, 7,480 px wide
    Figure_*.csv                            data plotted in each figure

A full run takes about 4 min and peaks at about 1.5 GB of memory (measured
on a 2-core machine). The console log repeats the numbers quoted in the
article (drainage times, the Fig. 3(b) Spearman statistic, the Mandel peak,
the Fig. 5 fit and slopes, and the grid-convergence errors), so results can
be checked without opening the figures.


Reproducibility
---------------
- The calculations are deterministic; there is no random input.
- Figures use Arial when it is installed and DejaVu Sans (bundled with
  Matplotlib) otherwise. The font changes text widths, not numbers.
- requirements.txt lists the package versions used to produce and check
  the figures.
- Each figure is rendered at full resolution and then resampled to a
  common width of 7.48 in, so all numbered figures share one width.


Model
-----
Linearized 1D Biot/Terzaghi consolidation after a step volumetric load:

    dp/dt = D_p d^2p/dx^2,      D_p = k M_c / eta

    p     excess pore pressure (Pa)
    D_p   poroelastic diffusivity (m^2 s^-1)
    k     intrinsic permeability (m^2)
    M_c   drained constrained modulus (Pa)
    eta   pore-fluid viscosity (Pa s)

Drainage time tau_drain = L_eff^2 / (gf D_p), with geometry factor

    GF_OPEN_OPEN   = pi^2       both faces drained
    GF_OPEN_SEALED = pi^2 / 4   open top, sealed base

Interior pressure sinks (Fig. 2):
- Hard sink: beta -> infinity, imposed as Dirichlet p = 0.
- Leaky sink: finite pressure-relief coefficient beta (m s^-1), imposed as
  a mesh-consistent Robin-type relief term.
- The channel spacing s is the full centre-to-centre distance between
  adjacent sinks, and the sink connectivity number is Bi_sink = beta s / D_p.

Figure 4 uses the analytical Mandel solution (Cheng and Detournay, 1988);
the MANDEL-CRYER PROBLEM section states the geometry and conventions.
Figure 5 calls the stretched-exponential exponent gamma, as the article
does; beta always denotes the pressure-relief coefficient.


Parameter values
----------------
- Permeability spans 1e-17 to 1e-14 m^2 (Kozeny-Carman plausible range)
  wherever a band is shown. Each figure function sets its other parameter
  values in its signature or first lines; the defaults reproduce the
  article.
- Literature markers in Fig. 3 are illustrative order-of-magnitude
  placements from each study's reported length scale and observation
  window, not fitted drainage states. LITERATURE_DATA records the source
  of every plotted modulus.


Validation
----------
The solver is checked against the exact Terzaghi series (Figure S1) and by
grid refinement for hard and leaky sinks (console). A cell-scale check
against Moeendarbary et al. (2013) confirms the order of magnitude of the
drainage time.


Code map
--------
    Core physics      poroelastic_diffusivity, drainage_time, terzaghi_exact
    1D solver         PoroelasticSolver1D
    Regime placement  ProtocolInputs, regime_placement, classify_regime
    Figure 2          simulate_with_channels, figure_2_channel_demo,
                      figure_2_heatmap
    Figure 3          LITERATURE_DATA, figure_3a, figure_3bcd,
                      spearman_exact_permutation
    Figure 4          mandel_eigenvalues, mandel_pressure, figure_4_mandel
    Figure 5          figure_5_identifiability
    Validation        figure_validation, convergence_check_channels,
                      plausibility_check_cells
    Output            save_figure_formats, compose_vertical, save_csv
"""

from __future__ import annotations

import itertools
import math
import warnings
import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import factorized
from scipy.optimize import brentq, least_squares
from scipy.stats import rankdata, spearmanr
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import rcParams
from PIL import Image


# ======================================================================
# STYLE AND OUTPUT
# ======================================================================

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

__version__ = "3.1.0"


def _beta_pow10_label(beta):
    """Legend text for a pressure-relief coefficient, e.g. beta = 1.2 x 10^-1 m s^-1."""
    exponent = int(math.floor(math.log10(beta)))
    mantissa = beta / 10.0 ** exponent
    if round(mantissa, 1) >= 10.0:
        mantissa, exponent = mantissa / 10.0, exponent + 1
    return (fr"$\beta={mantissa:.1f}\times10^{{{exponent}}}$"
            r" m s$^{-1}$")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "biofilm_sim_outputs")

# Artwork resolution policy.
ARTWORK_DPI = 500     # combination line + halftone (colour), minimum 500 dpi
LINE_ART_DPI = 1000   # bitmapped line drawings, minimum 1000 dpi

# Final width shared by every numbered file: 7.48 in (190 mm), i.e.
# 3,740 px at 500 dpi and 7,480 px at 1000 dpi. Wider canvases are
# resampled to this width.
FINAL_PAGE_WIDTH_IN = 7.48
FULL_PAGE_WIDTH_PX = int(round(FINAL_PAGE_WIDTH_IN * ARTWORK_DPI))  # 3740

# Small vertical gap between assembled panel blocks (default).
PANEL_GAP_PX = 35


def out(filename: str) -> str:
    """Return normalized absolute path inside the output directory next to the script."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.abspath(os.path.normpath(os.path.join(OUTPUT_DIR, filename.strip())))


def safe_save_image(image: Image.Image,
                    path: str,
                    fmt: str,
                    dpi: int = ARTWORK_DPI,
                    **kwargs) -> str:
    """
    Save a PIL image. If the target file is locked (for example, open in an
    image viewer on Windows), save it as '<name>_latest<ext>' instead.
    """
    norm_path = os.path.abspath(os.path.normpath(path))
    os.makedirs(os.path.dirname(norm_path), exist_ok=True)

    save_kwargs = dict(kwargs)
    save_kwargs["dpi"] = (int(dpi), int(dpi))

    try:
        image.save(norm_path, format=fmt, **save_kwargs)
        return norm_path
    except (OSError, PermissionError) as e:
        base, ext = os.path.splitext(norm_path)
        fallback_path = f"{base}_latest{ext}"
        try:
            image.save(fallback_path, format=fmt, **save_kwargs)
            print(f"  [WARNING] Could not overwrite '{norm_path}' (file may be open in a viewer). "
                  f"Saved as '{fallback_path}' instead.")
            return fallback_path
        except Exception as fallback_err:
            raise OSError(
                f"Failed to save image to '{norm_path}' or fallback '{fallback_path}'. "
                f"Please ensure the image file is closed in any open viewer/program. Details: {e}"
            ) from fallback_err


# ======================================================================
# ARTWORK SAVING AND ASSEMBLY
# ======================================================================

def save_figure_formats(fig: plt.Figure,
                        stem: str,
                        dpi: int = ARTWORK_DPI) -> Tuple[str, str]:
    """
    Save a Matplotlib figure in both PNG and LZW-compressed TIFF formats,
    width-normalised to FINAL_PAGE_WIDTH_IN at the export resolution.

    The figure is rendered at full resolution first (supersampling), then
    resampled down to the shared page width so that every numbered file
    delivers the same physical width: 3,740 px at 500 dpi and 7,480 px at
    1000 dpi.

    Parameters
    ----------
    fig
        Matplotlib figure to save.
    stem
        Output filename without extension.
    dpi
        Raster resolution (also the dpi tag of the delivered files).

    Returns
    -------
    png_path, tiff_path
        Paths to the saved files.
    """
    png_path = out(f"{stem}.png")
    tiff_path = out(f"{stem}.tiff")
    tmp_png = out(f"_render_{stem}.png")

    common = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "pad_inches": 0.04,
        "facecolor": "white",
        "edgecolor": "none",
    }
    fig.savefig(tmp_png, format="png", **common)

    target_w = int(round(FINAL_PAGE_WIDTH_IN * dpi))
    try:
        with Image.open(tmp_png) as rendered:
            image = rendered.convert("RGB")
        if image.width != target_w:
            try:
                resampling = Image.Resampling.LANCZOS
            except AttributeError:
                resampling = Image.LANCZOS
            target_h = int(round(image.height * target_w / image.width))
            image = image.resize((target_w, target_h), resample=resampling)

        actual_png = safe_save_image(image, png_path, fmt="PNG",
                                     dpi=dpi, optimize=True)
        actual_tiff = safe_save_image(image, tiff_path, fmt="TIFF",
                                      dpi=dpi, compression="tiff_lzw")
    finally:
        try:
            os.remove(tmp_png)
        except OSError:
            pass

    print(f"  [PNG saved]  {actual_png} ({target_w} px @ {dpi} dpi)")
    print(f"  [TIFF saved] {actual_tiff} ({target_w} px @ {dpi} dpi)")

    return actual_png, actual_tiff


def save_working_block(fig: plt.Figure,
                       filename: str,
                       dpi: int = ARTWORK_DPI) -> str:
    """Save a temporary high-resolution PNG used to assemble a full figure."""
    path = out(filename)

    fig.savefig(
        path,
        format="png",
        dpi=dpi,
        bbox_inches="tight",
        pad_inches=0.04,
        facecolor="white",
        edgecolor="none"
    )

    return path


def compose_vertical(block_paths: Sequence[str],
                     stem: str,
                     target_width_px: int = FULL_PAGE_WIDTH_PX,
                     gap_px: int = PANEL_GAP_PX,
                     dpi: int = ARTWORK_DPI,
                     remove_blocks: bool = True,
                     block_margins: Optional[Sequence[Tuple[int, int]]] = None) -> Tuple[str, str]:
    """
    Stack panel blocks vertically and save one complete numbered figure.

    Each block is resized proportionally to fit the target width, with optional
    per-block left and right margins (block_margins=[(left_px, right_px), ...]).
    """
    if not block_paths:
        raise ValueError("At least one block path must be supplied.")

    try:
        resampling = Image.Resampling.LANCZOS
    except AttributeError:
        resampling = Image.LANCZOS

    if block_margins is None:
        block_margins = [(0, 0)] * len(block_paths)

    images = []
    for path, (margin_left, margin_right) in zip(block_paths, block_margins):
        with Image.open(path) as source:
            src_image = source.convert("RGB").copy()

        content_width = max(100, target_width_px - margin_left - margin_right)
        new_height = int(round(src_image.height * content_width / src_image.width))
        resized_content = src_image.resize((content_width, new_height), resample=resampling)

        block_img = Image.new("RGB", (target_width_px, new_height), color=(255, 255, 255))
        block_img.paste(resized_content, (margin_left, 0))
        images.append(block_img)

    total_height = (
        sum(image.height for image in images)
        + gap_px * (len(images) - 1)
    )

    canvas = Image.new(
        "RGB",
        (target_width_px, total_height),
        color=(255, 255, 255)
    )

    y_position = 0
    for image in images:
        canvas.paste(image, (0, y_position))
        y_position += image.height + gap_px

    png_path = out(f"{stem}.png")
    tiff_path = out(f"{stem}.tiff")

    actual_png = safe_save_image(canvas, png_path, fmt="PNG", dpi=dpi, optimize=True)
    actual_tiff = safe_save_image(canvas, tiff_path, fmt="TIFF", dpi=dpi, compression="tiff_lzw")

    print(
        f"  [Composite PNG saved]  {actual_png} "
        f"({canvas.width} x {canvas.height} px)"
    )
    print(
        f"  [Composite TIFF saved] {actual_tiff} "
        f"({canvas.width} x {canvas.height} px)"
    )

    if remove_blocks:
        for path in block_paths:
            try:
                os.remove(path)
            except OSError:
                pass

    return actual_png, actual_tiff


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


# ======================================================================
# CONSTANTS
# ======================================================================

GF_OPEN_OPEN = math.pi ** 2
GF_OPEN_SEALED = math.pi ** 2 / 4.0


# ======================================================================
# CORE PHYSICS
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
    """Exact spatially averaged Terzaghi solution, open-open boundaries."""
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
    Find the 1/e crossing time of a relaxation curve.

    If relative_to_initial is True, the target is y(0)/e. This convention
    is used for sink simulations because zero-width hard-sink nodes slightly
    reduce the initial spatial mean.
    """
    y = np.asarray(y, dtype=float)
    target = y[0] / math.e if relative_to_initial else 1.0 / math.e

    if y[0] <= target:
        return 0.0

    for i in range(1, len(t)):
        if y[i] <= target:
            t0, t1 = t[i - 1], t[i]
            y0, y1 = y[i - 1], y[i]
            return t0 + (t1 - t0) * (y0 - target) / (y0 - y1)

    return float(t[-1])


def recommended_theta_for_sink(beta: float) -> float:
    """Use Backward Euler for finite-beta sinks and Crank-Nicolson otherwise."""
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
        if not math.isinf(self.beta) and self.beta < 0.0:
            raise ValueError(
                "sink_beta must be non-negative "
                "(use inf for the hard-sink limit)."
            )

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

        # Boundary conditions
        if self.bb == "open":
            A[0, :] = 0.0
            A[0, 0] = 1.0
            B[0, :] = 0.0
        else:
            A[0, 0] = 1.0 + 2.0 * rA
            A[0, 1] = -2.0 * rA
            B[0, 0] = 1.0 - 2.0 * rB
            B[0, 1] = 2.0 * rB

        if self.bt == "open":
            A[-1, :] = 0.0
            A[-1, -1] = 1.0
            B[-1, :] = 0.0
        else:
            A[-1, -1] = 1.0 + 2.0 * rA
            A[-1, -2] = -2.0 * rA
            B[-1, -1] = 1.0 - 2.0 * rB
            B[-1, -2] = 2.0 * rB

        # Interior pressure-relief interfaces
        if math.isinf(self.beta):
            for idx in self._sink_idx:
                A[idx, :] = 0.0
                A[idx, idx] = 1.0
                B[idx, :] = 0.0
        else:
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
    """Protocol metadata for drainage-regime placement.

    Drainage-number placement depends on L_eff, k_range, eta, M_c,
    t_obs, and boundary_type only. alpha and strain_mode are recorded
    protocol metadata (they govern the apparent-modulus reading of
    the model elsewhere) and do not enter the N_d computation here.
    """
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
    if p.boundary_type not in ("sealed", "open"):
        raise ValueError(
            "boundary_type must be 'sealed' or 'open'; "
            f"got {p.boundary_type!r}."
        )
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
    """
    Return regularly spaced interior sink positions inside (0, L).

    Spacing convention (used consistently in text, caption, and code):
    `spacing_um` is the FULL centre-to-centre spacing s between
    adjacent interior sinks. Sinks are placed at x = s, 2s, 3s, ...,
    so s is simultaneously the base-to-first-sink distance and the
    nearest-neighbour sink distance. Points between adjacent sinks
    therefore lie within s/2 of their nearest sink, and
    Bi_sink = beta * s / D_p is defined with the full spacing s.
    """
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
    """
    Simulate pressure relaxation for different interior channel spacings.

    Note: interior sinks use the FULL-spacing convention
    (see `channel_sink_positions`). The sink number is
    Bi_sink = beta * s / D_p with that full spacing s; the
    hard-sink limit is beta -> infinity (Dirichlet p = 0).
    """
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
                beta_label = (fr"{_beta_pow10_label(sink_beta)}, "
                              fr"$\mathrm{{Bi}}_\mathrm{{sink}}={Bi:.2g}$")
            label = fr"$s={s_um}$ $\mathrm{{\mu}}$m, {beta_label}"

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
    """Generate Figure 2(a)-(b): channel spacing and pressure relaxation."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))

    # Panel (a)
    ax = axes[0]
    colors = plt.cm.plasma(np.linspace(0.18, 0.85, len(results)))
    for (s_um, d), col in zip(results.items(), colors):
        lw = 2.7 if s_um is None else 2.1
        # Plot only t > 0 on the log axis. Shifting the t = 0 sample by a small
        # epsilon would stretch the axis down to that epsilon, far below the
        # dynamics and the validity window of the quasi-static model.
        t_raw = np.asarray(d["t"], dtype=float)
        pos = t_raw > 0.0
        ax.semilogx(t_raw[pos], np.asarray(d["p_avg_plot"])[pos],
                    lw=lw, color=col, label=d["label"])

    ax.axhline(1.0 / math.e, color="k", ls=":", lw=1.4, label=r"$1/e$")
    ax.set_ylim(-0.02, 1.05)
    # Physical display window: one decade below the fastest channelised
    # relaxation (~1e-3 s for s = 10 um) up to 5*tau_ref (~8 s) so the
    # slowest (no-channel) curve completes its decay on-scale.
    ax.set_xlim(1e-5, 1e1)
    ax.set_xlabel(r"Time $t$ (s)", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"Remaining normalized excess pore pressure",
                  fontsize=12, fontweight="bold")
    ax.set_title(
        "(a) Pore-pressure relaxation vs. channel spacing",
        fontsize=17, fontweight="normal"
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    # Panel (b)
    ax = axes[1]
    spacings = [s for s in results.keys() if s is not None]
    taus = np.array([results[s]["tau"] for s in spacings], dtype=float)
    s_arr = np.array(spacings, dtype=float)

    ax.loglog(s_arr, taus, "o-", lw=2.3, ms=9,
              color="#D62728", label=r"Simulation $\tau_{1/e}$")
    ax.loglog(s_arr, taus[0] * (s_arr / s_arr[0]) ** 2,
              "k--", lw=1.6, alpha=0.6, label=r"$\tau\propto s^2$ guide")

    if None in results:
        ax.axhline(results[None]["tau"], color="steelblue", ls=":",
                   lw=2.0, label=r"No-channel $\tau_{1/e}$")

    ax.set_xlabel(r"Channel spacing $s$ ($\mathrm{\mu}$m)",
                  fontsize=14, fontweight="bold")
    ax.set_ylabel(r"$\tau_{1/e}$ (s)", fontsize=14, fontweight="bold")
    ax.set_title(
        r"(b) $\tau_{\mathrm{drain}}$ set by spacing, not thickness",
        fontsize=17, fontweight="normal"
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
    Generate Figure 2(c)-(d).

    tau_ref is the no-channel exact Terzaghi tau_1/e in open-top
    sealed-base geometry. Finite beta is a reduced pressure-relief
    representation of channel connectivity.
    """
    if spacings_um is None:
        spacings_um = [10, 20, 30, 40, 50, 75, 100, 150]

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

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2))

    # Panel (c)
    ax = axes[0]
    Z = np.log10(np.clip(TAU / tau_ref, 1e-4, 1.5))
    im = ax.pcolormesh(S_arr, B_arr, Z, shading="auto",
                       cmap="magma_r", vmin=-3, vmax=0)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(r"$\log_{10}(\tau/\tau_{\mathrm{ref}})$",
                   fontsize=13, fontweight="bold")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"Channel spacing $s$ ($\mathrm{\mu}$m)",
                  fontsize=14, fontweight="bold")
    ax.set_ylabel(r"Sink pressure-relief coefficient $\beta$ (m s$^{-1}$)",
                  fontsize=14, fontweight="bold")
    ax.set_title(
        r"(c) Sink effectiveness requires $\mathrm{Bi}_{\mathrm{sink}}\gtrsim 1$",
        fontsize=17, fontweight="normal"
    )

    beta_bi1 = D_p / (S_arr * 1e-6)
    valid = (beta_bi1 >= B_arr.min()) & (beta_bi1 <= B_arr.max())
    if np.any(valid):
        ax.plot(S_arr[valid], beta_bi1[valid], "w--", lw=2.0,
                label=r"$\mathrm{Bi}_{\mathrm{sink}}=1$")
        ax.legend(fontsize=9, loc="lower left")

    # Panel (d)
    ax = axes[1]
    ax.loglog(S_arr, tau_hard / tau_ref, "k^-", lw=2.4, ms=9,
              label=r"$\beta\to\infty$ hard sink")

    picks = [B_arr[-3], B_arr[-5], B_arr[-7]]
    for beta in picks:
        idx = int(np.argmin(np.abs(B_arr - beta)))
        ax.loglog(S_arr, TAU[idx, :] / tau_ref, "o--", lw=1.8, ms=7,
                  label=_beta_pow10_label(B_arr[idx]))

    ax.axhline(1.0, color="gray", ls=":", lw=1.5,
               label=r"No-channel $\tau_{1/e}$ reference")
    ax.loglog(S_arr, (S_arr / S_arr[0]) ** 2 * tau_hard[0] / tau_ref,
              "k:", lw=1.2, alpha=0.55, label=r"$s^2$ guide")

    ax.set_xlabel(r"Channel spacing $s$ ($\mathrm{\mu}$m)",
                  fontsize=14, fontweight="bold")
    ax.set_ylabel(r"$\tau/\tau_{\mathrm{ref}}$",
                  fontsize=14, fontweight="bold")
    ax.set_title(
        "(d) Connectivity, not morphology, sets drainage",
        fontsize=17, fontweight="normal"
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
# FIGURE 3 SIMULATIONS
# ======================================================================

# Literature placements are illustrative order-of-magnitude estimates
# based on reported specimen/probe length scales and observation windows.
# They are not fitted drainage states.
#
# The E_* fields carry the reported modulus and its source. Markers without
# an extractable numeric modulus have E_Pa = None and an E_status starting
# with "na"; they are excluded from the Fig. 3(b) ensemble test, and E_src
# records why.

LITERATURE_DATA = [
    # PAO1 streamers grown at tau_wg = 5.1 N/m^2; Table II replicate mean +/- SD
    # (n = 6). t_obs is the stated 10 s measurement interval; L is the ~10^2 um
    # layer and streamer scale. These are the paper's own specimens, not those
    # of Stoodley et al. (1999).
    {"label": "Klapper 02", "L_um": 100, "t_s": 10.0,
     "marker": "D", "color": "#FDAE61", "cat": "deformation",
     "role": "in situ shear-induced deformation (flow cell) with viscoelastic-fluid model",
     "E_Pa": 64.67, "E_lo": 43.64, "E_hi": 85.70,
     "E_kind": "G = tau_w/strain, stepped wall-shear stress-strain test",
     "E_src": "apparent shear modulus G = 64.67 +/- 21.03 Pa, PAO1 grown at "
              "5.1 N/m^2, n = 6 microcolonies, Table II (Klapper et al., 2002)",
     "E_status": "reported"},

    # Modulus from the abstract of Stoodley et al. (1999); placement follows its
    # in situ flow-cell deformation protocol.
    {"label": "Stoodley 99", "L_um": 50, "t_s": 0.1,
     "marker": "s", "color": "#F46D43", "cat": "deformation",
     "role": "in situ shear-induced deformation study",
     "offset": (-6, -10),
     "E_Pa": 26.0, "E_lo": 17.0, "E_hi": 40.0,
     "E_kind": "E_app (structural deformation)",
     "E_src": "E_app = 17-40 Pa for mixed- and pure-culture biofilms; "
              "G = 27 Pa for the mixed culture "
              "(Stoodley et al., 1999, abstract)",
     "E_status": "reported"},

    {"label": "Shaw 04", "L_um": 200, "t_s": 300.0,
     "marker": "s", "color": "#FEE08B", "cat": "rheology",
     "role": "bulk rheology; order-of-minutes viscoelastic relaxation",
     "E_Pa": None, "E_lo": None, "E_hi": None,
     "E_kind": "",
     "E_src": "compilation paper; elastic shear moduli span "
              "1e-2-1e6 Pa with ~18 min common relaxation; no "
              "single value extracted",
     "E_status": "na-compilation"},

    {"label": "Wilking 13", "L_um": 200, "t_s": 100.0,
     "marker": "^", "color": "#4575B4", "cat": "channel transport",
     "role": "agar-grown B. subtilis channel-transport reference",
     "E_Pa": None, "E_lo": None, "E_hi": None,
     "E_kind": "",
     "E_src": "transport study; no mechanical modulus reported",
     "E_status": "na"},

    {"label": "Gloag 20", "L_um": 2, "t_s": 1.0,
     "marker": "o", "color": "#66BD63", "cat": "review",
     "role": "review-level micro-scale mechanics placement",
     "E_Pa": None, "E_lo": None, "E_hi": None,
     "E_kind": "",
     "E_src": "review-level marker; no single value",
     "E_status": "na-review"},

    {"label": "Körstgens 01", "L_um": 400, "t_s": 100.0,
     "marker": "s", "color": "#D73027", "cat": "compression",
     "role": "uniaxial compression thickness and creep/yield window",
     "E_Pa": None, "E_lo": None, "E_hi": None,
     "E_kind": "",
     "E_src": "compressive stress-strain to failure reported; "
              "numeric modulus not extractable from accessible text",
     "E_status": "na"},

    {"label": "Towler 03", "L_um": 40, "t_s": 180.0,
     "marker": "s", "color": "#FC8D59", "cat": "rheometry",
     "role": "rotating-disk rheometer creep window",
     # Shear modulus from the abstract of Towler et al. (2003).
     "E_Pa": 2.2, "E_lo": 0.2, "E_hi": 24.0,
     "E_kind": "G (creep, Burger fit)",
     "E_src": "shear modulus G = 0.2-24 Pa from rheometer creep "
              "(Towler et al., 2003, abstract); Discussion gives "
              "G1 = 0.3-45 Pa",
     "E_status": "reported"},

    {"label": "Kundukad 16", "L_um": 3, "t_s": 0.5,
     "marker": "o", "color": "#1A9850", "cat": "AFM",
     "role": "superficial AFM indentation scale",
     "E_Pa": 20.0, "E_lo": 17.0, "E_hi": 35.0,
     "E_kind": "E (AFM Hertz, superficial)",
     "E_src": "microcolonies 20 Pa, plains 35 Pa, size-matched "
              "17 Pa; plains heterogeneity 20-1000 Pa "
              "(Kundukad et al., 2016)",
     "E_status": "reported"},

    {"label": "Lau 09", "L_um": 2, "t_s": 0.5,
     "marker": "o", "color": "#A6D96A", "cat": "AFM",
     "role": "microbead force spectroscopy hold window",
     "E_Pa": 5.0e4, "E_lo": 1.5e4, "E_hi": 1.7e5,
     "E_kind": "E1 instantaneous (MBFS)",
     "E_src": "E1 = 1.5e4-1.7e5 Pa across strains (PAO1 early "
              "1.7e5 Pa); delayed E2 up to 1.1e6 Pa "
              "(Lau et al., 2009)",
     "E_status": "reported"},

    {"label": "Derlon 16", "L_um": 50, "t_s": 1000.0,
     "marker": "v", "color": "#C2A5CF", "cat": "hydraulic",
     "role": "membrane-biofilm hydraulic resistance evolution",
     "E_Pa": None, "E_lo": None, "E_hi": None,
     "E_kind": "",
     "E_src": "hydraulic resistance and deformation study; no "
              "direct modulus reported",
     "E_status": "na"},

    {"label": "Moeendarbary 13*", "L_um": 5, "t_s": 2.0,
     "marker": "*", "color": "gray", "cat": "cell reference",
     "role": "cell-scale poroelastic reference; not a biofilm point",
     "E_Pa": 600.0, "E_lo": 400.0, "E_hi": 900.0,
     "E_kind": "E (AFM indentation, cells)",
     "E_src": "HeLa 0.9 +/- 0.4 kPa, HT1080 0.4 +/- 0.2 kPa, "
              "MDCK 0.4 +/- 0.1 kPa (Moeendarbary et al., 2013)",
     "E_status": "reported-cell-ref"},
]


def figure_3a(save_path: Optional[str] = None,
              csv_path: Optional[str] = None) -> plt.Figure:
    """Generate Figure 3(a): drainage-regime map."""
    L_arr = np.logspace(0, 3, 55) * 1e-6
    t_arr = np.logspace(-2, 3.3, 55)

    k_low, k_high = 1e-17, 1e-14
    M_c, eta = 1e4, 1e-3
    k_mid = math.sqrt(k_low * k_high)
    D_mid = poroelastic_diffusivity(k_mid, M_c, eta)

    Z = np.zeros((len(t_arr), len(L_arr)))
    for i, t_obs in enumerate(t_arr):
        for j, L in enumerate(L_arr):
            tau1 = drainage_time(L, D_mid, GF_OPEN_SEALED)
            Z[i, j] = float(np.clip(
                terzaghi_exact(np.array([t_obs]), tau1, n_modes=20)[0],
                0.0, 1.0
            ))

    fig, ax = plt.subplots(figsize=(11.5, 6.6))
    LL, TT = np.meshgrid(L_arr * 1e6, t_arr)
    pcm = ax.pcolormesh(LL, TT, Z, shading="auto",
                        cmap="RdYlBu_r", vmin=0.0, vmax=1.0)
    cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\langle p\rangle/p_0$ (Terzaghi series, open-sealed)",
                   fontsize=12, fontweight="bold")

    D_low = poroelastic_diffusivity(k_low, M_c, eta)
    D_high = poroelastic_diffusivity(k_high, M_c, eta)
    L_line = np.logspace(0, 3, 300) * 1e-6
    tau_low_perm = L_line ** 2 / (GF_OPEN_SEALED * D_low)
    tau_high_perm = L_line ** 2 / (GF_OPEN_SEALED * D_high)

    ax.loglog(L_line * 1e6, tau_low_perm, "k--", lw=2.0,
              label=r"$N_{\mathrm{d}}=1$, low $k=10^{-17}$ m$^2$")
    ax.loglog(L_line * 1e6, tau_high_perm, "k:", lw=2.0,
              label=r"$N_{\mathrm{d}}=1$, high $k=10^{-14}$ m$^2$")

    _regime_kw = dict(fontsize=11.5, fontweight="bold", zorder=8)
    ax.text(1.4, 900, "DRAINED\n" r"($N_{\mathrm{d}}\gg1$)",
            color="#00CFFF", ha="left", va="top", **_regime_kw)
    ax.text(820, 0.015, "UNDRAINED\n" r"($N_{\mathrm{d}}\ll1$)",
            color="#FF6B6B", ha="right", va="bottom", **_regime_kw)

    # Inside the intermediate zone, clear of the dotted N_d = 1 line
    ax.text(65, 0.22, "INTERMEDIATE\n" r"($N_{\mathrm{d}}\approx1$)",
            color="black", ha="left", va="center", **_regime_kw)

    # Label offsets (points) for the literature markers
    panel_a_offsets = {
        "Gloag 20": (-10, 8),      # Top-left of marker
        "Lau 09": (-10, -14),      # Bottom-left of marker
        "Kundukad 16": (10, -6),   # Right of marker
        "Moeendarbary 13*": (10, 4),
        "Stoodley 99": (-10, -13),
        "Klapper 02": (10, 4),
        "Towler 03": (10, 3),
        "Wilking 13": (10, -8),
        "Shaw 04": (10, 4),
        "Körstgens 01": (10, 4),
        "Derlon 16": (10, 4),
    }

    for pt in LITERATURE_DATA:
        ms = 14 if pt["marker"] == "*" else 9.5
        edge = "gray" if pt["cat"] == "cell reference" else "k"
        ax.plot(pt["L_um"], pt["t_s"], pt["marker"], ms=ms,
                mec=edge, mew=1.1, color=pt["color"], zorder=6)
        label_offset = panel_a_offsets.get(pt["label"], pt.get("offset", (6, 6)))
        ha = "right" if label_offset[0] < 0 else "left"
        ax.annotate(pt["label"], (pt["L_um"], pt["t_s"]),
                    xytext=label_offset, textcoords="offset points",
                    fontsize=9.0, color="white", fontweight="bold",
                    ha=ha,
                    path_effects=[pe.withStroke(linewidth=2.8,
                                                foreground="black")])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1, 1000)
    ax.set_ylim(1e-2, 2000)
    ax.set_xlabel(r"Operative drainage length $L_{\mathrm{eff}}$ ($\mathrm{\mu}$m)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Observation time $t_{\mathrm{obs}}$ (s)",
                  fontsize=12, fontweight="bold")
    ax.set_title("(a) Model-based drainage-regime map",
                 fontsize=14.0, pad=10)
    ax.legend(fontsize=9.5, loc="lower left", framealpha=0.92)
    ax.grid(alpha=0.2, which="both")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Figure saved] {save_path}")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["label", "L_eff_um", "t_obs_s", "category",
                             "role", "E_Pa", "E_lo_Pa", "E_hi_Pa",
                             "E_kind", "E_status", "E_source"])
            for pt in LITERATURE_DATA:
                writer.writerow([
                    pt["label"], pt["L_um"], pt["t_s"],
                    pt["cat"], pt.get("role", ""),
                    pt.get("E_Pa", ""), pt.get("E_lo", ""),
                    pt.get("E_hi", ""), pt.get("E_kind", ""),
                    pt.get("E_status", ""), pt.get("E_src", ""),
                ])
        print(f"  [CSV saved] {csv_path}")

    return fig


def spearman_exact_permutation(x: Sequence[float], y: Sequence[float]) -> Tuple[float, float]:
    """
    Spearman's rho and its exact two-sided permutation P value.

    All n! pairings of the y ranks are enumerated (feasible for n <= 8),
    and P is the fraction with |rho| >= |rho_observed|. Average ranks
    handle ties. Used for the Fig. 3(b) ensemble test, where n = 5 makes
    the usual t-distribution approximation unreliable.
    """
    rx = rankdata(x)
    ry = rankdata(y)
    n = len(rx)
    if n > 8:
        raise ValueError("exact enumeration is limited to n <= 8")
    rho_obs = float(np.corrcoef(rx, ry)[0, 1])
    hits = 0
    total = 0
    for perm in itertools.permutations(range(n)):
        r = float(np.corrcoef(rx, ry[list(perm)])[0, 1])
        hits += abs(r) >= abs(rho_obs) - 1e-12
        total += 1
    return rho_obs, hits / total


def figure_3bcd(save_path: Optional[str] = None,
                csv_path_c: Optional[str] = None,
                csv_path_d: Optional[str] = None) -> plt.Figure:
    """
    Generate the lower block of Figure 3, panels (b)-(d), laid out in the
    order the text cites them:
        (b) Ensemble test: reported modulus vs drainage proxy (tall, left).
        (c) Geometry scaling: tau ~ L_eff^2 (top right).
        (d) Permeability scaling: tau_drain ~ 1/k (bottom right).
    csv_path_c receives the panel (c) data and csv_path_d the panel (d)
    data.
    """
    L_values = np.logspace(np.log10(15e-6), np.log10(600e-6), 7)   # 15-600 um: 1.6 decades (>= 1.5 needed for a slope verdict)
    k_geo, M_c_geo, eta_geo = 1e-15, 1e4, 1e-3
    D_p_geo = poroelastic_diffusivity(k_geo, M_c_geo, eta_geo)

    tau_num = []
    tau_exact = []
    for L in L_values:
        tau_oo = drainage_time(L, D_p_geo, GF_OPEN_OPEN)
        t_max = 20.0 * tau_oo
        t_grid = log_time_grid(t_max, n_t=600, t_min=tau_oo * 1e-5)
        sol = PoroelasticSolver1D(
            L=L, n_x=201, k=k_geo, M_c=M_c_geo, eta=eta_geo,
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

    L_perm, M_c_perm, eta_perm = 100e-6, 1e4, 1e-3
    k_values = np.logspace(-17, -14, 45)
    tau_values = np.array([
        drainage_time(L_perm, poroelastic_diffusivity(k, M_c_perm, eta_perm),
                      GF_OPEN_SEALED)
        for k in k_values
    ])
    k_low_case, k_high_case = 1e-17, 1e-14
    tau_low_case = drainage_time(
        L_perm, poroelastic_diffusivity(k_low_case, M_c_perm, eta_perm), GF_OPEN_SEALED
    )
    tau_high_case = drainage_time(
        L_perm, poroelastic_diffusivity(k_high_case, M_c_perm, eta_perm), GF_OPEN_SEALED
    )

    # 2-column GridSpec layout in reading order: left [(b) tall], right [(c) over (d)]
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(10.5, 5.6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[0.96, 1.04], height_ratios=[1.0, 1.0],
                           hspace=0.55, wspace=0.18, left=0.07, right=0.97, top=0.93, bottom=0.09)

    # ---------------- Panel (c): geometry scaling (top right) ----------------
    ax_geo = fig.add_subplot(gs[0, 1])
    ax_geo.loglog(L_values * 1e6, tau_num, "o-", lw=2.2, ms=7,
                color="#D62728", label=fr"Numerical (slope={slope_num:.2f})")
    ax_geo.loglog(L_values * 1e6, tau_exact, "k+--", lw=1.8, ms=8,
                label=fr"Exact series (slope={slope_ex:.2f})")
    ax_geo.loglog(L_values * 1e6, np.full_like(L_values, tau_visco),
                "s--", lw=2.0, ms=6.0, color="steelblue",
                label=r"Viscoelastic null: $\tau$ = const")
    ax_geo.set_xlabel(r"$L_{\mathrm{eff}}$ ($\mathrm{\mu}$m)",
                    fontsize=11.5, fontweight="bold")
    ax_geo.set_ylabel(r"$\tau_{1/e}$ (s)", fontsize=11.5, fontweight="bold")
    ax_geo.set_title(r"(c) Geometry scaling: $\tau\propto L_{\mathrm{eff}}^2$",
                   fontsize=14.0, pad=8)
    ax_geo.legend(fontsize=8.5, loc="upper left", framealpha=0.92)
    ax_geo.grid(alpha=0.3, which="both")

    # ---------------- Panel (d): permeability scaling (bottom right) ----------------
    ax_perm = fig.add_subplot(gs[1, 1])
    ax_perm.loglog(k_values, tau_values, "-", lw=2.2, color="0.35",
                label=r"Theory: $\tau_{\mathrm{drain}}\propto1/k$")
    ax_perm.scatter([k_low_case], [tau_low_case], s=110, color="#FF7F0E",
                 zorder=6, label=r"Low matrix perm. ($10^{-17}$ m$^2$)")
    ax_perm.scatter([k_high_case], [tau_high_case], s=110, marker="v",
                 color="#1F77B4", zorder=6, label=r"High matrix perm. ($10^{-14}$ m$^2$)")

    ax_perm.legend(fontsize=8.2, loc="lower left", framealpha=0.92)

    ax_perm.annotate("matrix permeability\nincrease",
                  xy=(k_high_case * 0.75, tau_high_case * 2.5),
                  xytext=(1.0e-16, tau_low_case * 0.10),
                  arrowprops=dict(arrowstyle="->", lw=1.8, color="#2B83BA"),
                  fontsize=8.5, fontweight="bold", color="#1F77B4")

    ax_perm.set_xlabel(r"Biofilm matrix permeability $k$ (m$^2$)",
                    fontsize=11.5, fontweight="bold")
    ax_perm.set_ylabel(r"$\tau_{\mathrm{drain}}$ (s)",
                    fontsize=11.5, fontweight="bold")
    ax_perm.set_title(r"(d) Permeability scaling: $\tau_{\mathrm{drain}}\propto 1/k$",
                   fontsize=14.0, pad=8)
    ax_perm.grid(alpha=0.3, which="both")

    # ---------------- Panel (b): ensemble test (tall left) ----------------
    ax_ens = fig.add_subplot(gs[:, 0])
    bio_pts = [p for p in LITERATURE_DATA
               if p.get("E_Pa") and p["cat"] != "cell reference"]
    cell_pts = [p for p in LITERATURE_DATA
                if p.get("E_Pa") and p["cat"] == "cell reference"]
    na_n = len([p for p in LITERATURE_DATA if not p.get("E_Pa")])

    xs = np.array([p["t_s"] / (p["L_um"] ** 2) for p in bio_pts])
    Es = np.array([p["E_Pa"] for p in bio_pts], dtype=float)
    los = np.array([p.get("E_lo") or p["E_Pa"] for p in bio_pts],
                   dtype=float)
    his = np.array([p.get("E_hi") or p["E_Pa"] for p in bio_pts],
                   dtype=float)

    xg = np.logspace(np.log10(xs.min()) - 0.5,
                     np.log10(xs.max()) + 0.5, 60)
    guide = 1.0e5 * (xg / 1.0e-4) ** (-1.0)
    ax_ens.loglog(xg, guide, "--", lw=1.5, color="0.5", zorder=2,
                label="drainage guide (not a fit)")

    ax_ens.errorbar(xs, Es,
                  yerr=np.vstack([Es - los, his - Es]),
                  fmt="none", ecolor="0.45", elinewidth=1.4,
                  capsize=4, zorder=4)

    # Dark labels with a thin white halo
    ens_label_offsets = {
        "Stoodley 99": (8, -15),      # Bottom-right of marker, clear of Klapper 02
        "Klapper 02": (10, 4),        # Top-right of marker
        "Kundukad 16": (-10, 8),      # Top-left of marker (ha="right", clear of stats box)
        "Towler 03": (10, -6),        # Bottom-right of marker (ha="left")
        "Lau 09": (-12, 6),           # Top-left of marker (ha="right")
    }

    for p, x, y in zip(bio_pts, xs, Es):
        ax_ens.plot(x, y, p["marker"], ms=10.5, color=p["color"],
                  mec="k", mew=1.1, zorder=6)
        lbl_off = ens_label_offsets.get(p["label"], (6, 6))
        ha_lbl = "right" if lbl_off[0] < 0 else "left"
        ax_ens.annotate(p["label"], (x, y), xytext=lbl_off,
                      textcoords="offset points", fontsize=9.5,
                      fontweight="bold", color="#111111", ha=ha_lbl,
                      path_effects=[pe.withStroke(linewidth=2.8,
                                                  foreground="white")])

    for p in cell_pts:
        xc = p["t_s"] / (p["L_um"] ** 2)
        yc = p["E_Pa"]
        ax_ens.plot(xc, yc, p["marker"],
                  ms=14, color="0.4", mec="k", mew=1.1, zorder=6)
        ax_ens.annotate(p["label"] + "\n(cell ref)",
                      (xc, yc),
                      xytext=(-12, 6), textcoords="offset points",
                      fontsize=8.5, color="#222222", fontweight="bold", ha="right", va="bottom",
                      path_effects=[pe.withStroke(linewidth=2.8,
                                                  foreground="white")])

    rho, pval = spearmanr(np.log10(xs), np.log10(Es))
    rho_ex, p_exact = spearman_exact_permutation(np.log10(xs), np.log10(Es))
    assert abs(rho_ex - rho) < 1e-9, "rank correlation mismatch"
    print(f"  [Fig. 3(b) check] Spearman rho={rho:.3f} exact two-sided P={p_exact:.3f} "
          f"(t-approx p={pval:.3f}) n={len(Es)} (biofilm points; cell ref excluded; "
          f"{na_n} markers without modulus)")

    # Descriptive statistic only (n = 5); the exact permutation P is printed to the console.
    stats_txt = (f"Spearman $\\rho_S$ = {rho:+.2f} ($n$ = {len(Es)}, descriptive)\n"
                 f"{na_n} literature markers lack reported modulus")
    ax_ens.text(0.035, 0.035, stats_txt, transform=ax_ens.transAxes,
              fontsize=8.2, va="bottom", ha="left",
              bbox=dict(boxstyle="round,pad=0.35", fc="#F8F9FA",
                        ec="0.7", alpha=0.96))

    ax_ens.set_xlabel(r"$t_{\mathrm{obs}}\,/\,L_{\mathrm{eff}}^2$ "
                    r"(s $\mathrm{\mu m}^{-2}$)",
                    fontsize=11.5, fontweight="bold")
    ax_ens.set_ylabel("Reported modulus (Pa)",
                    fontsize=11.5, fontweight="bold")
    ax_ens.set_title("(b) Ensemble test: modulus vs. drainage proxy",
                   fontsize=14.0, pad=8)
    ax_ens.legend(fontsize=8.2, loc="upper right", framealpha=0.92)
    ax_ens.grid(alpha=0.3, which="both")
    # Right limit keeps the "Towler 03" label inside panel (b). Display only.
    ax_ens.set_xlim(4.0e-6, 3.0)
    # Lower limit 0.1 Pa shows the full Towler 03 range (G = 0.2-24 Pa).
    # Display only.
    ax_ens.set_ylim(0.1, 6.0e6)
    ax_ens.text(0.98, 0.90, r"larger $N_d$ (more drained) $\rightarrow$",
              transform=ax_ens.transAxes, ha="right", fontsize=9.0,
              color="0.4")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Figure saved] {save_path}")

    if csv_path_c:
        save_csv(csv_path_c, ["L_eff_um", "tau_numerical_s", "tau_exact_s"],
                 [L_values * 1e6, tau_num, tau_exact])
    if csv_path_d:
        save_csv(csv_path_d, ["k_m2", "tau_drain_s"],
                 [k_values, tau_values])

    return fig


# ======================================================================
# MANDEL-CRYER PROBLEM
# ======================================================================
#
# Specified geometry (Mandel 1953; analytical series after Cheng and
# Detournay 1988, form as reproduced in the GEOS benchmark docs):
#
#   A saturated poroelastic strip of half-width a is sandwiched between
#   two rigid, frictionless, impermeable platens and loaded by a
#   constant total vertical force. Fluid drains only through the
#   lateral faces x = +-a. The pore pressure is uniform through the
#   thickness and the normalized solution is
#
#   p(x,t)/p0 = SUM_n [ 2 sin(a_n) / (a_n - sin a_n cos a_n) ]
#                 * ( cos(a_n x/a) - cos(a_n) )
#                 * exp( -a_n^2 D_p t / a^2 )
#
#   where a_n are the positive roots of
#   tan(a_n) = [(1 - nu)/(nu_u - nu)] a_n
#   and p0 is the instantaneous (Skempton) undrained pore pressure.
#
#   Because the outer region drains first and the rigid platen
#   redistributes the constant total load, the CENTER pressure rises
#   ABOVE p0 before decaying (Mandel-Cryer overshoot). The one-
#   dimensional open-top/sealed-base column used in Figs. 2-3 cannot
#   produce this non-monotone interior transient.
#
#   Convention mapping: the Cheng & Detournay hydraulic diffusivity
#   c_v is this script's poroelastic diffusivity D_p = k*M_c/eta
#   (same Biot-type consolidated diffusivity, written with the
#   constrained modulus). Parameter cross-check: M_c = 10^4 Pa at
#   nu = 0.3 corresponds to E ~ 7.4 kPa and G ~ 2.9 kPa, consistent
#   with the G = 2.86 kPa value used in the Cheng & Detournay
#   benchmark at nu = 0.3.
#
# Biofilm reading of the geometry: a laterally finite biofilm patch on
# an impermeable substrate, occluded on top by an impermeable cover or
# loading platen, with pressure release only at the patch edge (or the
# analogous drained sphere for plugs and aggregates, Cryer 1963).


def mandel_eigenvalues(c_coef: float, n_modes: int = 80) -> np.ndarray:
    """Positive roots of tan(a) = c_coef * a, one per (n*pi, n*pi+pi/2)."""
    roots = []
    for n_ in range(n_modes):
        lo = n_ * math.pi + 1e-9
        hi = n_ * math.pi + math.pi / 2.0 - 1e-9
        try:
            roots.append(brentq(
                lambda ang: math.tan(ang) - c_coef * ang, lo, hi,
                xtol=1e-13
            ))
        except ValueError:
            break
    return np.asarray(roots, dtype=float)


def mandel_pressure(xxa: np.ndarray,
                    ts: np.ndarray,
                    c_coef: float,
                    n_modes: int = 80) -> np.ndarray:
    """
    Normalized Mandel pore pressure p(x, t) / p0.

    Parameters
    ----------
    xxa : x/a in [0, 1] (the solution is even in x).
    ts  : dimensionless time t * D_p / a**2.
    c_coef : (1 - nu) / (nu_u - nu).

    Returns
    -------
    Array of shape (len(ts), len(xxa)).
    """
    xxa = np.atleast_1d(np.asarray(xxa, dtype=float))
    ts = np.atleast_1d(np.asarray(ts, dtype=float))
    alphas = mandel_eigenvalues(c_coef, n_modes=n_modes)
    P = np.zeros((ts.size, xxa.size))
    for a_n in alphas:
        A = 2.0 * math.sin(a_n) / (a_n - math.sin(a_n) * math.cos(a_n))
        P += (A * (np.cos(a_n * xxa) - math.cos(a_n))[None, :]
              * np.exp(-(a_n ** 2) * ts)[:, None])
    return P


def mandel_peak_overshoot(nu: float = 0.3, nu_u: float = 0.5,
                          n_modes: int = 80) -> Tuple[float, float]:
    """Peak center pressure p(0,t)/p0 and its dimensionless time t*."""
    c_coef = (1.0 - nu) / (nu_u - nu)
    ts = np.logspace(-4, 1, 2000)
    pc = mandel_pressure(np.array([0.0]), ts, c_coef, n_modes)[:, 0]
    i = int(np.argmax(pc))
    return float(pc[i]), float(ts[i])


def figure_4_mandel(save_path: Optional[str] = None,
                    csv_path: Optional[str] = None,
                    a_m: float = 100e-6,
                    k: float = 1e-15,
                    M_c: float = 1e4,
                    eta: float = 1e-3,
                    nu: float = 0.3,
                    nu_u: float = 0.5,
                    n_modes: int = 80) -> plt.Figure:
    """
    Generate Figure 4: Mandel-Cryer overshoot vs 1D monotone drainage.
    """
    from matplotlib.patches import Rectangle

    D_p = poroelastic_diffusivity(k, M_c, eta)
    t_cross = a_m ** 2 / D_p            # [s]; exactly 1 s for a=100 um
    c_coef = (1.0 - nu) / (nu_u - nu)

    # Verification checks
    ts_probe = np.logspace(-4, 1, 1500)
    pc_probe = mandel_pressure(np.array([0.0]), ts_probe, c_coef, n_modes)[:, 0]
    i_pk = int(np.argmax(pc_probe))
    peak_val = pc_probe[i_pk]
    peak_t = ts_probe[i_pk]
    print(f"  [Fig.4 check] peak p(0)/p0 = {peak_val:.4f} (+{100.0 * (peak_val - 1.0):.1f}%) at t* = {peak_t:.4f}")
    assert peak_val > 1.002, "Mandel-Cryer overshoot missing!"

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 4.8))
    fig.subplots_adjust(wspace=0.28, left=0.06, right=0.96, top=0.88, bottom=0.14)

    TITLE_FS = 14.0
    LABEL_FS = 11.5

    # ---------------- Panel (a): Specified Geometry ----------------
    ax = axes[0]

    # Platen and substrate are both impermeable and share one hatch style
    # Substrate (bottom sealed base)
    ax.add_patch(Rectangle((-1.20, -0.26), 2.40, 0.26,
                           fc="#E2E8F0", ec="#334155", lw=1.5, hatch="////", zorder=3))
    # Biofilm patch (middle saturated poroelastic medium)
    ax.add_patch(Rectangle((-1.0, 0.0), 2.0, 1.0,
                           fc="#DCEBF7", ec="#1E3A8A", lw=1.6, zorder=2))
    # Rigid platen (top sealed platen)
    ax.add_patch(Rectangle((-1.20, 1.0), 2.40, 0.26,
                           fc="#E2E8F0", ec="#334155", lw=1.5, hatch="////", zorder=3))

    # Applied compressive force
    ax.annotate("", xy=(0.0, 1.28), xytext=(0.0, 1.76),
                arrowprops=dict(arrowstyle="-|>", lw=2.4, color="#0F172A", mutation_scale=18))
    ax.text(0.10, 1.62, r"Total force $F$", fontsize=11, ha="left", va="center",
            fontweight="bold", color="#0F172A")

    # Platen label
    ax.text(0.0, 1.13, "Rigid impermeable platen (sealed top)",
            fontsize=8.5, ha="center", va="center", fontweight="bold", color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#475569", lw=0.8, alpha=0.96),
            zorder=6)

    # Substrate label
    ax.text(0.0, -0.13, "Impermeable substrate (sealed base)",
            fontsize=8.5, ha="center", va="center", fontweight="bold", color="#1E293B",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#475569", lw=0.8, alpha=0.96),
            zorder=6)

    # Lateral drainage arrows (permeable side boundaries at x = +-a)
    ax.annotate("", xy=(1.50, 0.5), xytext=(1.04, 0.5),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#0284C7"))
    ax.annotate("", xy=(-1.50, 0.5), xytext=(-1.04, 0.5),
                arrowprops=dict(arrowstyle="->", lw=2.0, color="#0284C7"))
    ax.text(1.52, 0.62, "drainage\nrelease", fontsize=8.5, ha="center",
            va="bottom", color="#0284C7", fontweight="bold")
    ax.text(-1.52, 0.62, "drainage\nrelease", fontsize=8.5, ha="center",
            va="bottom", color="#0284C7", fontweight="bold")

    # Biofilm patch interior details
    ax.text(0.0, 0.78, "Biofilm patch", fontsize=11.5,
        ha="center", va="center", fontweight="bold", color="#0F172A")
    ax.plot([0.0], [0.5], "o", ms=7, color="#B2182B", zorder=5, mec="k", mew=0.8)
    ax.text(0.0, 0.38, r"probed center $p(0,t)$", fontsize=9.2,
            ha="center", va="top", color="#B2182B", fontweight="bold")
    ax.text(0.0, 0.16, r"half-width $a = 100\,\mathrm{\mu m}$", fontsize=9.2,
            ha="center", va="center", color="#334155")

    ax.set_xlim(-1.75, 1.75)
    ax.set_ylim(-0.45, 1.95)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("(a) Mandel geometry (specified)",
                 fontsize=TITLE_FS, pad=8)

    # ---------------- Panel (b): Overshoot vs Monotone 1D ----------------
    ax = axes[1]
    t_s = np.logspace(-4, 1, 1300) * t_cross       # physical seconds
    ts_n = t_s / t_cross
    p_center = mandel_pressure(np.array([0.0]), ts_n, c_coef, n_modes)[:, 0]
    xg = np.linspace(0.0, 1.0, 2001)
    try:
        p_mean = np.trapezoid(
            mandel_pressure(xg, ts_n, c_coef, n_modes), xg, axis=1
        ) / 1.0
    except AttributeError:
        p_mean = np.trapz(
            mandel_pressure(xg, ts_n, c_coef, n_modes), xg, axis=1
        ) / 1.0
    tau1_1d = drainage_time(a_m, D_p, GF_OPEN_SEALED)
    p_1d = terzaghi_exact(t_s, tau1_1d, n_modes=30)

    ax.semilogx(t_s, p_center, lw=2.6, color="#B2182B",
                label=r"Mandel center $p(0,t)/p_0$")
    ax.semilogx(t_s, p_mean, lw=2.0, ls="--", color="#EF8A62",
                label=r"Mandel cross-section mean")
    ax.semilogx(t_s, p_1d, lw=2.0, ls=":", color="#1F78B4",
                label=r"1D open-top/sealed-base mean")
    ax.axhline(1.0, color="0.45", lw=1.2, ls="-.",
               label=r"initial undrained $p_0$")
    ax.axhline(0.0, color="0.8", lw=0.8)

    # Parameter box (top left)
    param_txt = (rf"$a = {a_m*1e6:.0f}\,\mathrm{{\mu m}},\; D_p = 10^{{{int(np.round(np.log10(D_p)))}}}\,\mathrm{{m^2\,s^{{-1}}}}$" "\n"
                 rf"$a^2/D_p = {t_cross:g}\,$s")
    ax.text(0.03, 0.95, param_txt, transform=ax.transAxes,
            fontsize=8.0, ha="left", va="top", color="#475569",
            bbox=dict(boxstyle="round,pad=0.25", fc="#F8FAFC", ec="#CBD5E1", lw=0.8, alpha=0.92))

    # Overshoot callout, right of the peak
    i_b = int(np.argmax(p_center))
    t_peak_val = t_s[i_b]
    p_peak_val = p_center[i_b]
    overshoot_pct = 100.0 * (p_peak_val - 1.0)

    ax.annotate(
        f"Mandel–Cryer overshoot\n" r"$p(0,t) > p_0$ " f"(+{overshoot_pct:.1f}%)",
        xy=(t_peak_val, p_peak_val),
        xytext=(0.45, 1.18),
        fontsize=8.8, ha="center", va="center", fontweight="bold",
        arrowprops=dict(arrowstyle="->", lw=1.8, color="#B2182B"),
        color="#990000",
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF5F5", ec="#B2182B", lw=1.1, alpha=0.96),
        zorder=7
    )

    ax.set_xlim(1e-4 * t_cross, 1e1 * t_cross)
    ax.set_ylim(-0.04, 1.32)
    ax.set_xlabel(r"Time $t$ (s)", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(r"$p\,/\,p_0$", fontsize=LABEL_FS, fontweight="bold")
    ax.set_title("(b) Center overshoot needs this geometry",
                 fontsize=TITLE_FS, pad=8)
    ax.legend(fontsize=8.2, loc="lower left", framealpha=0.92)
    ax.grid(alpha=0.3, which="both")

    # ---------------- Panel (c): Profiles Across Half-Width ----------------
    ax = axes[2]
    xs = np.linspace(-1.0, 1.0, 401)
    t_cases = [0.01, 0.04, 0.064, 0.25, 1.0]
    cmap = plt.cm.plasma(np.linspace(0.08, 0.88, len(t_cases)))
    for tv, col in zip(t_cases, cmap):
        prof = mandel_pressure(xs, np.array([tv]), c_coef, n_modes)[0]
        ax.plot(xs, prof, lw=2.1, color=col,
                label=rf"$t\,D_p/a^2={tv:g}$")
    ax.axhline(1.0, color="0.45", lw=1.2, ls="-.",
               label=r"initial $p_0$")
    ax.set_xlim(-1.0, 1.0)
    ax.set_ylim(-0.04, 1.25)
    ax.set_xlabel(r"$x\,/\,a$", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(r"$p\,/\,p_0$", fontsize=LABEL_FS, fontweight="bold")
    ax.set_title("(c) Edge drains, interior rises above $p_0$",
                 fontsize=TITLE_FS, pad=8)

    # Legend in the open space between the profiles
    ax.legend(fontsize=8.0, loc="center", bbox_to_anchor=(0.5, 0.44),
              framealpha=0.92, facecolor="white", edgecolor="#CBD5E1",
              handlelength=1.6, labelspacing=0.28, borderpad=0.35)
    ax.grid(alpha=0.3, which="both")

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Figure saved] {save_path}")

    if csv_path:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["t_s", "t_dimless", "p_center_norm", "p_mean_norm", "p_1d_norm"])
            for t_val, ts_n_val, pc_val, pm_val, p1_val in zip(t_s, ts_n, p_center, p_mean, p_1d):
                writer.writerow([t_val, ts_n_val, pc_val, pm_val, p1_val])
        print(f"  [CSV saved] {csv_path}")

    return fig


# ======================================================================
# MODEL RECOVERY AND IDENTIFIABILITY
# ======================================================================
#
# Synthetic discrimination study (Sec. IV.A and Table II of the article):
#   Panel (a): a single poroelastic relaxation curve is reproduced by a
#            stretched-exponential viscoelastic fit (near-perfect R^2),
#            so one curve cannot identify the mechanism.
#   Panel (b): a length series recovers the slope of the generating
#            model (poroelastic -> ~2, viscoelastic -> ~0, mixed ->
#            intermediate), which is why the verdict rests on the
#            slope, not on any single curve.
#   Panel (c): two (k, M_c) pairs with the same product k*M_c give
#            mechanically identical relaxation but 10x different
#            pressure-driven conductance, so relaxation alone identifies
#            only the composite and conductance data are required to
#            split k from M_c.

def _ve_kernel(t, tau_v, gamma):
    return np.exp(-(np.asarray(t, dtype=float) / tau_v) ** gamma)


def tau_1_over_e(t, g):
    """1/e crossing of a decreasing curve g(t), log-interpolated."""
    t = np.asarray(t, dtype=float)
    g = np.asarray(g, dtype=float)
    thr = 1.0 / math.e
    below = np.flatnonzero(g <= thr)
    if below.size == 0:
        return float("nan")
    i = int(below[0])
    if i == 0:
        return float(t[0])
    y0, y1 = g[i - 1], g[i]
    x0, x1 = math.log10(t[i - 1]), math.log10(t[i])
    if y0 == y1:
        return float(t[i])
    frac = (y0 - thr) / (y0 - y1)
    return float(10.0 ** (x0 + frac * (x1 - x0)))


def _fit_ve(t, g):
    """Least-squares stretched-exponential fit; returns tau_v, gamma, R^2."""
    def resid(p):
        return _ve_kernel(t, math.exp(p[0]), p[1]) - g
    tau0 = tau_1_over_e(t, g)
    r0 = [math.log(max(tau0, t[0] * 1.01)), 0.6]
    res = least_squares(
        resid, r0,
        bounds=([math.log(t[0]), 0.2], [math.log(t[-1]), 1.5])
    )
    tau_v, gamma = math.exp(res.x[0]), res.x[1]
    pred = _ve_kernel(t, tau_v, gamma)
    ss_res = float(np.sum((g - pred) ** 2))
    ss_tot = float(np.sum((g - np.mean(g)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return tau_v, gamma, r2


def figure_5_identifiability(save_path: Optional[str] = None,
                             csv_path: Optional[str] = None,
                             D_p: float = 1e-8) -> plt.Figure:
    """
    Generate Figure 5: Model recovery and structural identifiability.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 5.1))
    fig.subplots_adjust(wspace=0.28, left=0.06, right=0.96, top=0.88, bottom=0.14)

    TITLE_FS = 14.0
    LABEL_FS = 11.5
    summary = {"panel": [], "key": [], "value": []}

    # ---------------- Panel (a): One curve, two mechanisms ----------------
    ax = axes[0]
    t_a = np.logspace(-3, 2, 400)
    g_pe = terzaghi_exact(t_a, 1.0, n_modes=40)
    tau_v, gamma_v, r2 = _fit_ve(t_a, g_pe)
    g_ve = _ve_kernel(t_a, tau_v, gamma_v)
    residual_pct = (g_pe - g_ve) * 100.0

    # Primary relaxation curves
    l1, = ax.semilogx(t_a, g_pe, lw=2.6, color="#B2182B",
                      label="Poroelastic truth (1D column)", zorder=3)
    l2, = ax.semilogx(t_a, g_ve, lw=2.0, ls="--", color="#1F78B4",
                      label=rf"Viscoelastic fit ($\tau_v={tau_v:.3f}\,$s, $\gamma={gamma_v:.3f}$)",
                      zorder=4)

    ax.set_xlim(1e-3, 1e2)
    ax.set_ylim(-0.04, 1.15)
    ax.set_xlabel(r"Time $t$ (s)", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel("Remaining fraction $p(t)/p_0$", fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(rf"(a) One curve fits both ($R^2 = {r2:.4f}$)",
                 fontsize=TITLE_FS, pad=8)
    ax.grid(alpha=0.3, which="both")

    # Secondary axis: residual, scaled to [-5%, +5%]
    ax_res = ax.twinx()
    l3, = ax_res.semilogx(t_a, residual_pct, lw=1.5, ls="-.", color="#64748B",
                          label=r"Residual $(y_{\mathrm{PE}} - y_{\mathrm{VE}})$", zorder=2)
    ax_res.axhline(0.0, color="#94A3B8", lw=0.8, ls=":")
    ax_res.set_ylim(-5.0, 5.0)
    ax_res.set_ylabel("Residual (% points)", fontsize=10.0, color="#64748B", fontweight="bold")
    ax_res.tick_params(axis="y", labelcolor="#64748B", labelsize=8.5)

    # Combined legend (lower left)
    ax.legend([l1, l2, l3], [l1.get_label(), l2.get_label(), l3.get_label()],
              fontsize=8.0, loc="lower left", framealpha=0.92)

    summary["panel"].append("A")
    summary["key"].append("ve_fit_r2_to_pe_truth")
    summary["value"].append(f"{r2:.5f}")
    print(f"  [Fig.5 check] VE fit to PE truth: R^2={r2:.4f}, tau_v={tau_v:.3f} s, gamma={gamma_v:.3f}")

    # ---------------- Panel (b): Length-series model recovery ----------------
    ax = axes[1]
    Ls_um = np.logspace(math.log10(15.0), math.log10(600.0), 7)   # 15-600 um: 1.6 decades (>= 1.5 needed for a slope verdict)
    t_b = np.logspace(-4, 2.5, 900)
    tau_v_true, gamma_true = 1.0, 0.6

    tau_pe, tau_ve, tau_pve = [], [], []
    for L_um in Ls_um:
        L = L_um * 1e-6
        tau_c = drainage_time(L, D_p, GF_OPEN_OPEN)
        g = terzaghi_exact_open_open(t_b, tau_c, n_modes=40)
        tau_pe.append(tau_1_over_e(t_b, g))
        tau_ve.append(tau_1_over_e(t_b, _ve_kernel(t_b, tau_v_true, gamma_true)))
        g_mix = g * _ve_kernel(t_b, tau_v_true, gamma_true)
        tau_pve.append(tau_1_over_e(t_b, g_mix))

    tau_pe = np.array(tau_pe)
    tau_ve = np.array(tau_ve)
    tau_pve = np.array(tau_pve)

    def _slope(x, y):
        return float(np.polyfit(np.log10(x), np.log10(y), 1)[0])

    s_pe = _slope(Ls_um, tau_pe)
    s_ve = _slope(Ls_um, tau_ve)
    s_pve = _slope(Ls_um, tau_pve)
    print(f"  [Fig.5 check] recovered slopes: PE={s_pe:.2f} VE={s_ve:.2f} PVE={s_pve:.2f}")

    for name, val in (("slope_pe", s_pe), ("slope_ve", s_ve), ("slope_pve", s_pve)):
        summary["panel"].append("B")
        summary["key"].append(name)
        summary["value"].append(f"{val:.4f}")

    ax.loglog(Ls_um, tau_pe, "o-", lw=2.2, ms=8, color="#B2182B",
              label=fr"Poroelastic truth (slope {s_pe:.2f})")
    ax.loglog(Ls_um, tau_ve, "s-", lw=2.0, ms=7, color="#1F78B4",
              label=fr"Viscoelastic truth (slope {s_ve:.2f})")
    ax.loglog(Ls_um, tau_pve, "^-", lw=2.0, ms=7, color="#7B3294",
              label=fr"Mixed poroviscoelastic (slope {s_pve:.2f})")

    ax.set_xlabel(r"$L_{\mathrm{eff}}$ ($\mathrm{\mu}$m)", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel(r"Extracted $\tau_{1/e}$ (s)", fontsize=LABEL_FS, fontweight="bold")
    ax.set_title("(b) Length series recovers the model",
                 fontsize=TITLE_FS, pad=8)

    # Legend (lower right)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92)
    ax.grid(alpha=0.3, which="both")

    # ---------------- Panel (c): Invariant Relaxation vs Conductance ----------------
    ax = axes[2]
    L_c = 100e-6
    pairs = [
        (1e-16, 1e5, "#1F78B4", r"$(k, M_c) = (10^{-16}\,\mathrm{m^2},\,10^{5}\,\mathrm{Pa})$"),
        (1e-15, 1e4, "#E31A1C", r"$(k, M_c) = (10^{-15}\,\mathrm{m^2},\,10^{4}\,\mathrm{Pa})$")
    ]
    curves = []
    for k_p, M_p, col, lab in pairs:
        D_p_p = poroelastic_diffusivity(k_p, M_p, 1e-3)
        tau_p = drainage_time(L_c, D_p_p, GF_OPEN_SEALED)
        g_p = terzaghi_exact(t_b, tau_p, n_modes=40)
        curves.append(g_p)
        ax.semilogx(t_b, g_p, lw=2.4, color=col, label=lab)

    max_diff = float(np.max(np.abs(curves[0] - curves[1])))
    cond_ratio = pairs[1][0] / pairs[0][0]
    print(f"  [Fig.5 check] identical-relaxation max|dp|={max_diff:.2e}; conductance ratio={cond_ratio:.0f}x")

    summary["panel"].append("C")
    summary["key"].append("max_curve_diff")
    summary["value"].append(f"{max_diff:.3e}")
    summary["panel"].append("C")
    summary["key"].append("conductance_ratio")
    summary["value"].append(f"{cond_ratio:.1f}")

    ax.set_xlim(1e-3, 1e2)
    ax.set_ylim(-0.04, 1.15)
    ax.set_xlabel(r"Time $t$ (s)", fontsize=LABEL_FS, fontweight="bold")
    ax.set_ylabel("Remaining fraction $p(t)/p_0$", fontsize=LABEL_FS, fontweight="bold")
    ax.set_title(r"(c) Same curves, $10\times$ conductance",
                 fontsize=TITLE_FS, pad=8)

    # Legend at lower-left
    ax.legend(fontsize=8.0, loc="lower left", framealpha=0.92)
    ax.grid(alpha=0.3, which="both")

    # Info box (top right), clear of the early-time curves
    info_txt = (
        rf"Identical relaxation ($|\Delta p| \sim 10^{{{int(np.round(np.log10(max_diff)))}}}$)" "\n"
        rf"$G_h$ differs by {cond_ratio:.0f}$\times$ $\rightarrow$ conductance needed"
    )
    ax.text(0.97, 0.95, info_txt, transform=ax.transAxes,
            fontsize=8.5, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.35", fc="#FFFBEB", ec="#F59E0B", lw=0.9, alpha=0.96))

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Figure saved] {save_path}")

    if csv_path:
        save_csv(csv_path, ["panel", "key", "value"],
                 [summary["panel"], summary["key"], summary["value"]])

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
    ax.set_xlabel(r"$t/\tau_1$", fontsize=13, fontweight="bold")
    ax.set_ylabel(r"$\langle p\rangle/p_0$", fontsize=13, fontweight="bold")
    ax.set_title("(a) Numerical vs. exact Terzaghi",
                 fontsize=16, fontweight="normal")
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
    ax.set_xlabel(r"$t/\tau_1$", fontsize=13, fontweight="bold")
    ax.set_ylabel("Relative error", fontsize=13, fontweight="bold")
    ax.set_title("(b) Relative error vs. time",
                 fontsize=16, fontweight="normal")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  [Figure saved] {save_path}")

    return fig


def plausibility_check_cells(verbose: bool = True) -> Dict:
    """Order-of-magnitude calibration against Moeendarbary et al. (2013)."""
    L, k, M_c, eta = 5e-6, 1e-17, 1e3, 1e-3
    D_p = poroelastic_diffusivity(k, M_c, eta)
    tau1 = drainage_time(L, D_p, GF_OPEN_SEALED)
    tau_1e = terzaghi_1_over_e_crossing(tau1, bc="open-sealed")

    if verbose:
        print("[Plausibility check] Moeendarbary et al. (2013) HeLa cell check")
        print("  L=5 um, k=1e-17 m^2, M_c=1 kPa")
        print(f"  D_p = {D_p:.3g} m^2/s")
        print(f"  tau_1/e = {tau_1e:.2f} s")
        print("  Observed cell relaxation range: ~0.5-5 s")
        print("  Correct order of magnitude; contact geometry sets prefactor.\n")

    return {"D_p": D_p, "tau1": tau1, "tau_1e": tau_1e}


def convergence_check_channels(verbose: bool = True) -> Dict:
    """Grid-convergence check for hard and leaky interior sinks."""
    L = 200e-6
    spacing_um = 25
    k, M_c, eta = 1e-15, 1e4, 1e-3
    D_p = poroelastic_diffusivity(k, M_c, eta)
    nx_values = [201, 401, 801]

    # Hard sink
    taus_hard = []
    for nx in nx_values:
        res = simulate_with_channels(
            L=L, spacings_um=[spacing_um], k=k, M_c=M_c, eta=eta,
            n_x=nx, n_t=900, sink_beta=float("inf")
        )
        taus_hard.append(res[spacing_um]["tau"])
    rel_hard = abs(taus_hard[-1] - taus_hard[-2]) / taus_hard[-1]

    # Leaky sink at Bi_sink = 1
    s_m = spacing_um * 1e-6
    beta_bi1 = D_p / s_m
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
    print(f"biofilm_poroelastic_sim.py v{__version__}")
    print("Generating Figure 2, Figure 3, Figure 4, Figure 5, and Supplementary Figure S1")
    print("Each numbered figure is saved as a single PNG + TIFF file")
    print("Figures 1 and 6 are made by the two companion figure scripts")
    print("#" * 80 + "\n")

    plausibility_check_cells(verbose=True)

    # ------------------------------------------------------------------
    # Supplementary validation figure (line art, 1000 dpi)
    # ------------------------------------------------------------------
    print("[Figure S1] Generating validation figure...")
    validation_fig = figure_validation(save_path=None)
    save_figure_formats(
        validation_fig,
        stem="Figure_S1_Validation",
        dpi=LINE_ART_DPI
    )
    plt.close(validation_fig)

    convergence_check_channels(verbose=True)

    regime_placement(
        ProtocolInputs(
            L_eff=50e-6, k_range=(1e-17, 1e-14),
            M_c=1e4, eta=1e-3, t_obs=1.0, boundary_type="sealed"
        ),
        verbose=True
    )

    # ------------------------------------------------------------------
    # Figure 2 (assembled from the (a)-(b) block over the (c)-(d) block)
    # ------------------------------------------------------------------
    print("\n[Figure 2] Generating complete four-panel artwork...")

    channel_results = simulate_with_channels(
        L=200e-6, spacings_um=[None, 100, 50, 25, 10],
        k=1e-15, M_c=1e4, eta=1e-3, n_x=401, n_t=800,
        n_be_init=4, sink_beta=float("inf")
    )
    for s, d in channel_results.items():
        label = "no channels" if s is None else f"s={s} um"
        print(f"  {label:<15s} tau_1/e = {d['tau']:.4g} s")

    fig2_ab = figure_2_channel_demo(
        channel_results,
        save_path=None,
        csv_path=out("Figure_2ab_pressure_curves.csv")
    )
    block2_ab = save_working_block(fig2_ab, "_Figure_2_ab_working.png")
    plt.close(fig2_ab)

    fig2_cd = figure_2_heatmap(
        L=200e-6, spacings_um=[10, 20, 30, 40, 50, 75, 100, 150],
        beta_values=np.logspace(-5, 0, 12),
        k=1e-15, M_c=1e4, eta=1e-3, n_x=301, n_t=500, n_be_init=4,
        save_path=None,
        csv_path=out("Figure_2cd_tau_matrix.csv")
    )
    block2_cd = save_working_block(fig2_cd, "_Figure_2_cd_working.png")
    plt.close(fig2_cd)

    compose_vertical(
        [block2_ab, block2_cd],
        stem="Figure_2",
        target_width_px=FULL_PAGE_WIDTH_PX,
        gap_px=150,  # extra gap between the (a)-(b) and (c)-(d) blocks
        dpi=ARTWORK_DPI,
        remove_blocks=True
    )

    # ------------------------------------------------------------------
    # Figure 3 (assembled from the (a) block over the (b)-(d) block)
    # ------------------------------------------------------------------
    print("\n[Figure 3] Generating complete four-panel artwork...")

    fig3_a = figure_3a(
        save_path=None,
        csv_path=out("Figure_3ab_literature_points.csv")
    )
    block3_a = save_working_block(fig3_a, "_Figure_3_a_working.png")
    plt.close(fig3_a)

    fig3_bcd = figure_3bcd(
        save_path=None,
        csv_path_c=out("Figure_3c_tau_vs_L.csv"),
        csv_path_d=out("Figure_3d_tau_vs_k.csv")
    )
    block3_bcd = save_working_block(fig3_bcd, "_Figure_3_bcd_working.png")
    plt.close(fig3_bcd)

    compose_vertical(
        [block3_a, block3_bcd],
        stem="Figure_3",
        target_width_px=FULL_PAGE_WIDTH_PX,
        gap_px=260,
        dpi=ARTWORK_DPI,
        remove_blocks=True,
        block_margins=[(0, 0), (160, 240)]
    )

    # ------------------------------------------------------------------
    # Figure 4: Mandel-Cryer benchmark
    # ------------------------------------------------------------------
    print("\n[Figure 4] Generating Mandel-Cryer artwork...")

    fig4 = figure_4_mandel(
        save_path=None,
        csv_path=out("Figure_4_Mandel_pressure.csv")
    )
    save_figure_formats(fig4, stem="Figure_4", dpi=ARTWORK_DPI)
    plt.close(fig4)

    # ------------------------------------------------------------------
    # Figure 5: model recovery and identifiability
    # ------------------------------------------------------------------
    print("\n[Figure 5] Generating identifiability artwork...")

    fig5 = figure_5_identifiability(
        save_path=None,
        csv_path=out("Figure_5_model_recovery.csv")
    )
    save_figure_formats(fig5, stem="Figure_5", dpi=ARTWORK_DPI)
    plt.close(fig5)

    print("\n" + "#" * 80)
    print("Simulation complete.")
    print(f"All outputs saved to: {os.path.abspath(OUTPUT_DIR)}")
    print("Numbered artwork files:")
    print("  Figure_2.png / Figure_2.tiff            (combination, 500 dpi)")
    print("  Figure_3.png / Figure_3.tiff            (combination, 500 dpi)")
    print("  Figure_4.png / Figure_4.tiff            (Mandel-Cryer, 500 dpi)")
    print("  Figure_5.png / Figure_5.tiff            (identifiability, 500 dpi)")
    print("  Figure_S1_Validation.png / .tiff        (line art,   1000 dpi)")
    print("Figures 1 and 6: run figure1_drainage_boundary_access.py and "
          "figure6_clinical_active_architecture.py.")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
