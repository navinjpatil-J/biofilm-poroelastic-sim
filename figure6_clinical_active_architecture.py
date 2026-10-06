# -*- coding: utf-8 -*-
"""
figure6_clinical_active_architecture.py
=======================================

Figure 6 of the review article "Interfacial drainage regime as a coordinate
linking biofilm mechanics and transport": clinical drainage-regime placement
and a reversible test of active hydraulic architecture.

Version 3.1.0. Released under the MIT License (see LICENSE).

Panels
------
(a) Clinical biofilm settings on the (L_eff, t_obs) drainage-regime map.
    The background, the N_d = 1 lines, the colormap, the colorbar label,
    and the regime labels use the same parameters as Fig. 3(a) of
    biofilm_poroelastic_sim.py:
        k = 1e-17 (low) to 1e-14 (high) m^2, background at the geometric
        mean k; M_c = 1e4 Pa; eta = 1e-3 Pa s; open-top/sealed-base
        Terzaghi series (20 modes); tau = L_eff^2 / (pi^2/4 * D_p).
    The axes extend to 10 mm and 1e4 s, so the millimeter-scale airway
    mucus plugs cited in Section IV.B fit on the map.
(b) Schematic hydraulic conductance G_h/G_h,0 during acute suppression and
    restoration of matrix biosynthesis (active vs passive architecture).
(c) The corresponding drainage time, tau_drain/tau_0 = G_h,0/G_h, at fixed
    L_eff, eta and M_c. The peak equals G_h,0/G_h,min.

Panels (b) and (c) are schematic signatures, not fitted data. The active
response is causal: it begins only when suppression or restoration
starts, and it follows a two-stage first-order (critically damped) step
response.

How to run
----------
    python figure6_clinical_active_architecture.py

If biofilm_poroelastic_sim.py sits in the same folder, the script first
checks that panel (a) uses the same physics and parameters as Fig. 3(a)
and prints the result; otherwise the check is skipped.

Outputs (in biofilm_sim_outputs/ next to this script)
-----------------------------------------------------
    Figure_6.png                 3,740 px wide, 500 dpi
    Figure_6.tiff                LZW, 3,740 px wide, 500 dpi
    Figure_6_region_bounds.csv   drainage times and N_d range of each region
"""

import csv
import inspect
import math
import os
import re
import sys
from typing import Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import rcParams
from matplotlib.patches import Rectangle
from PIL import Image

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "hatch.linewidth": 0.6,
})

__version__ = "3.1.0"

SCRIPT_DIR = (os.path.dirname(os.path.abspath(__file__))
              if "__file__" in globals() else os.getcwd())
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "biofilm_sim_outputs")

# Artwork policy, identical to the main code: 500 dpi combination artwork,
# 7.48 in final width (3,740 px), 35 px gap between stacked blocks.
ARTWORK_DPI = 500
FINAL_PAGE_WIDTH_IN = 7.48
FULL_PAGE_WIDTH_PX = int(round(FINAL_PAGE_WIDTH_IN * ARTWORK_DPI))  # 3740
PANEL_GAP_PX = 35

# ----------------------------------------------------------------------
# Physics: same functions, constants and parameters as Fig. 3(a)
# ----------------------------------------------------------------------
GF_OPEN_SEALED = math.pi ** 2 / 4.0   # geometry factor, open-top/sealed-base

K_LOW, K_HIGH = 1e-17, 1e-14          # m^2, as in Fig. 3(a)
K_MID = math.sqrt(K_LOW * K_HIGH)     # background permeability
M_C, ETA = 1e4, 1e-3                  # Pa, Pa s, as in Fig. 3(a)
N_MODES = 20

L_LIM_UM = (1.0, 1e4)                 # map extent: 1 um to 10 mm
T_LIM_S = (1e-2, 1e4)                 # map extent: 10 ms to ~2.8 h

# Okabe-Ito palette (color-blind safe)
C_BLUE, C_GREEN, C_ORANGE, C_VERMILION = "#0072B2", "#009E73", "#E69F00", "#D55E00"


def poroelastic_diffusivity(k: float, M_c: float, eta: float) -> float:
    """Poroelastic diffusivity D_p = k M_c / eta [m^2/s] (as in main code)."""
    if k <= 0 or M_c <= 0 or eta <= 0:
        raise ValueError("k, M_c, and eta must be positive.")
    return k * M_c / eta


def drainage_time(L_eff, D_p: float, geometry_factor: float = GF_OPEN_SEALED):
    """Characteristic drainage time tau = L_eff^2 / (geometry_factor D_p)."""
    return np.asarray(L_eff, dtype=float) ** 2 / (geometry_factor * D_p)


def terzaghi_exact(t, tau1, n_modes: int = N_MODES) -> np.ndarray:
    """Exact spatially averaged Terzaghi solution, open-top/sealed-base
    (same series as the main code; vectorized over t and tau1)."""
    t = np.asarray(t, dtype=float)
    y = np.zeros(np.broadcast(t, np.asarray(tau1)).shape, dtype=float)
    for m in range(n_modes):
        n = 2 * m + 1
        y += 8.0 / (n ** 2 * math.pi ** 2) * np.exp(-(n ** 2) * t / tau1)
    return y


# ----------------------------------------------------------------------
# Clinical regions (order-of-magnitude regime bounds, Section IV.B)
# ----------------------------------------------------------------------
CLINICAL_REGIONS = [
    {
        # Device biofilm thickness of order 10-100 um (Vertes et al., 2012);
        # relaxation over 1e-2 to 1e1 s for D_p of order 1e-9 to 1e-8 m^2/s.
        "name": "Device-associated biofilms",
        "label": "Device-associated\nbiofilms",
        "L_um": (10.0, 120.0), "t_s": (1e-2, 1e1),
        "color": C_BLUE, "hatch": "xxx",
        "label_xy": (11.0, 1.2e-2), "label_va": "bottom",
    },
    {
        # Aggregates tens of um below the wound surface (James et al., 2008);
        # exudate, occlusion and compression can lengthen L_eff.
        "name": "Chronic wound biofilms",
        "label": "Chronic wound\nbiofilms",
        "L_um": (15.0, 600.0), "t_s": (0.3, 600.0),
        "color": C_GREEN, "hatch": "///",
        "label_xy": (17.0, 470.0), "label_va": "top",
    },
    {
        # Compacted aggregates (~0.2 mm) up to plug radii (plug diameters
        # 1-17 mm, lengths 2-50 mm; Huang et al., 2024); tau_drain of
        # order 1e2-1e3 s when no short internal sink is active.
        "name": "Airway mucus plugs or aggregates",
        "label": "Airway mucus plugs\nor aggregates",
        "L_um": (200.0, 1e4), "t_s": (1e2, 1e4),
        "color": C_ORANGE, "hatch": "\\\\\\",
        "label_xy": (225.0, 8.0e3), "label_va": "top",
    },
]

# ----------------------------------------------------------------------
# Schematic suppression/restoration protocol for panels (b) and (c)
# ----------------------------------------------------------------------
X_END = 8.0              # arbitrary time units
X_SUPPRESS = 2.0         # biosynthesis suppressed
X_RESTORE = 5.0          # biosynthesis restored
G_SUPPRESSED = 0.43      # steady suppressed conductance, G_h/G_h,0
TAU_RESPONSE = 0.40      # response time of the two-stage step response


def _step_response(s: np.ndarray) -> np.ndarray:
    """Remaining fraction after a step, two-stage first-order kinetics:
    (1 + s) exp(-s) for s >= 0 (zero slope at onset, strictly causal)."""
    s = np.clip(s, 0.0, None)
    return (1.0 + s) * np.exp(-s)


def active_conductance(x: np.ndarray) -> np.ndarray:
    """G_h/G_h,0 for actively maintained architecture (schematic)."""
    x = np.asarray(x, dtype=float)
    g = np.ones_like(x)
    sup = (x >= X_SUPPRESS) & (x < X_RESTORE)
    g[sup] = G_SUPPRESSED + (1.0 - G_SUPPRESSED) * _step_response(
        (x[sup] - X_SUPPRESS) / TAU_RESPONSE)
    g_at_restore = G_SUPPRESSED + (1.0 - G_SUPPRESSED) * _step_response(
        np.array([(X_RESTORE - X_SUPPRESS) / TAU_RESPONSE]))[0]
    res = x >= X_RESTORE
    g[res] = 1.0 - (1.0 - g_at_restore) * _step_response(
        (x[res] - X_RESTORE) / TAU_RESPONSE)
    return g


# ----------------------------------------------------------------------
# Saving and assembly (same policy as the main code)
# ----------------------------------------------------------------------

def out(filename: str) -> str:
    """Absolute path inside the shared output directory."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.abspath(os.path.join(OUTPUT_DIR, filename.strip()))


def safe_save_image(image: Image.Image, path: str, fmt: str,
                    dpi: int = ARTWORK_DPI, **kwargs) -> str:
    """Save a PIL image; if the file is locked (open in a viewer), save
    a '_latest' copy instead."""
    save_kwargs = dict(kwargs)
    save_kwargs["dpi"] = (int(dpi), int(dpi))
    try:
        image.save(path, format=fmt, **save_kwargs)
        return path
    except (OSError, PermissionError):
        base, ext = os.path.splitext(path)
        fallback = f"{base}_latest{ext}"
        image.save(fallback, format=fmt, **save_kwargs)
        print(f"  [WARNING] '{path}' is locked; saved '{fallback}' instead.")
        return fallback


def save_working_block(fig: plt.Figure, filename: str,
                       dpi: int = ARTWORK_DPI) -> str:
    """Save a temporary high-resolution PNG block for assembly."""
    path = out(filename)
    fig.savefig(path, format="png", dpi=dpi, bbox_inches="tight",
                pad_inches=0.04, facecolor="white", edgecolor="none")
    return path


def compose_vertical(block_paths: Sequence[str], stem: str,
                     target_width_px: int = FULL_PAGE_WIDTH_PX,
                     gap_px: int = PANEL_GAP_PX,
                     dpi: int = ARTWORK_DPI,
                     remove_blocks: bool = True) -> Tuple[str, str]:
    """Stack panel blocks at the shared page width; save PNG and LZW TIFF."""
    try:
        resampling = Image.Resampling.LANCZOS
    except AttributeError:
        resampling = Image.LANCZOS

    images = []
    for path in block_paths:
        with Image.open(path) as source:
            img = source.convert("RGB").copy()
        new_h = int(round(img.height * target_width_px / img.width))
        images.append(img.resize((target_width_px, new_h), resample=resampling))

    total_h = sum(im.height for im in images) + gap_px * (len(images) - 1)
    canvas = Image.new("RGB", (target_width_px, total_h), color=(255, 255, 255))
    y = 0
    for im in images:
        canvas.paste(im, (0, y))
        y += im.height + gap_px

    png = safe_save_image(canvas, out(f"{stem}.png"), "PNG", dpi=dpi, optimize=True)
    tif = safe_save_image(canvas, out(f"{stem}.tiff"), "TIFF", dpi=dpi,
                          compression="tiff_lzw")
    print(f"  [Composite PNG saved]  {png} ({canvas.width} x {canvas.height} px)")
    print(f"  [Composite TIFF saved] {tif} ({canvas.width} x {canvas.height} px)")

    if remove_blocks:
        for path in block_paths:
            try:
                os.remove(path)
            except OSError:
                pass
    return png, tif


# ----------------------------------------------------------------------
# Consistency check against the main code (skipped if not importable)
# ----------------------------------------------------------------------

def check_against_main_code() -> None:
    """Confirm that panel (a) uses the same physics and parameters as
    Fig. 3(a) of biofilm_poroelastic_sim.py, if that file sits alongside."""
    try:
        if SCRIPT_DIR not in sys.path:
            sys.path.insert(0, SCRIPT_DIR)
        import biofilm_poroelastic_sim as sim
    except Exception as exc:  # main code absent or its imports missing
        print(f"  [consistency] main code not importable "
              f"({exc.__class__.__name__}); check skipped.")
        return

    problems = []
    if not math.isclose(sim.GF_OPEN_SEALED, GF_OPEN_SEALED, rel_tol=1e-12):
        problems.append("GF_OPEN_SEALED differs")

    src = inspect.getsource(sim.figure_3a)
    m_k = re.search(r"k_low,\s*k_high\s*=\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)", src)
    m_m = re.search(r"M_c,\s*eta\s*=\s*([0-9.eE+-]+),\s*([0-9.eE+-]+)", src)
    if not m_k or (float(m_k.group(1)), float(m_k.group(2))) != (K_LOW, K_HIGH):
        problems.append("Fig. 3(a) k band differs")
    if not m_m or (float(m_m.group(1)), float(m_m.group(2))) != (M_C, ETA):
        problems.append("Fig. 3(a) M_c or eta differs")

    D_here = poroelastic_diffusivity(K_MID, M_C, ETA)
    D_main = sim.poroelastic_diffusivity(K_MID, M_C, ETA)
    for L_um, t_s in [(10, 0.05), (100, 1.0), (1000, 300.0), (5000, 2000.0)]:
        tau_main = sim.drainage_time(L_um * 1e-6, D_main, sim.GF_OPEN_SEALED)
        p_main = sim.terzaghi_exact(np.array([t_s]), tau_main, n_modes=N_MODES)[0]
        p_here = terzaghi_exact(t_s, drainage_time(L_um * 1e-6, D_here))
        if not math.isclose(float(p_main), float(p_here), rel_tol=1e-12, abs_tol=1e-15):
            problems.append(f"Terzaghi mean differs at L={L_um} um, t={t_s} s")

    if problems:
        print(f"  [consistency] WARNING vs biofilm_poroelastic_sim.py "
              f"{getattr(sim, '__version__', '?')}: " + "; ".join(problems))
    else:
        print(f"  [consistency] panel (a) matches Fig. 3(a) of "
              f"biofilm_poroelastic_sim.py {getattr(sim, '__version__', '?')} "
              f"(k band, M_c, eta, geometry factor, Terzaghi series).")


# ----------------------------------------------------------------------
# Panel (a): clinical settings on the drainage-regime map
# ----------------------------------------------------------------------

def draw_panel_a() -> plt.Figure:
    """Figure 6(a)."""
    L_um = np.logspace(math.log10(L_LIM_UM[0]), math.log10(L_LIM_UM[1]), 161)
    t_s = np.logspace(math.log10(T_LIM_S[0]), math.log10(T_LIM_S[1]), 161)
    LL, TT = np.meshgrid(L_um, t_s)

    D_mid = poroelastic_diffusivity(K_MID, M_C, ETA)
    Z = np.clip(terzaghi_exact(TT, drainage_time(LL * 1e-6, D_mid)), 0.0, 1.0)

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    pcm = ax.pcolormesh(LL, TT, Z, shading="auto", cmap="RdYlBu_r",
                        vmin=0.0, vmax=1.0, zorder=0)
    cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(r"$\langle p\rangle/p_0$ (Terzaghi series, open-sealed)",
                   fontsize=12, fontweight="bold")

    # N_d = 1 lines for the low and high bounds of the k band
    L_line = np.logspace(math.log10(L_LIM_UM[0]), math.log10(L_LIM_UM[1]), 400)
    for k, style, lab in [
        (K_LOW, "k--", r"$N_{\mathrm{d}}=1$, low $k=10^{-17}$ m$^2$"),
        (K_HIGH, "k:", r"$N_{\mathrm{d}}=1$, high $k=10^{-14}$ m$^2$"),
    ]:
        tau = drainage_time(L_line * 1e-6, poroelastic_diffusivity(k, M_C, ETA))
        ax.plot(L_line, tau, style, lw=2.0, label=lab, zorder=3)

    # Clinical regions: translucent fill, hatch, solid outline, label
    for reg in CLINICAL_REGIONS:
        (x0, x1), (y0, y1) = reg["L_um"], reg["t_s"]
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor=reg["color"],
                               edgecolor="none", alpha=0.20, zorder=4))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                               edgecolor=reg["color"], hatch=reg["hatch"],
                               lw=0.0, alpha=0.55, zorder=4))
        ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, facecolor="none",
                               edgecolor=reg["color"], lw=2.2, zorder=5))
        ax.text(*reg["label_xy"], reg["label"], color=reg["color"],
                fontsize=11, fontweight="bold", ha="left", va=reg["label_va"],
                zorder=7, bbox=dict(boxstyle="round,pad=0.25", fc="white",
                                    ec="none", alpha=0.88))

    # Management shifts the operative drainage length of one wound
    # (horizontal move at fixed intervention time, crossing N_d ~ 1).
    ax.annotate("", xy=(560.0, 15.0), xytext=(24.0, 15.0), zorder=7,
                arrowprops=dict(arrowstyle="<|-|>", color=C_GREEN, lw=2.0,
                                mutation_scale=14, shrinkA=0, shrinkB=0))
    ax.text(116.0, 21.0, "same wound can shift\nwith management",
            color=C_GREEN, fontsize=10, style="italic", ha="center",
            va="bottom", zorder=7,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))

    # Regime labels (colors as in Fig. 3(a))
    regime_kw = dict(fontsize=11.5, fontweight="bold", zorder=8)
    ax.text(1.4, 1.5e3, "DRAINED\n" r"($N_{\mathrm{d}}\gg1$)",
            color="#00CFFF", ha="left", va="top", **regime_kw)
    ax.text(8.6e3, 0.015, "UNDRAINED\n" r"($N_{\mathrm{d}}\ll1$)",
            color="#FF6B6B", ha="right", va="bottom", **regime_kw)
    ax.text(690.0, 48.0, "INTERMEDIATE\n" r"($N_{\mathrm{d}}\approx1$)",
            color="black", ha="left", va="center", **regime_kw)

    ax.text(0.01, 0.985, "Shaded regions are order-of-magnitude regime bounds, "
            "not patient-specific estimates.", transform=ax.transAxes,
            fontsize=9.5, color="white", ha="left", va="top", zorder=8,
            path_effects=[pe.withStroke(linewidth=2.0, foreground="black")])

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*L_LIM_UM)
    ax.set_ylim(*T_LIM_S)
    ax.set_xlabel(r"Operative drainage length $L_{\mathrm{eff}}$ ($\mathrm{\mu}$m)",
                  fontsize=12, fontweight="bold")
    ax.set_ylabel(r"Observation / intervention time $t_{\mathrm{obs}}$ (s)",
                  fontsize=12, fontweight="bold")
    ax.set_title("(a) Clinical settings occupy distinct but partly overlapping "
                 "drainage-time windows", fontsize=14.0, pad=10)
    ax.legend(fontsize=9.5, loc="lower left", framealpha=0.92)
    ax.grid(alpha=0.2, which="both")
    plt.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Panels (b) and (c): reversible test of active hydraulic architecture
# ----------------------------------------------------------------------

def _phase_axes(ax: plt.Axes) -> None:
    """Shade the three protocol phases and label them on the x axis."""
    ax.axvspan(0.0, X_SUPPRESS, color="#F3F3F3", zorder=0)
    ax.axvspan(X_SUPPRESS, X_RESTORE, color="#F7E4EE", zorder=0)
    ax.axvspan(X_RESTORE, X_END, color="#E8F2F9", zorder=0)
    for xb in (X_SUPPRESS, X_RESTORE):   # stop above the footnote
        ax.axvline(xb, ymin=0.075, color="gray", ls="--", lw=1.0, zorder=1)
    ax.set_xlim(0.0, X_END)
    ax.set_xticks([X_SUPPRESS / 2, (X_SUPPRESS + X_RESTORE) / 2,
                   (X_RESTORE + X_END) / 2])
    ax.set_xticklabels(["Baseline", "Suppression", "Restoration"], fontsize=12)
    ax.tick_params(axis="x", length=0)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(alpha=0.25, axis="y")


def draw_panels_bc() -> Tuple[plt.Figure, float]:
    """Figure 6(b) and 6(c). Returns the figure and the plotted peak."""
    x = np.linspace(0.0, X_END, 1601)
    g_active = active_conductance(x)
    tau_active = 1.0 / g_active           # tau_drain/tau_0 = G_h,0/G_h
    ones = np.ones_like(x)

    fig, (ax_b, ax_c) = plt.subplots(1, 2, figsize=(12.5, 5.4))
    note_kw = dict(fontsize=10, color="dimgray", style="italic",
                   ha="left", va="bottom", zorder=6)

    # (b) conductance
    _phase_axes(ax_b)
    ax_b.plot(x, g_active, color=C_BLUE, lw=2.8, label="Active maintenance", zorder=4)
    ax_b.plot(x, ones, color=C_ORANGE, lw=2.8, ls="--",
              label="Passive / inherited", zorder=5)
    x_min = 4.0
    ax_b.annotate(r"$G_h\!\downarrow$ during suppression",
                  xy=(x_min, float(active_conductance(np.array([x_min]))[0]) - 0.01),
                  xytext=(2.1, 0.25), color=C_BLUE, fontsize=11.5, ha="center",
                  arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.3), zorder=6)
    x_rec = 5.9
    ax_b.annotate("recovery on\nrestoration",
                  xy=(x_rec, float(active_conductance(np.array([x_rec]))[0]) - 0.01),
                  xytext=(6.55, 0.42), color=C_BLUE, fontsize=11.5, ha="center",
                  arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.3), zorder=6)
    ax_b.text(0.02, 0.025, "Schematic normalized pressure-driven hydraulic conductance.",
              transform=ax_b.transAxes, **note_kw)
    ax_b.set_ylim(-0.05, 1.55)
    ax_b.set_ylabel(r"Hydraulic conductance $G_h/G_{h,0}$",
                    fontsize=12, fontweight="bold")
    ax_b.set_title("(b) Conductance recovers if architecture is\nactively maintained",
                   fontsize=14.0, pad=10)
    ax_b.legend(fontsize=11, loc="upper left", framealpha=0.92)

    # (c) drainage time
    _phase_axes(ax_c)
    ax_c.plot(x, tau_active, color=C_BLUE, lw=2.8,
              label=r"Active: $\tau_{\mathrm{drain}}/\tau_0 = G_{h,0}/G_h$", zorder=4)
    ax_c.plot(x, ones, color=C_ORANGE, lw=2.8, ls="--",
              label="Passive / inherited", zorder=5)
    x_up = 2.75
    ax_c.annotate(r"$G_h\!\downarrow\;\Rightarrow\;\tau_{\mathrm{drain}}\!\uparrow$",
                  xy=(x_up, float(1.0 / active_conductance(np.array([x_up]))[0])),
                  xytext=(1.2, 1.9), color=C_VERMILION, fontsize=12, ha="center",
                  arrowprops=dict(arrowstyle="->", color=C_VERMILION, lw=1.3), zorder=6)
    x_dis = 6.0
    ax_c.annotate("biosynthesis-linked\nrecovery is the\ndiscriminant",
                  xy=(x_dis, float(1.0 / active_conductance(np.array([x_dis]))[0]) + 0.02),
                  xytext=(6.68, 2.42), color=C_BLUE, fontsize=11, ha="center",
                  va="center", arrowprops=dict(arrowstyle="->", color=C_BLUE, lw=1.3),
                  zorder=6)
    ax_c.text(0.02, 0.025, r"Controlled limit: fixed $L_{\mathrm{eff}}$, $\eta$, $M_c$.",
              transform=ax_c.transAxes, **note_kw)
    ax_c.set_ylim(-0.05, 3.25)
    ax_c.set_ylabel(r"Drainage time $\tau_{\mathrm{drain}}/\tau_0$",
                    fontsize=12, fontweight="bold")
    ax_c.set_title("(c) Paired signature: decreasing $G_h$\n"
                   r"increases $\tau_{\mathrm{drain}}$ (peak $= G_{h,0}/G_{h,\min}$)",
                   fontsize=14.0, pad=10)
    ax_c.legend(fontsize=11, loc="upper left", framealpha=0.92)

    plt.tight_layout(w_pad=3.0)
    return fig, float(tau_active.max())


# ----------------------------------------------------------------------
# Region table (lets the Section IV.B regime statements be checked)
# ----------------------------------------------------------------------

def write_region_csv(path: str) -> None:
    """Drainage times and N_d ranges of each region at the background k."""
    D_mid = poroelastic_diffusivity(K_MID, M_C, ETA)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["setting", "L_min_um", "L_max_um", "t_min_s", "t_max_s",
                    "tau_drain_min_s", "tau_drain_max_s", "Nd_min", "Nd_max",
                    "k_background_m2", "D_p_m2_s"])
        for reg in CLINICAL_REGIONS:
            (x0, x1), (y0, y1) = reg["L_um"], reg["t_s"]
            tau0 = float(drainage_time(x0 * 1e-6, D_mid))
            tau1 = float(drainage_time(x1 * 1e-6, D_mid))
            w.writerow([reg["name"], x0, x1, y0, y1, f"{tau0:.3g}", f"{tau1:.3g}",
                        f"{y0 / tau1:.3g}", f"{y1 / tau0:.3g}",
                        f"{K_MID:.3g}", f"{D_mid:.3g}"])
    print(f"  [CSV saved] {path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def make_figure_6() -> None:
    """Build Figure 6: panel (a) on top, panels (b) and (c) below."""
    print("\n[Figure 6] Clinical drainage-regime map and active-architecture test")
    print(f"  script version {__version__}")
    check_against_main_code()

    fig_a = draw_panel_a()
    block_a = save_working_block(fig_a, "_Figure_6_a_working.png")
    plt.close(fig_a)

    fig_bc, tau_peak = draw_panels_bc()
    block_bc = save_working_block(fig_bc, "_Figure_6_bc_working.png")
    plt.close(fig_bc)
    print(f"  panel (c) peak tau_drain/tau_0 = {tau_peak:.3f} "
          f"(= G_h,0/G_h,min; steady suppressed G_h/G_h,0 = {G_SUPPRESSED})")

    compose_vertical([block_a, block_bc], stem="Figure_6")
    write_region_csv(out("Figure_6_region_bounds.csv"))
    print("[Figure 6] done.\n")


if __name__ == "__main__":
    make_figure_6()
