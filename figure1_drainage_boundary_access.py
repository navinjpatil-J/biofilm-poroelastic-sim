# -*- coding: utf-8 -*-
"""
figure1_drainage_boundary_access.py
===================================

Figure 1 of the review article "Interfacial drainage regime as a coordinate
linking biofilm mechanics and transport": interfacial drainage access as a
boundary condition for apparent biofilm stiffness.

Version 3.1.0. Released under the MIT License (see LICENSE).

Panels
------
(a) Schematic. The same biofilm under the same volumetric load sits on an
    impermeable (glass-like) or a permeable (agar-like) base. With a sealed
    base the free top surface is the only drainage boundary, so the
    operative drainage length L_eff is about the full thickness. A permeable
    base adds a second pathway, so L_eff is shorter. The downward arrows are
    thinner and lighter because the agar has finite permeability, whereas
    the free surface offers no hydraulic resistance.
(b) Readout-time dependence (conceptual). E_app relaxes toward the drained
    limit as a stretched exponential for two drainage lengths
    (L_eff = 200 and 80 um), with tau_drain = C_tau L_eff^2 and
    C_tau = 1e-4 s um^-2, i.e. D_p of order 1e-8 m^2 s^-1 as in Figs. 2 and
    4. The readout time t_obs = 1 s lies between the two drainage times
    (0.64 s and 4 s), where the difference Delta E_app is largest.
(c) Boundary-perturbation diagnostic (conceptual). E_app against the
    effective permeability k_b of the basal boundary (not the matrix
    permeability k), with tau_drain proportional to 1/k_b, tau = 4 s at
    k_b = 1e-14 m^2, read at t_obs = 1 s. The band is a schematic envelope,
    not an experimental uncertainty; the dashed line is the expected
    response of a shear-dominated control.

Panels (b) and (c) illustrate the scaling; they are not fitted to data.

How to run
----------
    python figure1_drainage_boundary_access.py

Outputs (in biofilm_sim_outputs/ next to this script)
-----------------------------------------------------
    Figure_1.pdf    vector
    Figure_1.tiff   600 dpi, LZW-compressed
    Figure_1.png    600 dpi
"""

from __future__ import annotations

import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.patheffects as pe
from matplotlib.patches import Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, Rectangle
from matplotlib.transforms import blended_transform_factory

# ============================================================
# STYLE
# ============================================================
# One font-size scale for the whole figure (pt on the 16 in x 9 in canvas).
# Weights follow Figs. 2-6: normal titles, bold axis labels and annotations.
FS = {
    "title": 17,    # (a), (b), (c) panel titles
    "label": 14,    # axis labels
    "text": 13,     # every annotation, including the summary lines in (a)
    "tick": 12,     # tick labels
    "legend": 11,   # legends in (b) and (c)
}

rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Helvetica"],
    "font.size": FS["tick"],
    "axes.linewidth": 1.8,
    "axes.labelsize": FS["label"],
    "axes.labelweight": "bold",
    "xtick.labelsize": FS["tick"],
    "ytick.labelsize": FS["tick"],
    "legend.fontsize": FS["legend"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

SCRIPT_DIR = (os.path.dirname(os.path.abspath(__file__))
              if "__file__" in globals() else os.getcwd())
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "biofilm_sim_outputs")

# Layering (BIO behind, ANNOTATIONS above)
Z = {
    "substrate": 1,
    "biofilm": 2,
    "eps": 3,
    "p_circ": 3.5,
    "cells": 4,
    "cell_hi": 5,
    "load": 12,
    "drain": 13,
    "anno": 30,
    "title": 50,
}

COL = {
    "accessible": "#0072B2",
    "inaccessible": "#E69F00",
    "biofilm": "#E6F3E6",
    "mesh": "#2F6F4E",
    "cells": "#2E8B57",
    "cell_edge": "#1B5E3A",
    "p_high": "#E85555",
    "p_low": "#3CB4A0",
    "solid": "#606060",
    "porous": "#C0B296",
    "pore_fill": "#FFFFFF",
    "ink": "#2A2A2A",
    "prediction": "#D55E00",
    "null": "#666666",
    "drained_limit": "#228B22",
}

# ============================================================
# LAYOUT (figure fractions on the 16 in x 9 in canvas)
# ============================================================
AX_A = [0.02, 0.08, 0.46, 0.88]   # (a) schematic
AX_B = [0.56, 0.58, 0.40, 0.34]   # (b) readout-time plot
AX_C = [0.56, 0.12, 0.40, 0.34]   # (c) boundary-perturbation plot
TITLE_GAP = 0.0135                # plot top to title baseline
TITLE_Y_TOP = AX_B[1] + AX_B[3] + TITLE_GAP   # baseline shared by (a) and (b)
TITLE_Y_C = AX_C[1] + AX_C[3] + TITLE_GAP     # baseline of (c)


# ============================================================
# HELPERS
# ============================================================
def _arrow(ax, p0, p1, color, lw=3.0, alpha=1.0, ls="-", ms=20, z=6):
    a = FancyArrowPatch(
        p0, p1,
        arrowstyle="->",
        mutation_scale=ms,
        lw=lw,
        color=color,
        alpha=alpha,
        linestyle=ls,
        zorder=z,
    )
    return ax.add_patch(a)


def _halo(color="white", lw=3.0):
    return [pe.Stroke(linewidth=lw, foreground=color), pe.Normal()]


def panel_title(ax, s: str, y_fig: float):
    """Panel title in the APR style "(x) Title".

    Centred on the panel, with its baseline at y_fig (figure fraction), so
    titles (a) and (b) share one baseline. Size, weight and colour are the
    same for all three panels.
    """
    trans = blended_transform_factory(ax.transAxes, ax.figure.transFigure)
    t = ax.text(
        0.5, y_fig, s,
        transform=trans,
        ha="center", va="baseline",
        fontsize=FS["title"], weight="normal", color=COL["ink"],
        clip_on=False, zorder=Z["title"],
    )
    return t


# ============================================================
# PANEL (a) DRAWING FUNCTIONS
# ============================================================
def draw_biofilm(ax, xc, yb, w, h, stype, seed=42):
    xl, xr, yt = xc - w / 2, xc + w / 2, yb + h
    sub_h = h * 0.15
    sub_y = yb - sub_h

    # Substrate
    scol = COL["solid"] if stype == "impermeable" else COL["porous"]
    ax.add_patch(Rectangle(
        (xl, sub_y), w, sub_h,
        facecolor=scol, edgecolor=COL["ink"], lw=2, zorder=Z["substrate"]
    ))

    # Substrate labels
    label = "Impermeable\n(glass-like)" if stype == "impermeable" else "Permeable\n(agar-like)"
    t = ax.text(
        xc, sub_y + sub_h * 0.5, label,
        ha="center", va="center",
        color=COL["ink"], weight="bold", fontsize=FS["text"],
        zorder=Z["anno"],
    )
    t.set_path_effects(_halo(lw=2.4))

    # Pores for permeable substrate
    if stype == "permeable":
        for r in range(4):
            for c in range(11):
                px = xl + (c + 0.5) * (w / 11)
                py = sub_y + (r + 0.5) * (sub_h / 4)
                ax.add_patch(Circle(
                    (px, py), w * 0.018,
                    fc=COL["pore_fill"], ec="gray", alpha=0.6,
                    zorder=Z["substrate"] + 0.1
                ))

    # Biofilm body
    ax.add_patch(FancyBboxPatch(
        (xl, yb), w, h,
        boxstyle="round,pad=0.01",
        fc=COL["biofilm"], ec=COL["ink"], lw=2, alpha=0.85,
        zorder=Z["biofilm"]
    ))

    rng = np.random.default_rng(seed)

    # EPS fibers
    for _ in range(36):
        x1, x2 = rng.uniform(xl + w * 0.10, xr - w * 0.10, 2)
        y1, y2 = rng.uniform(yb + h * 0.10, yt - h * 0.10, 2)
        rad = rng.uniform(-0.55, 0.55)
        lw_eps = rng.uniform(1.0, 1.8)
        a = rng.uniform(0.08, 0.18)
        eps = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle="-",
            connectionstyle=f"arc3,rad={rad}",
            color=COL["mesh"],
            alpha=a,
            lw=lw_eps,
            zorder=Z["eps"],
        )
        eps.set_capstyle("round")
        eps.set_joinstyle("round")
        ax.add_patch(eps)

    # Cells
    for _ in range(34):
        cx = rng.uniform(xl + w * 0.12, xr - w * 0.12)
        cy = rng.uniform(yb + h * 0.12, yt - h * 0.12)
        ang = rng.uniform(0, 180)
        cell_L = rng.uniform(w * 0.070, w * 0.095)
        cell_W = rng.uniform(w * 0.026, w * 0.040)
        cell = Ellipse(
            (cx, cy), cell_L, cell_W, angle=ang,
            fc=COL["cells"], ec="none", alpha=0.93, zorder=Z["cells"],
        )
        cell.set_path_effects([
            pe.SimplePatchShadow(offset=(1.0, -1.0), alpha=0.18, rho=0.95),
            pe.Stroke(linewidth=1.4, foreground=COL["cell_edge"]),
            pe.Normal(),
        ])
        ax.add_patch(cell)
        hi = Ellipse(
            (cx - w * 0.006, cy + w * 0.006),
            cell_L * 0.55, cell_W * 0.55,
            angle=ang, fc="white", ec="none", alpha=0.14, zorder=Z["cell_hi"],
        )
        ax.add_patch(hi)

    # Load arrows
    for f in [0.25, 0.5, 0.75]:
        xx = xl + f * w
        _arrow(
            ax, (xx, yt + h * 0.12), (xx, yt + h * 0.015),
            color=COL["ink"], lw=3, alpha=1.0, ls="-", ms=20, z=Z["load"],
        )

    # Pressure bubble
    if stype == "impermeable":
        pc = COL["p_high"]
        pl = r"excess $p$ high" + "\n" + "(undrained-like)"
    else:
        pc = COL["p_low"]
        pl = r"excess $p$ low" + "\n" + "(more drained-like)"

    cen = (xc, yb + h * 0.5)
    ax.add_patch(Circle(
        cen, min(w, h) * 0.28, fc=pc, alpha=0.18, zorder=Z["p_circ"]
    ))
    tt = ax.text(
        cen[0], cen[1], pl,
        ha="center", va="center",
        weight="bold", color=pc, fontsize=FS["text"],
        zorder=Z["anno"],
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=pc, lw=1.2, alpha=0.97),
    )
    tt.set_path_effects(_halo(lw=2.8))

    # ============================================================
    # Drainage arrows and L_eff indicator
    # ============================================================
    # Common x-positions for BOTH schematics: isolates boundary condition
    # as the sole varying attribute.
    x_drain = [0.30, 0.50, 0.70]

    if stype == "impermeable":
        ac = COL["inaccessible"]
        # SINGLY-DRAINED (Glass): sealed base is no-flux; only free top is a sink.
        # Arrows span near-base to top so visual length ≈ thickness.
        for xf in x_drain:
            xx = xl + xf * w
            _arrow(
                ax, (xx, yb + 0.05 * h), (xx, yt + 0.03 * h),
                color=ac, lw=3.0, alpha=0.95, ls="-", ms=18, z=Z["drain"],
            )

        # No-flux marker at the sealed base
        for xf in x_drain:
            xx = xl + xf * w
            ax.plot(
                [xx - 0.035 * w, xx + 0.035 * w],
                [yb + 0.025 * h, yb + 0.025 * h],
                color=ac, lw=3.0, alpha=0.95,
                solid_capstyle="round", zorder=Z["drain"],
            )

        # L_eff bracket: full thickness
        ax.add_patch(FancyArrowPatch(
            (xl - 0.07 * w, yb), (xl - 0.07 * w, yt),
            arrowstyle="<->", mutation_scale=18, lw=2.8, color=ac, alpha=0.95,
            zorder=Z["drain"],
        ))
        t2 = ax.text(
            xl - 0.10 * w, yb + h / 2,
            r"$L_{\mathrm{eff}}\approx$ thickness",
            ha="right", va="center",
            color=ac, weight="bold", fontsize=FS["text"], zorder=Z["anno"],
        )
        t2.set_path_effects(_halo(lw=2.6))

    else:
        ac = COL["accessible"]
        # ASYMMETRICALLY DOUBLY-DRAINED (Agar): ideal open top + finite-k base.
        # Upward arrows: dominant pathway (same weight as sealed-case top arrows).
        # Downward arrows: weaker agar pathway (lighter weight).
        # Watershed sits below mid-plane (~0.40 h).

        # Upper portion → upward to free top surface
        for xf in x_drain:
            xx = xl + xf * w
            _arrow(
                ax, (xx, yb + 0.40 * h), (xx, yt + 0.03 * h),
                color=ac, lw=3.0, alpha=0.95, ls="-", ms=18, z=Z["drain"],
            )

        # Lower portion → downward into permeable substrate (weaker flux)
        for xf in x_drain:
            xx = xl + xf * w
            _arrow(
                ax, (xx, yb + 0.38 * h), (xx, sub_y + sub_h * 0.9),
                color=ac, lw=2.2, alpha=0.70, ls="-", ms=14, z=Z["drain"],
            )

        # L_eff bracket: shorter than full thickness
        ax.add_patch(FancyArrowPatch(
            (xr + 0.08 * w, yb), (xr + 0.08 * w, yb + 0.50 * h),
            arrowstyle="<->", mutation_scale=18, lw=2.8, color=ac, alpha=0.95,
            zorder=Z["drain"],
        ))
        t2 = ax.text(
            xr + 0.11 * w, yb + 0.25 * h,
            r"$L_{\mathrm{eff}}$ shorter",
            ha="left", va="center",
            color=ac, weight="bold", fontsize=FS["text"], zorder=Z["anno"],
        )
        t2.set_path_effects(_halo(lw=2.6))


def draw_panel_A(ax):
    ax.set_aspect("equal")
    ax.set_xlim(-0.04, 1.06)
    ax.set_ylim(-0.21, 1.03)
    ax.axis("off")

    panel_title(
        ax, r"(a) Drainage boundary sets effective path length $L_{\mathrm{eff}}$",
        TITLE_Y_TOP,
    )

    draw_biofilm(ax, 0.25, 0.22, 0.42, 0.58, "impermeable", seed=1)
    draw_biofilm(ax, 0.75, 0.22, 0.42, 0.58, "permeable", seed=2)

    mid = ax.text(
        0.5, 0.85, "Same biofilm\nSame loading",
        ha="center", weight="bold", fontsize=FS["text"],
        color=COL["ink"], zorder=Z["anno"],
    )
    mid.set_path_effects(_halo(lw=2.6))

    slow = ax.text(
        0.25, -0.01, "SLOW DRAINAGE",
        ha="center", color=COL["inaccessible"],
        weight="bold", fontsize=FS["text"], zorder=Z["anno"],
    )
    slow.set_path_effects(_halo(lw=2.6))

    fast = ax.text(
        0.75, -0.01, "FAST DRAINAGE",
        ha="center", color=COL["accessible"],
        weight="bold", fontsize=FS["text"], zorder=Z["anno"],
    )
    fast.set_path_effects(_halo(lw=2.6))

    bottom_str = (
        r"$\tau_{\mathrm{drain}} \propto L_{\mathrm{eff}}^{2}$" "\n"
        r"Boundary sensitivity is largest when $t_{\mathrm{obs}} \sim \tau_{\mathrm{drain}}$" "\n"
        r"At matched $t_{\mathrm{obs}}$, larger $L_{\mathrm{eff}}$ yields higher $E_{\mathrm{app}}$"
    )
    bottom = ax.text(
        0.5, -0.15, bottom_str,
        ha="center", va="center",
        fontsize=FS["text"], fontweight="normal",
        color=COL["ink"], alpha=0.95,
        zorder=Z["anno"],
        wrap=False, clip_on=False,
    )
    bottom.set_path_effects(_halo(lw=2.2))


def draw_panel_B(ax):
    panel_title(ax, r"(b) Readout-time dependence at fixed $t_{\mathrm{obs}}$", TITLE_Y_TOP)

    time = np.logspace(-2, 2, 500)
    E_drained, E_undrained = 0.30, 1.0
    gamma = 0.85  # stretched-exponential exponent (gamma in the article)

    # C_tau = 1e-4 s/um^2  → consistent with D_p ≈ 1e-8 m^2/s
    C_tau = 1e-4
    L_large = 200.0  # um
    L_small = 80.0   # um

    tau_large = C_tau * (L_large ** 2)   # = 4.0 s
    tau_small = C_tau * (L_small ** 2)   # = 0.64 s

    E_app_large = E_drained + (E_undrained - E_drained) * np.exp(-(time / tau_large) ** gamma)
    E_app_small = E_drained + (E_undrained - E_drained) * np.exp(-(time / tau_small) ** gamma)

    ax.semilogx(
        time, E_app_large / E_undrained, color=COL["inaccessible"], lw=3.5,
        label=r"Sealed base, glass-like ($L_{\mathrm{eff}}$ larger)",
    )
    ax.semilogx(
        time, E_app_small / E_undrained, color=COL["accessible"], lw=3.5,
        label=r"Open base, agar-like ($L_{\mathrm{eff}}$ smaller)",
    )
    ax.axhline(
        E_drained / E_undrained, color=COL["drained_limit"], lw=2.5, ls="--",
        label="Drained limit",
    )

    # t_obs = 1.0 s in intermediate diagnostic window between tau_small and tau_large
    t_obs = 1.0
    ax.axvline(t_obs, color="black", lw=1.8, alpha=0.5)

    E1 = (E_drained + (E_undrained - E_drained) * np.exp(-(t_obs / tau_large) ** gamma)) / E_undrained
    E2 = (E_drained + (E_undrained - E_drained) * np.exp(-(t_obs / tau_small) ** gamma)) / E_undrained

    ax.scatter([t_obs], [E1], s=120, color=COL["inaccessible"], ec="white", zorder=10)
    ax.scatter([t_obs], [E2], s=120, color=COL["accessible"], ec="white", zorder=10)

    ax.annotate(
        "", xy=(t_obs * 1.15, E2), xytext=(t_obs * 1.15, E1),
        arrowprops=dict(
            arrowstyle="<->", color=COL["prediction"], lw=2.5,
            shrinkA=0, shrinkB=0,
        ),
        zorder=15,
    )
    delta_label = ax.text(
        t_obs * 1.45, (E1 + E2) / 2, r"$\Delta E_{\mathrm{app}}$",
        color=COL["prediction"], fontsize=FS["text"], weight="bold",
        va="center", ha="left", zorder=15,
    )
    delta_label.set_path_effects(_halo(lw=2.0))

    t_obs_label = ax.text(
        t_obs, 0.22, r"$t_{\mathrm{obs}}$",
        color=COL["ink"], fontsize=FS["text"], weight="bold",
        va="bottom", ha="center", zorder=15,
    )
    t_obs_label.set_path_effects(_halo(lw=2.0))

    ax.set_xlabel("Time (s)", fontsize=FS["label"], weight="bold")
    ax.set_ylabel(r"$E_{\mathrm{app}}/E_{\mathrm{undrained}}$", fontsize=FS["label"], weight="bold")
    ax.set_ylim(0.18, 1.08)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=False, fontsize=FS["legend"])


def draw_panel_C(ax):
    panel_title(ax, r"(c) Boundary perturbation diagnostic $E_{\mathrm{app}}(k_{\mathrm{b}})$", TITLE_Y_C)

    k_b_plot = np.logspace(-16, -12, 200)

    # tau_ref = 4.0 s matches tau_large in (b); t_obs = 1.0 s matches (b)
    k_b_ref = 1e-14
    tau_ref = 4.0
    E_drained, E_undrained = 0.30, 1.0
    t_obs = 1.0
    gamma = 0.85  # stretched-exponential exponent (gamma in the article)

    # Schematic tau(k_b) at fixed geometry and matrix state.
    # k_b = effective basal-boundary permeability (not matrix k).
    tau_of_k_b = tau_ref * (k_b_ref / k_b_plot)
    E_app = E_drained + (E_undrained - E_drained) * np.exp(-(t_obs / tau_of_k_b) ** gamma)

    ax.semilogx(
        k_b_plot, E_app / E_undrained, color=COL["prediction"], lw=3.5,
        label="Schematic model",
    )

    # Schematic envelope, not experimental uncertainty
    ax.fill_between(
        k_b_plot,
        (E_app / E_undrained) * 0.9,
        (E_app / E_undrained) * 1.1,
        color=COL["prediction"],
        alpha=0.2,
    )

    # Negative control expectation
    ax.semilogx(
        k_b_plot, np.full_like(k_b_plot, 0.65), color=COL["null"], lw=2.5, ls="--",
        label="Shear-dominated control",
    )

    # Anchor points for directionality
    k_b_pts = np.array([1e-16, 1e-15, 1e-14, 1e-13])
    tau_pts = tau_ref * (k_b_ref / k_b_pts)
    E_pts = E_drained + (E_undrained - E_drained) * np.exp(-(t_obs / tau_pts) ** gamma)
    ax.scatter(k_b_pts, E_pts / E_undrained, s=120, color=COL["prediction"], ec="white", zorder=10)

    t_glass = ax.text(
        k_b_pts[0], (E_pts[0] / E_undrained) - 0.06, "Glass",
        ha="center", weight="bold", fontsize=FS["text"], color=COL["ink"],
    )
    t_glass.set_path_effects(_halo(lw=2.2))

    t_agar = ax.text(
        k_b_pts[-1], (E_pts[-1] / E_undrained) - 0.06, "Agar",
        ha="center", weight="bold", fontsize=FS["text"], color=COL["ink"],
    )
    t_agar.set_path_effects(_halo(lw=2.2))

    regime_text = ax.text(
        1e-14, 1.02, r"Use when $t_{\mathrm{obs}} \sim \tau_{\mathrm{drain}}$",
        fontsize=FS["text"], style="italic", ha="center", va="bottom",
        color=COL["ink"], alpha=0.8,
    )
    regime_text.set_path_effects(_halo(lw=1.8))

    ax.set_xlabel(r"Boundary permeability $k_{\mathrm{b}}$ ($\mathrm{m}^{2}$)", fontsize=FS["label"], weight="bold")
    ax.set_ylabel(r"$E_{\mathrm{app}}/E_{\mathrm{undrained}}$", fontsize=FS["label"], weight="bold")
    ax.set_ylim(0.18, 1.08)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", frameon=False, fontsize=FS["legend"])


def build_figure():
    fig = plt.figure(figsize=(16, 9), facecolor="white")

    ax_A = fig.add_axes(AX_A)
    ax_B = fig.add_axes(AX_B)
    ax_C = fig.add_axes(AX_C)

    draw_panel_A(ax_A)
    draw_panel_B(ax_B)
    draw_panel_C(ax_C)
    return fig


def main():
    fig = build_figure()

    # Vector PDF plus 600 dpi TIFF (LZW) and PNG. Combination artwork needs
    # at least 500 dpi; the 16 in canvas gives about 9,600 px at 600 dpi.
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stem = os.path.join(OUTPUT_DIR, "Figure_1")
    export_dpi = 600

    fig.savefig(f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(f"{stem}.tiff", bbox_inches="tight", dpi=export_dpi,
                facecolor="white", edgecolor="none",
                pil_kwargs={"compression": "tiff_lzw"})
    fig.savefig(f"{stem}.png", bbox_inches="tight", dpi=export_dpi,
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"Saved {stem}.pdf, .tiff and .png ({export_dpi} dpi)")


if __name__ == "__main__":
    main()
