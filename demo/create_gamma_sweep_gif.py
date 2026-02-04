"""
Create an animated GIF showing Nash equilibrium for random matrix games.

- Matrix stretches HORIZONTALLY as gamma increases (transposed view)
- Square matrix at gamma = 1
- Clean layout with no text overlap

Usage:
    python -m demo.create_gamma_sweep_gif
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle, FancyBboxPatch
import matplotlib.patheffects as pe
from PIL import Image
from tqdm import tqdm
import io

from src.rs_zeroT import solve_zeroT_rs
from src.zeroT_lp import solve_minmax_lp, support_fraction


def create_frame(
    gamma: float,
    C: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
    f_scaled: float,
    theory_f: float,
    gamma_history: list,
    f_history: list,
    all_gammas: np.ndarray,
    all_theory_f: np.ndarray,
    frame_idx: int,
    total_frames: int,
    f_min: float,
    f_max: float,
    base_size: int,
) -> Image.Image:
    """Create a single frame."""
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 10,
        'mathtext.fontset': 'cm',
    })
    
    fig = plt.figure(figsize=(15, 11), facecolor='white')
    
    N, M = C.shape
    
    # Colors
    c_min = '#2563eb'
    c_max = '#dc2626'
    c_theory = '#f59e0b'
    c_num = '#059669'
    
    # === MATRIX SECTION (top) ===
    ax_mat = fig.add_axes([0.02, 0.42, 0.96, 0.53])
    ax_mat.set_xlim(0, 100)
    ax_mat.set_ylim(0, 100)
    ax_mat.set_aspect('equal')
    ax_mat.axis('off')
    
    # Title at very top
    ax_mat.text(50, 98, r'$\min_{\mathbf{x} \in \Delta_N} \max_{\mathbf{y} \in \Delta_M} \, \mathbf{x}^\top C \, \mathbf{y}$',
                fontsize=20, ha='center', va='top', color='#1e293b')
    
    # === Matrix size: HORIZONTAL stretch with gamma ===
    # At gamma=1: square. gamma>1: wider (more columns M relative to rows N)
    # But since gamma = N/M, when gamma>1, N>M, so we transpose the display
    # Display: rows = M (fixed), cols = N (varies with gamma)
    # This makes the matrix WIDER when gamma is larger
    
    C_display = C.T  # Transpose: now shape is (M, N)
    M_disp, N_disp = C_display.shape  # M_disp = M (fixed height), N_disp = N (width varies)
    
    p_display = p  # length N (for columns)
    q_display = q  # length M (for rows)
    
    # Matrix dimensions in plot units
    base_dim = 32
    mat_height = base_dim  # Fixed height
    mat_width = base_dim * gamma  # Width scales with gamma
    
    if mat_width > base_dim * 1.7:
        mat_width = base_dim * 1.7
    if mat_width < base_dim * 0.4:
        mat_width = base_dim * 0.4
    
    # Position matrix
    mat_cx = 50
    mat_cy = 55
    mat_left = mat_cx - mat_width / 2
    mat_bottom = mat_cy - mat_height / 2
    
    # Draw matrix
    cmap = LinearSegmentedColormap.from_list('payoff',
        ['#1e3a8a', '#3b82f6', '#bfdbfe', '#ffffff', '#fecaca', '#ef4444', '#991b1b'], N=256)
    vmax = np.abs(C_display).max()
    
    cell_w = mat_width / N_disp
    cell_h = mat_height / M_disp
    
    for i in range(M_disp):
        for j in range(N_disp):
            val = C_display[i, j]
            color = cmap((val + vmax) / (2 * vmax))
            rect = Rectangle(
                (mat_left + j * cell_w, mat_bottom + (M_disp - 1 - i) * cell_h),
                cell_w, cell_h,
                facecolor=color, edgecolor='white', linewidth=0.1
            )
            ax_mat.add_patch(rect)
    
    # Matrix border
    ax_mat.add_patch(Rectangle(
        (mat_left, mat_bottom), mat_width, mat_height,
        facecolor='none', edgecolor='#334155', linewidth=2
    ))
    
    # === Strategy bars ===
    
    # Top bar: x strategy (minimizer) - length N_disp
    x_strat = N * p_display
    bar_h_top = 4
    bar_top_y = mat_bottom + mat_height + 2
    max_x = max(3.5, x_strat.max() * 1.1)
    
    for j in range(N_disp):
        val = x_strat[j]
        w = cell_w * 0.88
        h = (val / max_x) * bar_h_top
        x = mat_left + j * cell_w + cell_w * 0.06
        if val > 0.02:
            ax_mat.add_patch(Rectangle(
                (x, bar_top_y), w, h,
                facecolor=c_min, edgecolor='none', alpha=0.8
            ))
    
    # Left bar: y strategy (maximizer) - length M_disp
    y_strat = M * q_display
    bar_w_left = 4
    bar_left_x = mat_left - bar_w_left - 2
    max_y = max(3.5, y_strat.max() * 1.1)
    
    for i in range(M_disp):
        val = y_strat[i]
        h = cell_h * 0.88
        w = (val / max_y) * bar_w_left
        y = mat_bottom + (M_disp - 1 - i) * cell_h + cell_h * 0.06
        if val > 0.02:
            ax_mat.add_patch(Rectangle(
                (bar_left_x + bar_w_left - w, y), w, h,
                facecolor=c_max, edgecolor='none', alpha=0.8
            ))
    
    # === Labels - carefully positioned to avoid overlap ===
    
    # Gamma badge (top-left, separate from everything)
    ax_mat.add_patch(FancyBboxPatch(
        (3, 88), 16, 8,
        boxstyle="round,pad=0.01,rounding_size=0.3",
        facecolor='#eef2ff', edgecolor='#6366f1', linewidth=2
    ))
    ax_mat.text(11, 92, rf'$\gamma = {gamma:.3f}$',
                fontsize=14, ha='center', va='center', color='#4338ca', fontweight='bold')
    
    # Matrix label (below matrix, centered)
    ax_mat.text(mat_cx, mat_bottom - 4,
                rf'$C \in \mathbb{{R}}^{{{N} \times {M}}}$',
                fontsize=13, ha='center', va='top', color='#475569')
    
    # x_i label (above top bar, right side)
    ax_mat.text(mat_left + mat_width + 2, bar_top_y + bar_h_top / 2,
                rf'$x_i = N p_i$', fontsize=11, ha='left', va='center', color=c_min)
    
    # y_j label (left of left bar, top)
    ax_mat.text(bar_left_x - 1, mat_bottom + mat_height,
                rf'$y_j = M q_j$', fontsize=11, ha='right', va='top', color=c_max)
    
    # Dimension labels on axes
    ax_mat.text(mat_cx, bar_top_y + bar_h_top + 3,
                rf'Minimizer: $N = {N}$', fontsize=11, ha='center', va='bottom', color=c_min, fontweight='bold')
    ax_mat.text(bar_left_x - 1, mat_cy,
                rf'Max: $M={M}$', fontsize=10, ha='right', va='center', color=c_max, fontweight='bold', rotation=90)
    
    # === Stats box (top-right, separate) ===
    rho_x = support_fraction(p)
    rho_y = support_fraction(q)
    error = abs(f_scaled - theory_f) / (abs(theory_f) + 1e-10) * 100
    
    ax_mat.add_patch(FancyBboxPatch(
        (78, 73), 20, 23,
        boxstyle="round,pad=0.01,rounding_size=0.3",
        facecolor='#fafafa', edgecolor='#e2e8f0', linewidth=1.5
    ))
    
    stats_x = 80
    stats_y = 93
    ax_mat.text(stats_x, stats_y, 'Statistics', fontsize=10, ha='left', va='center',
                fontweight='bold', color='#475569')
    
    stats = [
        (rf'$f_{{LP}} = {f_scaled:.4f}$', c_num),
        (rf'$f_{{RS}} = {theory_f:.4f}$', c_theory),
        (f'Error: {error:.2f}%', '#dc2626' if error > 3 else '#16a34a'),
        (rf'$\rho_x = {rho_x*100:.0f}\%$', c_min),
        (rf'$\rho_y = {rho_y*100:.0f}\%$', c_max),
    ]
    for i, (txt, col) in enumerate(stats):
        ax_mat.text(stats_x, stats_y - 3.5 - i * 3.3, txt,
                    fontsize=9, ha='left', va='center', color=col)
    
    # === Colorbar (far right) ===
    cbar_x = 97
    cbar_y = mat_bottom + 2
    cbar_h = mat_height - 4
    cbar_w = 1.5
    
    for k in range(20):
        val = -vmax + (2 * vmax) * k / 19
        color = cmap((val + vmax) / (2 * vmax))
        ax_mat.add_patch(Rectangle(
            (cbar_x, cbar_y + k * cbar_h / 20), cbar_w, cbar_h / 20,
            facecolor=color, edgecolor='none'
        ))
    ax_mat.add_patch(Rectangle(
        (cbar_x, cbar_y), cbar_w, cbar_h,
        facecolor='none', edgecolor='#94a3b8', linewidth=0.5
    ))
    
    # === PLOT SECTION (bottom) ===
    ax_plot = fig.add_axes([0.1, 0.08, 0.85, 0.28])
    ax_plot.set_facecolor('#fafafa')
    
    # Theory curve with fill
    ax_plot.fill_between(all_gammas, all_theory_f, alpha=0.12, color=c_theory)
    ax_plot.plot(all_gammas, all_theory_f, '-', color=c_theory, linewidth=3.5,
                 label=r'RS Theory $f(\gamma)$', alpha=0.95)
    
    # Numerical points
    if len(gamma_history) > 1:
        ax_plot.plot(gamma_history, f_history, 'o-', color=c_num, linewidth=2,
                     markersize=4, label='LP Numerical', alpha=0.85)
    
    # Current point
    ax_plot.scatter([gamma], [f_scaled], s=200, color=c_num, zorder=10,
                    edgecolors='white', linewidth=2.5)
    ax_plot.scatter([gamma], [theory_f], s=140, color=c_theory, zorder=9,
                    marker='D', edgecolors='white', linewidth=2)
    
    # Vertical line
    ax_plot.axvline(x=gamma, color='#6366f1', linestyle='--', alpha=0.4, linewidth=1.5)
    
    # gamma=1 marker
    ax_plot.axvline(x=1.0, color='#94a3b8', linestyle=':', alpha=0.5, linewidth=1)
    
    # Labels
    ax_plot.set_xlabel(r'$\gamma = N / M$', fontsize=13)
    ax_plot.set_ylabel(r'Scaled Value $f(\gamma)$', fontsize=13)
    ax_plot.set_title(r'Nash Equilibrium Value: RS Theory vs LP', fontsize=14, pad=8, color='#334155')
    
    # Legend
    ax_plot.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='#e2e8f0')
    
    # Limits with proper padding
    x_margin = (all_gammas.max() - all_gammas.min()) * 0.05
    y_margin = (f_max - f_min) * 0.15
    ax_plot.set_xlim(all_gammas.min() - x_margin, all_gammas.max() + x_margin)
    ax_plot.set_ylim(f_min - y_margin, f_max + y_margin)
    
    ax_plot.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#cbd5e1')
    ax_plot.tick_params(labelsize=10)
    
    # Progress bar
    prog = (frame_idx + 1) / total_frames
    prog_ax = fig.add_axes([0.1, 0.025, 0.85, 0.012])
    prog_ax.barh(0, prog, height=1, color='#6366f1', alpha=0.6)
    prog_ax.barh(0, 1, height=1, color='#e2e8f0', alpha=0.4, zorder=0)
    prog_ax.set_xlim(0, 1)
    prog_ax.axis('off')
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white', bbox_inches='tight', pad_inches=0.08)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    return img


def main():
    print("=" * 60)
    print("  Nash Equilibrium Animation")
    print("  Matrix stretches HORIZONTALLY with gamma")
    print("=" * 60)
    
    seed = 42
    base_size = 50
    gamma_values = np.concatenate([
        np.linspace(0.4, 1.0, 24),
        np.linspace(1.0, 1.6, 26)
    ])
    
    rng = np.random.default_rng(seed)
    max_N = int(2.0 * base_size) + 10
    C_full = rng.standard_normal(size=(max_N, base_size))
    
    print("\n[1/3] RS theory...")
    theory_gammas = np.linspace(0.35, 1.7, 100)
    theory_f_values = []
    for g in tqdm(theory_gammas, desc="      "):
        try:
            theory_f_values.append(solve_zeroT_rs(g).f)
        except:
            theory_f_values.append(np.nan)
    theory_f_values = np.array(theory_f_values)
    
    print("\n[2/3] Value range...")
    all_f = []
    for gamma in tqdm(gamma_values, desc="      "):
        N = max(3, int(round(gamma * base_size)))
        C = C_full[:N, :base_size]
        lp = solve_minmax_lp(C, return_strategies=False)
        all_f.append((N * base_size) ** 0.25 * lp.value)
    
    f_min = min(min(all_f), np.nanmin(theory_f_values))
    f_max = max(max(all_f), np.nanmax(theory_f_values))
    
    print(f"\n[3/3] Rendering {len(gamma_values)} frames...")
    
    frames = []
    gamma_hist, f_hist = [], []
    
    for i, gamma in enumerate(tqdm(gamma_values, desc="      ")):
        N = max(3, int(round(gamma * base_size)))
        gamma_eff = N / base_size
        C = C_full[:N, :base_size]
        
        lp = solve_minmax_lp(C, return_strategies=True)
        f_scaled = (N * base_size) ** 0.25 * lp.value
        
        try:
            theory_f = solve_zeroT_rs(gamma_eff).f
        except:
            theory_f = f_scaled
        
        gamma_hist.append(gamma_eff)
        f_hist.append(f_scaled)
        
        frame = create_frame(
            gamma=gamma_eff, C=C, p=lp.p, q=lp.q,
            f_scaled=f_scaled, theory_f=theory_f,
            gamma_history=gamma_hist.copy(), f_history=f_hist.copy(),
            all_gammas=theory_gammas, all_theory_f=theory_f_values,
            frame_idx=i, total_frames=len(gamma_values),
            f_min=f_min, f_max=f_max, base_size=base_size
        )
        frames.append(frame)
    
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "nash_equilibrium_gamma_sweep.gif"
    )
    
    print("\n      Saving...")
    frames = [frames[0]] * 12 + frames + [frames[-1]] * 15
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=True)
    
    print(f"\n" + "=" * 60)
    print(f"  Done! {os.path.getsize(output_path)/1024/1024:.2f} MB, {len(frames)} frames")
    print("=" * 60)


if __name__ == "__main__":
    main()
