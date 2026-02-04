"""
Nash Equilibrium Animation for Random Matrix Games.
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
from src.zeroT_lp import solve_minmax_lp


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
) -> Image.Image:
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 14,
        'mathtext.fontset': 'cm',
    })
    
    fig = plt.figure(figsize=(16, 11), facecolor='white')
    
    N, M = C.shape
    
    # Colors
    c_min = '#2563eb'
    c_max = '#dc2626'  
    c_theory = '#f59e0b'
    c_num = '#059669'
    
    # === MATRIX SECTION (top 52%) ===
    ax_mat = fig.add_axes([0.01, 0.44, 0.98, 0.55])
    ax_mat.set_xlim(0, 100)
    ax_mat.set_ylim(0, 100)
    ax_mat.set_aspect('equal')
    ax_mat.axis('off')
    
    # Large Title
    ax_mat.text(50, 97, r'$\min_{\mathbf{x} \in \Delta_N} \max_{\mathbf{y} \in \Delta_M} \, \mathbf{x}^{\!\top} C \, \mathbf{y}$',
                fontsize=32, ha='center', va='top', color='#0f172a', fontweight='bold')
    
    # Gamma badge (large)
    ax_mat.add_patch(FancyBboxPatch((3, 82), 22, 12,
        boxstyle="round,pad=0.01,rounding_size=0.5",
        facecolor='#eef2ff', edgecolor='#6366f1', linewidth=3))
    ax_mat.text(14, 88, rf'$\gamma = {gamma:.3f}$',
                fontsize=22, ha='center', va='center', color='#4338ca', fontweight='bold')
    
    # Matrix (transposed)
    C_disp = C.T
    M_disp, N_disp = C_disp.shape
    
    base_dim = 40
    mat_height = base_dim
    mat_width = base_dim * gamma
    mat_width = np.clip(mat_width, base_dim * 0.45, base_dim * 1.65)
    
    mat_cx, mat_cy = 50, 48
    mat_left = mat_cx - mat_width / 2
    mat_bottom = mat_cy - mat_height / 2
    
    # Draw matrix
    cmap = LinearSegmentedColormap.from_list('payoff',
        ['#1e3a8a', '#3b82f6', '#93c5fd', '#ffffff', '#fca5a5', '#ef4444', '#991b1b'], N=256)
    vmax = np.abs(C_disp).max()
    
    cell_w = mat_width / N_disp
    cell_h = mat_height / M_disp
    
    for i in range(M_disp):
        for j in range(N_disp):
            val = C_disp[i, j]
            color = cmap((val + vmax) / (2 * vmax))
            ax_mat.add_patch(Rectangle(
                (mat_left + j * cell_w, mat_bottom + (M_disp - 1 - i) * cell_h),
                cell_w, cell_h, facecolor=color, edgecolor='white', linewidth=0.08
            ))
    
    ax_mat.add_patch(Rectangle(
        (mat_left, mat_bottom), mat_width, mat_height,
        facecolor='none', edgecolor='#1e293b', linewidth=3
    ))
    
    # Strategy bars - top (x)
    x_strat = N * p
    bar_h = 6
    bar_y = mat_bottom + mat_height + 3
    max_x = max(4, x_strat.max() * 1.1)
    
    for j in range(N_disp):
        val = x_strat[j]
        if val > 0.02:
            ax_mat.add_patch(Rectangle(
                (mat_left + j * cell_w + cell_w * 0.08, bar_y),
                cell_w * 0.84, (val / max_x) * bar_h,
                facecolor=c_min, alpha=0.85
            ))
    
    # Strategy bars - left (y)
    y_strat = M * q
    bar_w = 6
    bar_x = mat_left - bar_w - 3
    max_y = max(4, y_strat.max() * 1.1)
    
    for i in range(M_disp):
        val = y_strat[i]
        if val > 0.02:
            ax_mat.add_patch(Rectangle(
                (bar_x + bar_w - (val / max_y) * bar_w, 
                 mat_bottom + (M_disp - 1 - i) * cell_h + cell_h * 0.08),
                (val / max_y) * bar_w, cell_h * 0.84,
                facecolor=c_max, alpha=0.85
            ))
    
    # Labels (large)
    ax_mat.text(mat_cx, mat_bottom - 5, rf'$C \in \mathbb{{R}}^{{{N} \times {M}}}$',
                fontsize=20, ha='center', va='top', color='#334155', fontweight='bold')
    
    ax_mat.text(mat_left + mat_width + 4, bar_y + bar_h / 2,
                rf'$x_i$', fontsize=20, ha='left', va='center', color=c_min, fontweight='bold')
    ax_mat.text(bar_x - 3, mat_cy,
                rf'$y_j$', fontsize=20, ha='right', va='center', color=c_max, fontweight='bold')
    
    ax_mat.text(mat_cx, bar_y + bar_h + 4, rf'Minimizer $(N={N})$',
                fontsize=16, ha='center', va='bottom', color=c_min, fontweight='bold')
    ax_mat.text(bar_x - 4, mat_cy + 12, rf'Maximizer',
                fontsize=14, ha='right', va='center', color=c_max, fontweight='bold')
    ax_mat.text(bar_x - 4, mat_cy + 5, rf'$(M={M})$',
                fontsize=13, ha='right', va='center', color=c_max)
    
    # === PLOT SECTION (bottom 44%) ===
    ax_plot = fig.add_axes([0.08, 0.10, 0.88, 0.32])
    ax_plot.set_facecolor('#fefefe')
    
    # RS Theory curve
    ax_plot.fill_between(all_gammas, all_theory_f, alpha=0.12, color=c_theory, zorder=1)
    ax_plot.plot(all_gammas, all_theory_f, '-', color=c_theory, linewidth=4.5,
                 label=r'Replica Theory', alpha=0.95, zorder=2,
                 path_effects=[pe.Stroke(linewidth=7, foreground='white'), pe.Normal()])
    
    # LP Numerical
    if len(gamma_history) > 1:
        ax_plot.plot(gamma_history, f_history, 'o-', color=c_num, linewidth=2.5,
                     markersize=7, label=r'Numerical (LP)', alpha=0.9, zorder=3)
    
    # Current points
    pulse = 1 + 0.12 * np.sin(frame_idx * 0.5)
    ax_plot.scatter([gamma], [f_scaled], s=300 * pulse, color=c_num, zorder=10,
                    edgecolors='white', linewidth=3)
    ax_plot.scatter([gamma], [theory_f], s=220 * pulse, color=c_theory, zorder=9,
                    marker='D', edgecolors='white', linewidth=2.5)
    
    # Vertical line at current gamma
    ax_plot.axvline(x=gamma, color='#6366f1', linestyle='--', alpha=0.5, linewidth=2, zorder=5)
    
    # PROMINENT gamma=1 line
    ax_plot.axvline(x=1.0, color='#ef4444', linestyle='-', alpha=0.8, linewidth=3, zorder=4)
    ax_plot.text(1.0, f_max + (f_max - f_min) * 0.08, r'$\gamma=1$', 
                fontsize=18, ha='center', color='#ef4444', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='#ef4444', linewidth=2))
    
    # Labels (large)
    ax_plot.set_xlabel(r'$\gamma = N / M$', fontsize=20, fontweight='bold', labelpad=10)
    ax_plot.set_ylabel(r'$f(\gamma)$', fontsize=22, fontweight='bold', labelpad=8)
    ax_plot.set_title(r'Nash Equilibrium Value', 
                      fontsize=24, fontweight='bold', pad=12, color='#0f172a')
    
    # Legend (large)
    ax_plot.legend(loc='upper left', fontsize=16, framealpha=0.95, 
                   edgecolor='#cbd5e1', fancybox=True, borderpad=1)
    
    # Limits
    x_margin = 0.06
    y_margin = (f_max - f_min) * 0.22
    ax_plot.set_xlim(all_gammas.min() - x_margin, all_gammas.max() + x_margin)
    ax_plot.set_ylim(f_min - y_margin * 0.5, f_max + y_margin)
    
    # Grid
    ax_plot.grid(True, alpha=0.4, linestyle='-', linewidth=0.6, color='#cbd5e1', zorder=0)
    ax_plot.tick_params(labelsize=14)
    
    for spine in ax_plot.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#94a3b8')
    
    # Progress bar BELOW x-axis label
    prog = (frame_idx + 1) / total_frames
    prog_ax = fig.add_axes([0.08, 0.02, 0.88, 0.018])
    prog_ax.barh(0, prog, height=1, color='#6366f1', alpha=0.7)
    prog_ax.barh(0, 1, height=1, color='#e2e8f0', alpha=0.4, zorder=0)
    prog_ax.set_xlim(0, 1)
    prog_ax.axis('off')
    
    # Convert
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white', bbox_inches='tight', pad_inches=0.01)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    return img


def main():
    print("=" * 60)
    print("  Nash Equilibrium Animation")
    print("=" * 60)
    
    seed = 42
    base_size = 100
    gamma_values = np.concatenate([
        np.linspace(0.4, 1.0, 18),
        np.linspace(1.0, 1.6, 18)
    ])
    
    rng = np.random.default_rng(seed)
    max_N = int(2.0 * base_size) + 10
    C_full = rng.standard_normal(size=(max_N, base_size))
    
    print(f"\n  Matrix: up to {max_N} × {base_size}")
    
    print("\n[1/3] RS theory...")
    theory_gammas = np.linspace(0.35, 1.7, 100)
    theory_f_values = []
    for g in tqdm(theory_gammas, desc="      "):
        try:
            theory_f_values.append(solve_zeroT_rs(g).f)
        except:
            theory_f_values.append(np.nan)
    theory_f_values = np.array(theory_f_values)
    
    print("\n[2/3] LP solutions...")
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
            gamma_history=gamma_hist.copy(), 
            f_history=f_hist.copy(),
            all_gammas=theory_gammas, all_theory_f=theory_f_values,
            frame_idx=i, total_frames=len(gamma_values),
            f_min=f_min, f_max=f_max
        )
        frames.append(frame)
    
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "nash_equilibrium_gamma_sweep.gif"
    )
    
    print("\n      Saving...")
    frames = [frames[0]] * 10 + frames + [frames[-1]] * 12
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=True)
    
    print(f"\n" + "=" * 60)
    print(f"  Done! {os.path.getsize(output_path)/1024/1024:.2f} MB, {len(frames)} frames")
    print("=" * 60)


if __name__ == "__main__":
    main()
