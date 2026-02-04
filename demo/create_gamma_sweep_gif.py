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
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
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
        'font.size': 13,
        'mathtext.fontset': 'cm',
    })
    
    fig = plt.figure(figsize=(14, 14), facecolor='#fafafa')
    
    N, M = C.shape
    
    # Colors
    c_min = '#3b82f6'
    c_max = '#ef4444'  
    c_theory = '#f59e0b'
    c_num = '#10b981'
    
    # === MATRIX SECTION (top 40%) ===
    ax_mat = fig.add_axes([0.02, 0.56, 0.96, 0.42])
    ax_mat.set_xlim(0, 100)
    ax_mat.set_ylim(0, 100)
    ax_mat.set_aspect('equal')
    ax_mat.axis('off')
    
    # Title (top center) - LARGER
    ax_mat.text(50, 98, r'$\min_{\mathbf{x}} \max_{\mathbf{y}} \, \mathbf{x}^{\top} C \, \mathbf{y}$',
                fontsize=36, ha='center', va='top', color='#1e293b', fontweight='bold')
    
    # Matrix (transposed for horizontal stretch)
    C_disp = C.T
    M_disp, N_disp = C_disp.shape
    
    base_dim = 38
    mat_height = base_dim
    mat_width = base_dim * gamma
    mat_width = np.clip(mat_width, base_dim * 0.5, base_dim * 1.6)
    
    mat_cx, mat_cy = 52, 48
    mat_left = mat_cx - mat_width / 2
    mat_bottom = mat_cy - mat_height / 2
    
    # Draw matrix with gradient border
    cmap = LinearSegmentedColormap.from_list('payoff',
        ['#1e40af', '#3b82f6', '#93c5fd', '#f8fafc', '#fca5a5', '#ef4444', '#b91c1c'], N=256)
    vmax = np.abs(C_disp).max()
    
    cell_w = mat_width / N_disp
    cell_h = mat_height / M_disp
    
    for i in range(M_disp):
        for j in range(N_disp):
            val = C_disp[i, j]
            color = cmap((val + vmax) / (2 * vmax))
            ax_mat.add_patch(Rectangle(
                (mat_left + j * cell_w, mat_bottom + (M_disp - 1 - i) * cell_h),
                cell_w, cell_h, facecolor=color, edgecolor='white', linewidth=0.05
            ))
    
    # Matrix border with shadow
    ax_mat.add_patch(Rectangle(
        (mat_left + 0.3, mat_bottom - 0.3), mat_width, mat_height,
        facecolor='none', edgecolor='#64748b', linewidth=4, alpha=0.3
    ))
    ax_mat.add_patch(Rectangle(
        (mat_left, mat_bottom), mat_width, mat_height,
        facecolor='none', edgecolor='#1e293b', linewidth=2.5
    ))
    
    # Strategy bars - top (minimizer x)
    x_strat = N * p
    bar_h = 5
    bar_y = mat_bottom + mat_height + 2
    max_x = max(4, x_strat.max() * 1.15)
    
    for j in range(N_disp):
        val = x_strat[j]
        if val > 0.02:
            h = (val / max_x) * bar_h
            ax_mat.add_patch(Rectangle(
                (mat_left + j * cell_w + cell_w * 0.1, bar_y),
                cell_w * 0.8, h,
                facecolor=c_min, alpha=0.9, edgecolor='white', linewidth=0.2
            ))
    
    # Strategy bars - left (maximizer y)  
    y_strat = M * q
    bar_w = 5
    bar_x = mat_left - bar_w - 2
    max_y = max(4, y_strat.max() * 1.15)
    
    for i in range(M_disp):
        val = y_strat[i]
        if val > 0.02:
            w = (val / max_y) * bar_w
            ax_mat.add_patch(Rectangle(
                (bar_x + bar_w - w, mat_bottom + (M_disp - 1 - i) * cell_h + cell_h * 0.1),
                w, cell_h * 0.8,
                facecolor=c_max, alpha=0.9, edgecolor='white', linewidth=0.2
            ))
    
    
    # Matrix size label (below matrix - centered) - LARGER
    ax_mat.text(mat_cx, mat_bottom - 4,
                rf'$C \in \mathbb{{R}}^{{{N} \times {M}}}$',
                fontsize=20, ha='center', va='top', color='#475569')
    
    # Strategy labels (positioned to avoid overlap) - LARGER
    # x label - far right of top bars
    ax_mat.text(mat_left + mat_width + 3, bar_y + bar_h * 0.5,
                rf'$x_i$', fontsize=22, ha='left', va='center', color=c_min, fontweight='bold')
    
    # Minimizer label - above bars, right side
    ax_mat.text(mat_left + mat_width, bar_y + bar_h + 2,
                rf'Min $(N\!=\!{N})$', fontsize=15, ha='right', va='bottom', color=c_min)
    
    # y label - far left of left bars
    ax_mat.text(bar_x - 2, mat_cy,
                rf'$y_j$', fontsize=22, ha='right', va='center', color=c_max, fontweight='bold')
    
    # Maximizer label - left side, top
    ax_mat.text(bar_x - 2, mat_bottom + mat_height + 1,
                rf'Max', fontsize=15, ha='right', va='bottom', color=c_max)
    ax_mat.text(bar_x - 2, mat_bottom + mat_height - 3,
                rf'$(M\!=\!{M})$', fontsize=13, ha='right', va='top', color=c_max)
    
    # === PLOT SECTION (bottom) - TALLER ===
    ax_plot = fig.add_axes([0.10, 0.06, 0.85, 0.46])
    
    # Subtle gradient background
    for i in range(20):
        alpha = 0.015 * (20 - i) / 20
        ax_plot.axhspan(f_min - 0.2 + i * (f_max - f_min + 0.4) / 20,
                        f_min - 0.2 + (i + 1) * (f_max - f_min + 0.4) / 20,
                        color='#6366f1', alpha=alpha, zorder=0)
    
    ax_plot.set_facecolor('#fefefe')
    
    # Fill under theory curve
    ax_plot.fill_between(all_gammas, all_theory_f, alpha=0.08, color=c_theory, zorder=1)
    
    # RS Theory curve with glow effect
    ax_plot.plot(all_gammas, all_theory_f, '-', color='white', linewidth=8, alpha=0.6, zorder=2)
    ax_plot.plot(all_gammas, all_theory_f, '-', color=c_theory, linewidth=4,
                 label=r'Replica Theory', zorder=3)
    
    # LP Numerical with markers
    if len(gamma_history) > 1:
        ax_plot.plot(gamma_history, f_history, '-', color=c_num, linewidth=2.5, alpha=0.7, zorder=4)
        ax_plot.scatter(gamma_history, f_history, s=50, color=c_num, 
                       edgecolors='white', linewidth=1, zorder=5, label=r'Numerical (LP)')
    
    # Animated current points
    pulse = 1 + 0.15 * np.sin(frame_idx * 0.4)
    
    # Theory point with ring effect
    for ring in [1.8, 1.4, 1.0]:
        ax_plot.scatter([gamma], [theory_f], s=200 * ring * pulse, 
                       color=c_theory, alpha=0.2 / ring, zorder=8)
    ax_plot.scatter([gamma], [theory_f], s=180 * pulse, color=c_theory, zorder=10,
                    edgecolors='white', linewidth=2.5, marker='D')
    
    # Numerical point with ring effect
    for ring in [1.8, 1.4, 1.0]:
        ax_plot.scatter([gamma], [f_scaled], s=220 * ring * pulse,
                       color=c_num, alpha=0.2 / ring, zorder=8)
    ax_plot.scatter([gamma], [f_scaled], s=200 * pulse, color=c_num, zorder=11,
                    edgecolors='white', linewidth=2.5)
    
    # Current gamma vertical line
    ax_plot.axvline(x=gamma, color='#818cf8', linestyle='--', alpha=0.6, linewidth=2, zorder=6)
    
    # PROMINENT gamma=1 line (no label)
    ax_plot.axvline(x=1.0, color='#dc2626', linestyle='-', alpha=0.9, linewidth=3.5, zorder=7)
    
    # Gamma badge (top right of plot)
    ax_plot.text(0.97, 0.95, rf'$\gamma = {gamma:.3f}$',
                transform=ax_plot.transAxes, fontsize=20, ha='right', va='top',
                color='#4338ca', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#eef2ff', 
                         edgecolor='#6366f1', linewidth=2.5))
    
    # Axis labels
    ax_plot.set_xlabel(r'$\gamma = N / M$', fontsize=18, fontweight='bold', labelpad=12)
    ax_plot.set_ylabel(r'$f(\gamma)$', fontsize=20, fontweight='bold', labelpad=10)
    
    # Title
    ax_plot.set_title(r'Nash Equilibrium Value', 
                      fontsize=22, fontweight='bold', pad=15, color='#1e293b')
    
    # Legend
    ax_plot.legend(loc='upper left', fontsize=14, framealpha=0.95, 
                   edgecolor='#e2e8f0', fancybox=True, borderpad=0.8,
                   handlelength=2.5, handletextpad=0.8)
    
    # Limits
    x_margin = 0.08
    y_margin = (f_max - f_min) * 0.18
    ax_plot.set_xlim(all_gammas.min() - x_margin, all_gammas.max() + x_margin)
    ax_plot.set_ylim(f_min - y_margin * 0.6, f_max + y_margin * 1.2)
    
    # Grid
    ax_plot.grid(True, alpha=0.35, linestyle='-', linewidth=0.5, color='#cbd5e1', zorder=0)
    ax_plot.tick_params(labelsize=13, length=6, width=1.5)
    
    # Spines styling
    for spine in ax_plot.spines.values():
        spine.set_linewidth(1.8)
        spine.set_color('#94a3b8')
    
    
    # Convert
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#fafafa', bbox_inches='tight', pad_inches=0.02)
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
