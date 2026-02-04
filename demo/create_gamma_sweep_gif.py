"""
Nash Equilibrium Animation for Random Matrix Games.

- Large matrix for better RS theory convergence
- Matrix stretches horizontally with gamma
- Dynamic comparison visualization

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
from matplotlib.patches import Rectangle, FancyBboxPatch, Circle
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
    theory_history: list,
    all_gammas: np.ndarray,
    all_theory_f: np.ndarray,
    frame_idx: int,
    total_frames: int,
    f_min: float,
    f_max: float,
) -> Image.Image:
    """Create a single frame."""
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 12,
        'mathtext.fontset': 'cm',
    })
    
    fig = plt.figure(figsize=(18, 12), facecolor='#fafafa')
    
    N, M = C.shape
    
    # Colors
    c_min = '#2563eb'
    c_max = '#dc2626'  
    c_theory = '#f59e0b'
    c_num = '#059669'
    
    # === Layout: Matrix top (45%), Plot bottom (55%) ===
    
    # MATRIX SECTION
    ax_mat = fig.add_axes([0.02, 0.48, 0.96, 0.50])
    ax_mat.set_xlim(0, 100)
    ax_mat.set_ylim(0, 100)
    ax_mat.set_aspect('equal')
    ax_mat.axis('off')
    
    # Title
    ax_mat.text(50, 98, r'$\min_{\mathbf{x} \in \Delta_N} \max_{\mathbf{y} \in \Delta_M} \, \mathbf{x}^{\!\top} C \, \mathbf{y}$',
                fontsize=26, ha='center', va='top', color='#0f172a', fontweight='bold')
    
    # Matrix (transposed for horizontal stretch)
    C_disp = C.T
    M_disp, N_disp = C_disp.shape
    
    # Matrix size
    base_dim = 36
    mat_height = base_dim
    mat_width = base_dim * gamma
    mat_width = np.clip(mat_width, base_dim * 0.45, base_dim * 1.65)
    
    mat_cx, mat_cy = 50, 52
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
        facecolor='none', edgecolor='#1e293b', linewidth=2.5
    ))
    
    # Strategy bars
    x_strat = N * p
    bar_h = 5
    bar_y = mat_bottom + mat_height + 2.5
    max_x = max(4, x_strat.max() * 1.1)
    
    for j in range(N_disp):
        val = x_strat[j]
        if val > 0.02:
            ax_mat.add_patch(Rectangle(
                (mat_left + j * cell_w + cell_w * 0.08, bar_y),
                cell_w * 0.84, (val / max_x) * bar_h,
                facecolor=c_min, alpha=0.85
            ))
    
    y_strat = M * q
    bar_w = 5
    bar_x = mat_left - bar_w - 2.5
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
    
    # Labels
    ax_mat.add_patch(FancyBboxPatch((2, 86), 18, 10,
        boxstyle="round,pad=0.01,rounding_size=0.4",
        facecolor='#eef2ff', edgecolor='#6366f1', linewidth=2.5))
    ax_mat.text(11, 91, rf'$\gamma = {gamma:.3f}$',
                fontsize=18, ha='center', va='center', color='#4338ca', fontweight='bold')
    
    ax_mat.text(mat_cx, mat_bottom - 5, rf'$C \in \mathbb{{R}}^{{{N} \times {M}}}$',
                fontsize=16, ha='center', va='top', color='#334155')
    
    ax_mat.text(mat_left + mat_width + 3, bar_y + bar_h / 2,
                rf'$x_i$', fontsize=16, ha='left', va='center', color=c_min, fontweight='bold')
    ax_mat.text(bar_x - 2, mat_cy,
                rf'$y_j$', fontsize=16, ha='right', va='center', color=c_max, fontweight='bold')
    
    ax_mat.text(mat_cx, bar_y + bar_h + 3, rf'Minimizer $(N={N})$',
                fontsize=14, ha='center', va='bottom', color=c_min, fontweight='bold')
    ax_mat.text(bar_x - 3, mat_bottom + mat_height + 2, rf'Max',
                fontsize=12, ha='right', va='bottom', color=c_max, fontweight='bold')
    ax_mat.text(bar_x - 3, mat_bottom + mat_height - 2, rf'$(M={M})$',
                fontsize=11, ha='right', va='top', color=c_max)
    
    # Stats
    rho_x = support_fraction(p)
    rho_y = support_fraction(q)
    error = abs(f_scaled - theory_f) / (abs(theory_f) + 1e-10) * 100
    
    ax_mat.add_patch(FancyBboxPatch((77, 68), 21, 28,
        boxstyle="round,pad=0.01,rounding_size=0.4",
        facecolor='white', edgecolor='#cbd5e1', linewidth=2))
    
    ax_mat.text(79, 93, 'Statistics', fontsize=13, ha='left', color='#334155', fontweight='bold')
    stats = [
        (rf'$f_{{\mathrm{{LP}}}} = {f_scaled:.4f}$', c_num, 14),
        (rf'$f_{{\mathrm{{RS}}}} = {theory_f:.4f}$', c_theory, 14),
        (f'Error: {error:.2f}%', '#dc2626' if error > 2 else '#16a34a', 13),
        (rf'$\rho_x = {rho_x*100:.0f}\%$', c_min, 13),
        (rf'$\rho_y = {rho_y*100:.0f}\%$', c_max, 13),
    ]
    for i, (txt, col, fs) in enumerate(stats):
        ax_mat.text(79, 88 - i * 4.2, txt, fontsize=fs, ha='left', color=col)
    
    # === PLOT SECTION (taller) ===
    ax_plot = fig.add_axes([0.08, 0.06, 0.88, 0.38])
    ax_plot.set_facecolor('#fefefe')
    
    # Background gradient effect
    for i in range(10):
        ax_plot.axhspan(f_min - 0.1 + i * (f_max - f_min + 0.3) / 10,
                        f_min - 0.1 + (i + 1) * (f_max - f_min + 0.3) / 10,
                        color='#f8fafc', alpha=0.5 - i * 0.04, zorder=0)
    
    # RS Theory - thick prominent line with glow
    ax_plot.fill_between(all_gammas, all_theory_f, alpha=0.15, color=c_theory, zorder=1)
    ax_plot.plot(all_gammas, all_theory_f, '-', color=c_theory, linewidth=5,
                 label=r'RS Theory $f(\gamma)$', alpha=0.95, zorder=2,
                 path_effects=[pe.Stroke(linewidth=8, foreground='white'), pe.Normal()])
    
    # LP Numerical
    if len(gamma_history) > 1:
        ax_plot.plot(gamma_history, f_history, 'o-', color=c_num, linewidth=2.5,
                     markersize=6, label='LP Numerical', alpha=0.9, zorder=3)
    
    # Current points with animation effect
    # Pulsing circle effect
    pulse = 1 + 0.15 * np.sin(frame_idx * 0.5)
    ax_plot.scatter([gamma], [f_scaled], s=350 * pulse, color=c_num, zorder=10,
                    edgecolors='white', linewidth=3, alpha=0.9)
    ax_plot.scatter([gamma], [theory_f], s=250 * pulse, color=c_theory, zorder=9,
                    marker='D', edgecolors='white', linewidth=2.5)
    
    # Connecting line showing error
    ax_plot.plot([gamma, gamma], [theory_f, f_scaled], 
                 color='#6366f1', linestyle='-', linewidth=2, alpha=0.6, zorder=8)
    
    # Winner indicator
    diff = f_scaled - theory_f
    if abs(diff) > 0.001:
        winner_y = (f_scaled + theory_f) / 2
        if diff > 0:
            ax_plot.annotate('', xy=(gamma + 0.03, f_scaled - 0.02),
                            xytext=(gamma + 0.03, theory_f + 0.02),
                            arrowprops=dict(arrowstyle='->', color=c_num, lw=2))
        else:
            ax_plot.annotate('', xy=(gamma + 0.03, theory_f - 0.02),
                            xytext=(gamma + 0.03, f_scaled + 0.02),
                            arrowprops=dict(arrowstyle='->', color=c_theory, lw=2))
    
    # Error band visualization
    if len(gamma_history) > 2:
        errors = np.array(f_history) - np.array([solve_zeroT_rs(g).f if g > 0.3 else f_history[i] 
                                                  for i, g in enumerate(gamma_history)])
        ax_plot.fill_between(gamma_history, 
                            np.array(f_history) - np.abs(errors) * 0.3,
                            np.array(f_history) + np.abs(errors) * 0.3,
                            alpha=0.1, color=c_num, zorder=1)
    
    # Vertical line
    ax_plot.axvline(x=gamma, color='#6366f1', linestyle='--', alpha=0.5, linewidth=2, zorder=5)
    
    # Gamma = 1 marker
    ax_plot.axvline(x=1.0, color='#94a3b8', linestyle=':', alpha=0.6, linewidth=1.5)
    ax_plot.text(1.0, f_max + 0.03, r'$\gamma=1$', fontsize=12, ha='center', color='#64748b')
    
    # Current gamma annotation
    ax_plot.annotate(rf'$\gamma={gamma:.2f}$', 
                    xy=(gamma, f_min - 0.05), fontsize=14, ha='center', color='#4338ca',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#eef2ff', edgecolor='#6366f1', linewidth=1.5))
    
    # Dynamic comparison text
    if error < 1:
        match_text = "Excellent Match!"
        match_color = '#16a34a'
    elif error < 3:
        match_text = "Good Match"
        match_color = '#65a30d'
    else:
        match_text = f"Δ = {error:.1f}%"
        match_color = '#ea580c'
    
    ax_plot.text(0.98, 0.95, match_text, transform=ax_plot.transAxes,
                fontsize=16, ha='right', va='top', color=match_color, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor=match_color, linewidth=2))
    
    # Labels
    ax_plot.set_xlabel(r'$\gamma = N / M$', fontsize=16, fontweight='bold', labelpad=8)
    ax_plot.set_ylabel(r'Scaled Game Value $f(\gamma)$', fontsize=16, fontweight='bold', labelpad=8)
    ax_plot.set_title(r'Nash Equilibrium: RS Theory vs LP Numerical', 
                      fontsize=20, fontweight='bold', pad=12, color='#0f172a')
    
    # Legend
    ax_plot.legend(loc='upper left', fontsize=14, framealpha=0.95, 
                   edgecolor='#e2e8f0', fancybox=True, borderpad=1)
    
    # Limits
    x_margin = 0.08
    y_margin = (f_max - f_min) * 0.18
    ax_plot.set_xlim(all_gammas.min() - x_margin, all_gammas.max() + x_margin)
    ax_plot.set_ylim(f_min - y_margin, f_max + y_margin)
    
    # Grid
    ax_plot.grid(True, alpha=0.4, linestyle='-', linewidth=0.6, color='#cbd5e1', zorder=0)
    ax_plot.tick_params(labelsize=12)
    
    for spine in ax_plot.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#94a3b8')
    
    # Progress bar
    prog = (frame_idx + 1) / total_frames
    prog_ax = fig.add_axes([0.08, 0.015, 0.88, 0.015])
    prog_ax.barh(0, prog, height=1, color='#6366f1', alpha=0.7)
    prog_ax.barh(0, 1, height=1, color='#e2e8f0', alpha=0.4, zorder=0)
    prog_ax.text(prog, 0.5, f' {prog*100:.0f}%', fontsize=10, va='center', color='white' if prog > 0.1 else '#6366f1')
    prog_ax.set_xlim(0, 1)
    prog_ax.axis('off')
    
    # Convert
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#fafafa', bbox_inches='tight', pad_inches=0.02)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    return img


def main():
    print("=" * 65)
    print("  Nash Equilibrium Animation - Large Scale")
    print("=" * 65)
    
    seed = 42
    base_size = 100  # Large matrix for better RS convergence
    gamma_values = np.concatenate([
        np.linspace(0.4, 1.0, 18),
        np.linspace(1.0, 1.6, 18)
    ])
    
    rng = np.random.default_rng(seed)
    max_N = int(2.0 * base_size) + 10
    C_full = rng.standard_normal(size=(max_N, base_size))
    
    print(f"\n  Matrix size: up to {max_N} × {base_size}")
    
    print("\n[1/3] RS theory curve...")
    theory_gammas = np.linspace(0.35, 1.7, 120)
    theory_f_values = []
    for g in tqdm(theory_gammas, desc="      "):
        try:
            theory_f_values.append(solve_zeroT_rs(g).f)
        except:
            theory_f_values.append(np.nan)
    theory_f_values = np.array(theory_f_values)
    
    print("\n[2/3] Computing LP solutions...")
    all_f, all_theory = [], []
    for gamma in tqdm(gamma_values, desc="      "):
        N = max(3, int(round(gamma * base_size)))
        C = C_full[:N, :base_size]
        lp = solve_minmax_lp(C, return_strategies=False)
        f_val = (N * base_size) ** 0.25 * lp.value
        all_f.append(f_val)
        try:
            all_theory.append(solve_zeroT_rs(N / base_size).f)
        except:
            all_theory.append(f_val)
    
    f_min = min(min(all_f), np.nanmin(theory_f_values))
    f_max = max(max(all_f), np.nanmax(theory_f_values))
    
    print(f"\n[3/3] Rendering {len(gamma_values)} frames...")
    
    frames = []
    gamma_hist, f_hist, theory_hist = [], [], []
    
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
        theory_hist.append(theory_f)
        
        frame = create_frame(
            gamma=gamma_eff, C=C, p=lp.p, q=lp.q,
            f_scaled=f_scaled, theory_f=theory_f,
            gamma_history=gamma_hist.copy(), 
            f_history=f_hist.copy(),
            theory_history=theory_hist.copy(),
            all_gammas=theory_gammas, all_theory_f=theory_f_values,
            frame_idx=i, total_frames=len(gamma_values),
            f_min=f_min, f_max=f_max
        )
        frames.append(frame)
    
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets", "nash_equilibrium_gamma_sweep.gif"
    )
    
    print("\n      Saving GIF...")
    frames = [frames[0]] * 10 + frames + [frames[-1]] * 12
    frames[0].save(output_path, save_all=True, append_images=frames[1:],
                   duration=100, loop=0, optimize=True)
    
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    
    # Calculate average error
    avg_error = np.mean(np.abs(np.array(all_f) - np.array(all_theory)) / np.array(all_theory)) * 100
    
    print(f"\n" + "=" * 65)
    print(f"  ✓ Done!")
    print(f"    File size: {size_mb:.2f} MB")
    print(f"    Frames: {len(frames)}")
    print(f"    Matrix: {base_size}×{base_size} at γ=1")
    print(f"    Average LP-RS error: {avg_error:.2f}%")
    print("=" * 65)


if __name__ == "__main__":
    main()
