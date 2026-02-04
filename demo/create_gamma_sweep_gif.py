"""
Create an animated GIF showing how Nash equilibrium value and mixed strategies
change as gamma (N/M ratio) varies for a random matrix game.

- Matrix is SQUARE when gamma = 1
- Matrix size visually changes with gamma

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
    })
    
    fig = plt.figure(figsize=(16, 10), facecolor='#f8fafc')
    
    N, M = C.shape
    
    # Modern color palette
    color_min = '#3b82f6'   # Blue
    color_max = '#ef4444'   # Red
    color_theory = '#f59e0b'  # Amber
    color_numerical = '#10b981'  # Emerald
    color_bg = '#f1f5f9'
    
    # === Layout ===
    # Top section: Matrix visualization (larger area)
    # Bottom section: Value plot
    
    ax_matrix = fig.add_axes([0.08, 0.38, 0.84, 0.55])
    ax_matrix.set_facecolor('#f8fafc')
    ax_matrix.set_xlim(0, 100)
    ax_matrix.set_ylim(0, 100)
    ax_matrix.set_aspect('equal')
    ax_matrix.axis('off')
    
    # === Calculate matrix display size ===
    # At gamma=1, matrix should be SQUARE
    # Matrix width is fixed, height scales with gamma
    max_size = 42  # Maximum dimension at gamma=1
    
    mat_width = max_size
    mat_height = max_size * gamma  # Height = width * gamma, so square at gamma=1
    
    # Limit maximum height
    if mat_height > max_size * 1.8:
        mat_height = max_size * 1.8
    
    # Center position for matrix
    center_x = 50
    center_y = 55
    
    # Margins for strategy bars
    bar_margin = 8
    
    mat_left = center_x - mat_width/2
    mat_bottom = center_y - mat_height/2
    
    # === Draw payoff matrix ===
    cmap = LinearSegmentedColormap.from_list('payoff', 
        ['#1e3a8a', '#3b82f6', '#93c5fd', '#ffffff', '#fca5a5', '#ef4444', '#991b1b'], N=256)
    
    vmax = np.abs(C).max()
    cell_w = mat_width / M
    cell_h = mat_height / N
    
    # Matrix cells
    for i in range(N):
        for j in range(M):
            val = C[i, j]
            color = cmap((val + vmax) / (2 * vmax))
            rect = Rectangle(
                (mat_left + j * cell_w, mat_bottom + (N - 1 - i) * cell_h),
                cell_w, cell_h,
                facecolor=color, edgecolor='white', linewidth=0.15
            )
            ax_matrix.add_patch(rect)
    
    # Matrix border with shadow effect
    shadow = Rectangle(
        (mat_left + 0.5, mat_bottom - 0.5), mat_width, mat_height,
        facecolor='none', edgecolor='#94a3b8', linewidth=3, alpha=0.3
    )
    ax_matrix.add_patch(shadow)
    border = Rectangle(
        (mat_left, mat_bottom), mat_width, mat_height,
        facecolor='none', edgecolor='#1e293b', linewidth=2.5
    )
    ax_matrix.add_patch(border)
    
    # === Minimizer strategy (left side) ===
    x_strategy = N * p
    bar_w = 5
    bar_left = mat_left - bar_w - 3
    max_x = max(4, x_strategy.max() * 1.1)
    
    for i in range(N):
        val = x_strategy[i]
        h = cell_h * 0.9
        w = (val / max_x) * bar_w
        y = mat_bottom + (N - 1 - i) * cell_h + cell_h * 0.05
        
        # Bar
        if val > 0.05:
            rect = Rectangle(
                (bar_left + bar_w - w, y), w, h,
                facecolor=color_min, edgecolor='none', alpha=0.85
            )
            ax_matrix.add_patch(rect)
    
    # === Maximizer strategy (top) ===
    y_strategy = M * q
    bar_h = 4
    bar_bottom = mat_bottom + mat_height + 2
    max_y = max(4, y_strategy.max() * 1.1)
    
    for j in range(M):
        val = y_strategy[j]
        w = cell_w * 0.9
        h = (val / max_y) * bar_h
        x = mat_left + j * cell_w + cell_w * 0.05
        
        if val > 0.05:
            rect = Rectangle(
                (x, bar_bottom), w, h,
                facecolor=color_max, edgecolor='none', alpha=0.85
            )
            ax_matrix.add_patch(rect)
    
    # === Labels (positioned to avoid overlap) ===
    # Title
    ax_matrix.text(50, 98, r'Min-Max Game: $\min_{\mathbf{x}} \max_{\mathbf{y}} \; \mathbf{x}^\top C \, \mathbf{y}$',
                   fontsize=20, ha='center', va='top', fontweight='bold', color='#0f172a',
                   path_effects=[pe.withStroke(linewidth=3, foreground='white')])
    
    # Gamma badge (top right, clear area)
    gamma_box = FancyBboxPatch(
        (78, 88), 18, 10,
        boxstyle="round,pad=0.02,rounding_size=0.5",
        facecolor='#eef2ff', edgecolor='#6366f1', linewidth=2,
        transform=ax_matrix.transData
    )
    ax_matrix.add_patch(gamma_box)
    ax_matrix.text(87, 93, f'γ = {gamma:.3f}', fontsize=14, ha='center', va='center',
                   fontweight='bold', color='#4338ca')
    
    # Matrix label (below matrix)
    ax_matrix.text(center_x, mat_bottom - 4, f'Payoff Matrix C',
                   fontsize=13, ha='center', va='top', color='#475569', fontweight='bold')
    ax_matrix.text(center_x, mat_bottom - 8.5, f'{N} × {M}',
                   fontsize=12, ha='center', va='top', color='#64748b')
    
    # Strategy labels (positioned outside bars)
    ax_matrix.text(bar_left - 1, center_y, f'$x_i$',
                   fontsize=13, ha='right', va='center', color=color_min, fontweight='bold')
    ax_matrix.text(bar_left - 1, center_y - 5, f'N={N}',
                   fontsize=10, ha='right', va='center', color='#64748b')
    
    ax_matrix.text(center_x, bar_bottom + bar_h + 2.5, f'$y_j$  (M={M})',
                   fontsize=13, ha='center', va='bottom', color=color_max, fontweight='bold')
    
    # === Stats panel (left side, below strategy) ===
    rho_x = support_fraction(p)
    rho_y = support_fraction(q)
    error = abs(f_scaled - theory_f) / (abs(theory_f) + 1e-10) * 100
    
    stats_x = 6
    stats_y = 35
    
    stats_box = FancyBboxPatch(
        (2, 15), 22, 25,
        boxstyle="round,pad=0.02,rounding_size=0.3",
        facecolor='white', edgecolor='#e2e8f0', linewidth=1.5,
        transform=ax_matrix.transData
    )
    ax_matrix.add_patch(stats_box)
    
    ax_matrix.text(stats_x, stats_y, 'Statistics', fontsize=11, ha='left', va='center',
                   fontweight='bold', color='#334155')
    
    stats_items = [
        (f'LP:  {f_scaled:.4f}', color_numerical),
        (f'RS:  {theory_f:.4f}', color_theory),
        (f'Err: {error:.2f}%', '#dc2626' if error > 3 else '#16a34a'),
        (f'ρx:  {rho_x:.0%}', color_min),
        (f'ρy:  {rho_y:.0%}', color_max),
    ]
    for i, (text, color) in enumerate(stats_items):
        ax_matrix.text(stats_x, stats_y - 4 - i * 3.5, text,
                       fontsize=10, ha='left', va='center', color=color, fontfamily='monospace')
    
    # === Colorbar (right side) ===
    cbar_x = 94
    cbar_y = mat_bottom
    cbar_h = mat_height * 0.6
    cbar_w = 2
    
    for i in range(20):
        val = -vmax + (2 * vmax) * i / 19
        color = cmap((val + vmax) / (2 * vmax))
        rect = Rectangle(
            (cbar_x, cbar_y + i * cbar_h / 20), cbar_w, cbar_h / 20,
            facecolor=color, edgecolor='none'
        )
        ax_matrix.add_patch(rect)
    
    ax_matrix.text(cbar_x + cbar_w + 1, cbar_y, f'{-vmax:.1f}', fontsize=8, va='center', color='#64748b')
    ax_matrix.text(cbar_x + cbar_w + 1, cbar_y + cbar_h/2, '0', fontsize=8, va='center', color='#64748b')
    ax_matrix.text(cbar_x + cbar_w + 1, cbar_y + cbar_h, f'+{vmax:.1f}', fontsize=8, va='center', color='#64748b')
    
    # === Bottom: Value Plot ===
    ax_plot = fig.add_axes([0.1, 0.08, 0.8, 0.25])
    ax_plot.set_facecolor('white')
    
    # Plot RS theory (smooth curve)
    ax_plot.fill_between(all_gammas, all_theory_f, alpha=0.15, color=color_theory)
    ax_plot.plot(all_gammas, all_theory_f, '-', color=color_theory,
                 linewidth=3.5, label='RS Theory', alpha=0.95,
                 path_effects=[pe.Stroke(linewidth=5, foreground='white'), pe.Normal()])
    
    # Plot numerical
    if len(gamma_history) > 1:
        ax_plot.plot(gamma_history, f_history, 'o-', color=color_numerical,
                     linewidth=2, markersize=5, label='LP Numerical', alpha=0.9)
    
    # Current points (prominent)
    ax_plot.scatter([gamma], [f_scaled], s=250, color=color_numerical,
                    zorder=10, edgecolors='white', linewidth=3)
    ax_plot.scatter([gamma], [theory_f], s=180, color=color_theory,
                    zorder=9, marker='D', edgecolors='white', linewidth=2)
    
    # Vertical line at current gamma
    ax_plot.axvline(x=gamma, color='#6366f1', linestyle='--', alpha=0.4, linewidth=1.5)
    
    # Mark gamma = 1
    ax_plot.axvline(x=1.0, color='#94a3b8', linestyle=':', alpha=0.6, linewidth=1)
    ax_plot.text(1.0, f_max + 0.02, 'γ=1', fontsize=9, ha='center', color='#64748b')
    
    # Axis labels
    ax_plot.set_xlabel(r'$\gamma = N/M$', fontsize=13, fontweight='bold', labelpad=5)
    ax_plot.set_ylabel(r'$f(\gamma)$', fontsize=13, fontweight='bold', labelpad=5)
    ax_plot.set_title('Nash Equilibrium Value Comparison', fontsize=14, fontweight='bold', pad=8, color='#1e293b')
    
    # Legend
    ax_plot.legend(loc='upper right', fontsize=11, framealpha=0.95,
                   edgecolor='#e2e8f0', fancybox=True, borderpad=0.8)
    
    # Axis limits with proper padding
    x_pad = 0.08
    y_pad = (f_max - f_min) * 0.12
    ax_plot.set_xlim(all_gammas.min() - x_pad, all_gammas.max() + x_pad)
    ax_plot.set_ylim(f_min - y_pad, f_max + y_pad)
    
    # Grid
    ax_plot.grid(True, alpha=0.4, linestyle='-', linewidth=0.5, color='#cbd5e1')
    ax_plot.tick_params(labelsize=10)
    
    # Spines
    for spine in ax_plot.spines.values():
        spine.set_color('#cbd5e1')
    
    # Progress indicator
    progress = (frame_idx + 1) / total_frames
    prog_ax = fig.add_axes([0.1, 0.02, 0.8, 0.012])
    prog_ax.set_xlim(0, 1)
    prog_ax.barh(0, progress, height=1, color='#6366f1', alpha=0.7)
    prog_ax.barh(0, 1, height=1, color='#e2e8f0', alpha=0.5, zorder=0)
    prog_ax.axis('off')
    
    # Convert to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#f8fafc',
                edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    return img


def main():
    print("=" * 60)
    print("  Creating Nash Equilibrium Animation")
    print("  Matrix is SQUARE at γ = 1")
    print("=" * 60)
    
    # Parameters - M = N at gamma = 1
    seed = 42
    base_size = 60  # At gamma=1, matrix is base_size × base_size
    gamma_values = np.concatenate([
        np.linspace(0.35, 1.0, 26),
        np.linspace(1.0, 1.65, 24)
    ])
    
    rng = np.random.default_rng(seed)
    
    max_N = int(2.0 * base_size) + 10
    C_full = rng.standard_normal(size=(max_N, base_size))
    
    # Pre-compute RS theory
    print("\n[1/3] Computing RS theory...")
    theory_gammas = np.linspace(0.3, 1.8, 100)
    theory_f_values = []
    for g in tqdm(theory_gammas, desc="      Theory"):
        try:
            res = solve_zeroT_rs(g)
            theory_f_values.append(res.f)
        except:
            theory_f_values.append(np.nan)
    theory_f_values = np.array(theory_f_values)
    
    # Pre-compute range
    print("\n[2/3] Computing range...")
    all_f_values = []
    for gamma in tqdm(gamma_values, desc="      Range"):
        M = base_size
        N = max(3, int(round(gamma * M)))
        C = C_full[:N, :M]
        lp = solve_minmax_lp(C, return_strategies=False)
        f_scaled = (N * M) ** 0.25 * lp.value
        all_f_values.append(f_scaled)
    
    f_min = min(min(all_f_values), np.nanmin(theory_f_values))
    f_max = max(max(all_f_values), np.nanmax(theory_f_values))
    
    # Generate frames
    print(f"\n[3/3] Generating {len(gamma_values)} frames...")
    
    frames = []
    gamma_history = []
    f_history = []
    
    for i, gamma in enumerate(tqdm(gamma_values, desc="      Render")):
        M = base_size
        N = max(3, int(round(gamma * M)))
        gamma_eff = N / M
        
        C = C_full[:N, :M]
        
        lp = solve_minmax_lp(C, return_strategies=True)
        f_scaled = (N * M) ** 0.25 * lp.value
        
        try:
            theory = solve_zeroT_rs(gamma_eff)
            theory_f = theory.f
        except:
            theory_f = f_scaled
        
        gamma_history.append(gamma_eff)
        f_history.append(f_scaled)
        
        frame = create_frame(
            gamma=gamma_eff,
            C=C,
            p=lp.p,
            q=lp.q,
            f_scaled=f_scaled,
            theory_f=theory_f,
            gamma_history=gamma_history.copy(),
            f_history=f_history.copy(),
            all_gammas=theory_gammas,
            all_theory_f=theory_f_values,
            frame_idx=i,
            total_frames=len(gamma_values),
            f_min=f_min,
            f_max=f_max,
            base_size=base_size,
        )
        frames.append(frame)
    
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets",
        "nash_equilibrium_gamma_sweep.gif"
    )
    
    print(f"\n      Saving GIF...")
    
    frames = [frames[0]] * 12 + frames + [frames[-1]] * 18
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0,
        optimize=True
    )
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"\n" + "=" * 60)
    print(f"  ✓ Done!")
    print(f"    Size: {file_size:.2f} MB | Frames: {len(frames)}")
    print(f"    γ range: {gamma_values.min():.2f} → {gamma_values.max():.2f}")
    print(f"    At γ=1: {base_size}×{base_size} (square)")
    print("=" * 60)


if __name__ == "__main__":
    main()
