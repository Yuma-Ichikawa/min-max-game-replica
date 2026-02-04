"""
Create an animated GIF showing how Nash equilibrium value and mixed strategies
change as gamma (N/M ratio) varies for a random matrix game.

The matrix SIZE visually changes as gamma changes.

Usage:
    python -m demo.create_gamma_sweep_gif
"""
from __future__ import annotations

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch, Rectangle
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
    game_value: float,
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
    base_M: int,
    max_N: int,
) -> Image.Image:
    """Create a single frame for the animation."""
    
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
    })
    
    fig = plt.figure(figsize=(14, 12), facecolor='#fafafa')
    
    N, M = C.shape
    
    # Colors
    color_min = '#2563eb'  # Blue
    color_max = '#dc2626'  # Red
    color_theory = '#f59e0b'  # Amber
    color_numerical = '#10b981'  # Emerald
    
    # Layout: Top area for matrix (with dynamic sizing), Bottom for plot
    # Use a large top area where matrix can grow/shrink
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.2, 1], hspace=0.25)
    
    # === TOP: Matrix visualization area ===
    ax_top = fig.add_subplot(gs[0])
    ax_top.set_xlim(0, 100)
    ax_top.set_ylim(0, 100)
    ax_top.set_aspect('equal')
    ax_top.axis('off')
    ax_top.set_facecolor('#fafafa')
    
    # Calculate matrix display size (proportional to actual size)
    # Scale factor: matrix takes up to 70% of the available space
    scale = 0.55
    mat_width = (M / base_M) * 50 * scale
    mat_height = (N / max_N) * 80 * scale
    
    # Center position
    center_x, center_y = 50, 52
    
    # Matrix position (with margins for strategy bars)
    margin_left = 12  # Space for p bar
    margin_top = 8    # Space for q bar
    
    mat_left = center_x - mat_width/2 + margin_left/2
    mat_bottom = center_y - mat_height/2 - margin_top/2
    
    # Draw payoff matrix as colored rectangles
    cmap = LinearSegmentedColormap.from_list('payoff', 
        ['#1e40af', '#60a5fa', '#ffffff', '#f87171', '#b91c1c'], N=256)
    vmax = np.abs(C).max()
    
    cell_w = mat_width / M
    cell_h = mat_height / N
    
    # Draw matrix cells
    for i in range(N):
        for j in range(M):
            val = C[i, j]
            color = cmap((val + vmax) / (2 * vmax))
            rect = Rectangle(
                (mat_left + j * cell_w, mat_bottom + (N - 1 - i) * cell_h),
                cell_w, cell_h,
                facecolor=color,
                edgecolor='white',
                linewidth=0.1
            )
            ax_top.add_patch(rect)
    
    # Matrix border
    border = Rectangle(
        (mat_left, mat_bottom), mat_width, mat_height,
        facecolor='none', edgecolor='#374151', linewidth=2
    )
    ax_top.add_patch(border)
    
    # === Minimizer strategy p (left of matrix) ===
    x_strategy = N * p
    bar_width = 8
    bar_left = mat_left - bar_width - 2
    
    max_x = max(3, x_strategy.max())
    for i in range(N):
        val = x_strategy[i]
        bar_h = cell_h * 0.85
        bar_w = (val / max_x) * bar_width
        y_pos = mat_bottom + (N - 1 - i) * cell_h + cell_h * 0.075
        
        # Background bar
        bg_rect = Rectangle(
            (bar_left, y_pos), bar_width, bar_h,
            facecolor='#e5e7eb', edgecolor='none', alpha=0.5
        )
        ax_top.add_patch(bg_rect)
        
        # Value bar
        if val > 0.01:
            rect = Rectangle(
                (bar_left + bar_width - bar_w, y_pos), bar_w, bar_h,
                facecolor=color_min, edgecolor='none', alpha=0.85
            )
            ax_top.add_patch(rect)
    
    # Label for p
    ax_top.text(bar_left + bar_width/2, mat_bottom + mat_height + 3,
                f'$x_i$ (N={N})', fontsize=11, ha='center', va='bottom',
                color=color_min, fontweight='bold')
    
    # === Maximizer strategy q (top of matrix) ===
    y_strategy = M * q
    bar_height = 6
    bar_bottom = mat_bottom + mat_height + 2
    
    max_y = max(3, y_strategy.max())
    for j in range(M):
        val = y_strategy[j]
        bar_w = cell_w * 0.85
        bar_h = (val / max_y) * bar_height
        x_pos = mat_left + j * cell_w + cell_w * 0.075
        
        # Background bar
        bg_rect = Rectangle(
            (x_pos, bar_bottom), bar_w, bar_height,
            facecolor='#e5e7eb', edgecolor='none', alpha=0.5
        )
        ax_top.add_patch(bg_rect)
        
        # Value bar
        if val > 0.01:
            rect = Rectangle(
                (x_pos, bar_bottom), bar_w, bar_h,
                facecolor=color_max, edgecolor='none', alpha=0.85
            )
            ax_top.add_patch(rect)
    
    # Label for q
    ax_top.text(mat_left + mat_width + 3, bar_bottom + bar_height/2,
                f'$y_j$ (M={M})', fontsize=11, ha='left', va='center',
                color=color_max, fontweight='bold')
    
    # === Info text ===
    rho_x = support_fraction(p)
    rho_y = support_fraction(q)
    error = abs(f_scaled - theory_f) / (abs(theory_f) + 1e-10) * 100
    
    # Title
    ax_top.text(50, 97, r'Random Matrix Game: $\min_{\mathbf{x}} \max_{\mathbf{y}} \; \mathbf{x}^T C \mathbf{y}$',
                fontsize=18, ha='center', va='top', fontweight='bold', color='#1f2937')
    
    # Gamma display (large, prominent)
    ax_top.text(50, 88, f'γ = N/M = {gamma:.3f}',
                fontsize=16, ha='center', va='top', fontweight='bold',
                color='#6366f1',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#e0e7ff', 
                         edgecolor='#6366f1', linewidth=2))
    
    # Matrix size indicator
    ax_top.text(mat_left + mat_width/2, mat_bottom - 3,
                f'Payoff Matrix C ({N} × {M})',
                fontsize=12, ha='center', va='top', color='#374151', fontweight='bold')
    
    # Stats on the right
    stats_x = 85
    stats_y = 65
    stats = [
        (f'LP: {f_scaled:.4f}', color_numerical),
        (f'RS: {theory_f:.4f}', color_theory),
        (f'Err: {error:.1f}%', '#ef4444' if error > 3 else '#22c55e'),
        (f'ρx: {rho_x:.0%}', color_min),
        (f'ρy: {rho_y:.0%}', color_max),
    ]
    for i, (text, color) in enumerate(stats):
        ax_top.text(stats_x, stats_y - i*6, text,
                    fontsize=11, ha='left', va='center', color=color, fontweight='bold')
    
    # Colorbar legend (small, on the side)
    ax_top.text(stats_x, 35, 'Payoff:', fontsize=10, ha='left', color='#6b7280')
    for i, (label, color) in enumerate([('High', '#b91c1c'), ('0', '#ffffff'), ('Low', '#1e40af')]):
        ax_top.add_patch(Rectangle((stats_x + i*4, 30), 3.5, 3.5, 
                                    facecolor=color, edgecolor='#9ca3af', linewidth=0.5))
    ax_top.text(stats_x, 28, 'Low        High', fontsize=8, ha='left', color='#9ca3af')
    
    # === BOTTOM: Game value plot ===
    ax_plot = fig.add_subplot(gs[1])
    ax_plot.set_facecolor('#ffffff')
    
    # RS Theory curve
    ax_plot.plot(all_gammas, all_theory_f, '-', color=color_theory,
                 linewidth=4, label='RS Theory $f(\\gamma)$', alpha=0.9,
                 path_effects=[pe.Stroke(linewidth=6, foreground='white'), pe.Normal()])
    
    # LP Numerical points
    if len(gamma_history) > 1:
        ax_plot.plot(gamma_history, f_history, 'o-', color=color_numerical,
                     linewidth=2.5, markersize=6, label='LP Numerical', alpha=0.9)
    
    # Current point
    ax_plot.scatter([gamma], [f_scaled], s=300, color=color_numerical,
                    zorder=10, edgecolors='white', linewidth=3, marker='o')
    ax_plot.scatter([gamma], [theory_f], s=200, color=color_theory,
                    zorder=9, marker='D', edgecolors='white', linewidth=2)
    
    # Vertical line
    ax_plot.axvline(x=gamma, color='#6366f1', linestyle='--', alpha=0.5, linewidth=2)
    
    ax_plot.set_xlabel(r'$\gamma = N/M$', fontsize=14, fontweight='bold')
    ax_plot.set_ylabel(r'Scaled Value $f(\gamma)$', fontsize=14, fontweight='bold')
    ax_plot.set_title('Nash Equilibrium Value: RS Theory vs LP Numerical', 
                      fontsize=15, fontweight='bold', pad=10)
    ax_plot.legend(loc='upper right', fontsize=12, framealpha=0.95,
                   edgecolor='#d1d5db', fancybox=True)
    ax_plot.set_xlim(all_gammas.min() - 0.05, all_gammas.max() + 0.05)
    ax_plot.set_ylim(f_min - 0.05, f_max + 0.15)
    ax_plot.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax_plot.tick_params(labelsize=11)
    
    # Add frame counter
    ax_plot.text(0.02, 0.02, f'Frame {frame_idx+1}/{total_frames}',
                 transform=ax_plot.transAxes, fontsize=9, color='#9ca3af',
                 ha='left', va='bottom')
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#fafafa',
                edgecolor='none', bbox_inches='tight', pad_inches=0.15)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    return img


def main():
    print("=" * 60)
    print("  Creating Nash Equilibrium Animation")
    print("  (Matrix size changes dynamically!)")
    print("=" * 60)
    
    # Parameters
    seed = 42
    base_M = 80  # Fixed M
    gamma_values = np.concatenate([
        np.linspace(0.3, 1.0, 25),
        np.linspace(1.0, 1.8, 25)
    ])
    
    rng = np.random.default_rng(seed)
    
    max_N = int(2.0 * base_M) + 10
    C_full = rng.standard_normal(size=(max_N, base_M))
    
    # Pre-compute RS theory
    print("\n[1/3] Computing RS theory curve...")
    theory_gammas = np.linspace(0.25, 2.0, 100)
    theory_f_values = []
    for g in tqdm(theory_gammas, desc="      Theory"):
        try:
            res = solve_zeroT_rs(g)
            theory_f_values.append(res.f)
        except:
            theory_f_values.append(np.nan)
    theory_f_values = np.array(theory_f_values)
    
    # Pre-compute range
    print("\n[2/3] Computing value range...")
    all_f_values = []
    for gamma in tqdm(gamma_values, desc="      Range"):
        M = base_M
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
        M = base_M
        N = max(3, int(round(gamma * M)))
        gamma_eff = N / M
        
        C = C_full[:N, :M]
        
        lp = solve_minmax_lp(C, return_strategies=True)
        game_value = lp.value
        f_scaled = (N * M) ** 0.25 * game_value
        
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
            game_value=game_value,
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
            base_M=base_M,
            max_N=max_N,
        )
        frames.append(frame)
    
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets",
        "nash_equilibrium_gamma_sweep.gif"
    )
    
    print(f"\n      Saving GIF...")
    
    # Add pause frames
    frames = [frames[0]] * 10 + frames + [frames[-1]] * 15
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,
        loop=0,
        optimize=True
    )
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"\n" + "=" * 60)
    print(f"  ✓ Animation created!")
    print(f"    File: {os.path.basename(output_path)}")
    print(f"    Size: {file_size:.2f} MB | Frames: {len(frames)}")
    print(f"    Matrix: {int(0.3*base_M)}×{base_M} → {int(1.8*base_M)}×{base_M}")
    print("=" * 60)


if __name__ == "__main__":
    main()
