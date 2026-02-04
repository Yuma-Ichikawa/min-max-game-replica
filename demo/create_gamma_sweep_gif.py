"""
Create an animated GIF showing how Nash equilibrium value and mixed strategies
change as gamma (N/M ratio) varies for a random matrix game.

Visualizes:
- The payoff matrix C as a heatmap (size changes with gamma)
- Mixed strategies x = N*p (minimizer) and y = M*q (maximizer) as marginal bars
- Nash equilibrium value comparison: RS Theory vs LP Numerical

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
from matplotlib.patches import FancyBboxPatch
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
    theory_history: list,
    all_gammas: np.ndarray,
    all_theory_f: np.ndarray,
    frame_idx: int,
    total_frames: int,
    f_min: float,
    f_max: float,
    base_M: int,
) -> Image.Image:
    """Create a single frame for the animation."""
    
    # Use a clean, modern style
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.size': 11,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
    })
    
    fig = plt.figure(figsize=(16, 9), facecolor='white')
    
    N, M = C.shape
    
    # Colors - vibrant and modern
    color_min = '#2563eb'  # Blue for minimizer
    color_max = '#dc2626'  # Red for maximizer  
    color_theory = '#f59e0b'  # Amber for theory
    color_numerical = '#10b981'  # Emerald for numerical
    
    # Create custom layout
    # Main area: matrix with marginal strategies
    # Right side: game value plot (square)
    gs = GridSpec(3, 4, figure=fig,
                  width_ratios=[0.8, 3, 0.15, 2.5],
                  height_ratios=[0.8, 3, 0.6],
                  hspace=0.08, wspace=0.15)
    
    # === Maximizer strategy (top bar) ===
    ax_q = fig.add_subplot(gs[0, 1])
    y_strategy = M * q
    
    # Bar plot for q
    bars_q = ax_q.bar(np.arange(M), y_strategy, width=1.0, 
                      color=color_max, alpha=0.85, edgecolor='white', linewidth=0.3)
    ax_q.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_q.set_xlim(-0.5, M - 0.5)
    ax_q.set_ylim(0, max(3, y_strategy.max() * 1.2))
    ax_q.set_xticks([])
    ax_q.set_ylabel(r'$y_j$', fontsize=13, color=color_max, fontweight='bold')
    ax_q.spines['top'].set_visible(False)
    ax_q.spines['right'].set_visible(False)
    ax_q.spines['bottom'].set_visible(False)
    ax_q.tick_params(axis='y', colors='gray', labelsize=9)
    ax_q.set_title(f'Maximizer Strategy (M = {M})', fontsize=12, 
                   color=color_max, fontweight='bold', pad=5)
    
    # === Minimizer strategy (left bar) ===
    ax_p = fig.add_subplot(gs[1, 0])
    x_strategy = N * p
    
    bars_p = ax_p.barh(np.arange(N), x_strategy, height=1.0,
                       color=color_min, alpha=0.85, edgecolor='white', linewidth=0.3)
    ax_p.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax_p.set_ylim(-0.5, N - 0.5)
    ax_p.set_xlim(max(3, x_strategy.max() * 1.2), 0)  # Reversed
    ax_p.set_yticks([])
    ax_p.set_xlabel(r'$x_i$', fontsize=13, color=color_min, fontweight='bold')
    ax_p.invert_yaxis()
    ax_p.spines['top'].set_visible(False)
    ax_p.spines['right'].set_visible(False)
    ax_p.spines['left'].set_visible(False)
    ax_p.tick_params(axis='x', colors='gray', labelsize=9)
    ax_p.set_ylabel(f'Minimizer\n(N = {N})', fontsize=12, 
                    color=color_min, fontweight='bold', rotation=0, 
                    labelpad=30, va='center')
    
    # === Payoff Matrix (center) ===
    ax_mat = fig.add_subplot(gs[1, 1])
    
    # Custom colormap: blue-white-red
    cmap = LinearSegmentedColormap.from_list('payoff', 
        ['#1e40af', '#93c5fd', '#ffffff', '#fca5a5', '#b91c1c'], N=256)
    
    vmax = np.abs(C).max()
    im = ax_mat.imshow(C, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax,
                       interpolation='nearest')
    
    ax_mat.set_xticks([])
    ax_mat.set_yticks([])
    ax_mat.spines['top'].set_linewidth(2)
    ax_mat.spines['bottom'].set_linewidth(2)
    ax_mat.spines['left'].set_linewidth(2)
    ax_mat.spines['right'].set_linewidth(2)
    
    # === Colorbar ===
    ax_cbar = fig.add_subplot(gs[1, 2])
    cbar = plt.colorbar(im, cax=ax_cbar)
    cbar.set_label('Payoff', fontsize=11, labelpad=10)
    cbar.ax.tick_params(labelsize=9)
    
    # === Game Value Plot (right, square) ===
    ax_val = fig.add_subplot(gs[:2, 3], aspect='equal')
    ax_val.set_facecolor('#fafafa')
    
    # Plot RS theory curve
    ax_val.plot(all_gammas, all_theory_f, '-', color=color_theory,
                linewidth=3.5, label='RS Theory', alpha=0.9,
                path_effects=[pe.Stroke(linewidth=5, foreground='white'), pe.Normal()])
    
    # Plot numerical results
    if len(gamma_history) > 1:
        ax_val.plot(gamma_history, f_history, 'o-', color=color_numerical,
                    linewidth=2.5, markersize=5, label='LP Numerical', alpha=0.9)
    
    # Current point - large and prominent
    ax_val.scatter([gamma], [f_scaled], s=300, color=color_numerical,
                   zorder=10, edgecolors='white', linewidth=3, marker='o')
    ax_val.scatter([gamma], [theory_f], s=200, color=color_theory,
                   zorder=9, marker='D', edgecolors='white', linewidth=2)
    
    # Connecting line between theory and numerical
    ax_val.plot([gamma, gamma], [theory_f, f_scaled], 
                color='gray', linestyle=':', linewidth=2, alpha=0.6)
    
    # Vertical line at current gamma
    ax_val.axvline(x=gamma, color='gray', linestyle='--', alpha=0.3, linewidth=1)
    
    ax_val.set_xlabel(r'$\gamma = N/M$', fontsize=14, fontweight='bold')
    ax_val.set_ylabel(r'$f(\gamma)$', fontsize=14, fontweight='bold')
    ax_val.set_title('Nash Equilibrium Value', fontsize=15, fontweight='bold', pad=10)
    ax_val.legend(loc='upper right', fontsize=11, framealpha=0.95, 
                  edgecolor='gray', fancybox=True)
    ax_val.set_xlim(all_gammas.min() - 0.05, all_gammas.max() + 0.05)
    ax_val.set_ylim(f_min - 0.05, f_max + 0.15)
    ax_val.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    ax_val.tick_params(labelsize=11)
    
    # Make it look square by setting equal limits range
    x_range = all_gammas.max() - all_gammas.min() + 0.1
    y_range = f_max - f_min + 0.2
    
    # === Info panel (bottom) ===
    ax_info = fig.add_subplot(gs[2, :2])
    ax_info.axis('off')
    
    rho_x = support_fraction(p)
    rho_y = support_fraction(q)
    error = abs(f_scaled - theory_f) / (abs(theory_f) + 1e-10) * 100
    
    # Create info text with nice formatting
    info_parts = [
        f"γ = {gamma:.3f}",
        f"Matrix: {N} × {M}",
        f"LP: {f_scaled:.4f}",
        f"Theory: {theory_f:.4f}",
        f"Error: {error:.2f}%",
        f"Support: ({rho_x:.0%}, {rho_y:.0%})"
    ]
    
    # Draw info boxes
    box_width = 0.15
    start_x = 0.02
    for i, text in enumerate(info_parts):
        x_pos = start_x + i * (box_width + 0.01)
        
        # Choose color based on content
        if 'γ' in text:
            box_color = '#e0e7ff'
            text_color = '#3730a3'
        elif 'LP' in text:
            box_color = '#d1fae5'
            text_color = '#065f46'
        elif 'Theory' in text:
            box_color = '#fef3c7'
            text_color = '#92400e'
        elif 'Error' in text:
            box_color = '#fee2e2' if error > 5 else '#d1fae5'
            text_color = '#991b1b' if error > 5 else '#065f46'
        else:
            box_color = '#f3f4f6'
            text_color = '#374151'
        
        bbox = FancyBboxPatch((x_pos, 0.2), box_width, 0.6,
                              boxstyle="round,pad=0.02,rounding_size=0.02",
                              facecolor=box_color, edgecolor='#9ca3af',
                              linewidth=1.5, transform=ax_info.transAxes,
                              zorder=1)
        ax_info.add_patch(bbox)
        ax_info.text(x_pos + box_width/2, 0.5, text,
                     transform=ax_info.transAxes, fontsize=11,
                     ha='center', va='center', color=text_color,
                     fontweight='bold', zorder=2)
    
    # === Main title with animation effect ===
    progress = (frame_idx + 1) / total_frames
    title_text = r'Min-Max Game: $\min_{\mathbf{x}} \max_{\mathbf{y}} \; \mathbf{x}^T C \mathbf{y}$'
    
    fig.suptitle(title_text, fontsize=20, fontweight='bold', y=0.98,
                 color='#1f2937')
    
    # Progress bar at the very bottom
    ax_progress = fig.add_axes([0.1, 0.02, 0.8, 0.015])
    ax_progress.set_xlim(0, 1)
    ax_progress.set_ylim(0, 1)
    ax_progress.barh(0.5, progress, height=1.0, color='#6366f1', alpha=0.8)
    ax_progress.barh(0.5, 1.0, height=1.0, color='#e5e7eb', alpha=0.5, zorder=0)
    ax_progress.axis('off')
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='white',
                edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    return img


def main():
    print("=" * 60)
    print("  Creating Nash Equilibrium Animation")
    print("=" * 60)
    
    # Parameters
    seed = 42
    base_M = 100  # Fixed M, N changes with gamma
    gamma_values = np.concatenate([
        np.linspace(0.3, 1.0, 28),
        np.linspace(1.0, 1.7, 22)
    ])
    
    rng = np.random.default_rng(seed)
    
    # Generate full random matrix (large enough for all gammas)
    max_N = int(2.0 * base_M) + 10
    C_full = rng.standard_normal(size=(max_N, base_M))
    
    # Pre-compute RS theory curve
    print("\n[1/3] Pre-computing RS theory curve...")
    theory_gammas = np.linspace(0.2, 2.0, 100)
    theory_f_values = []
    for g in tqdm(theory_gammas, desc="      RS Theory"):
        try:
            res = solve_zeroT_rs(g)
            theory_f_values.append(res.f)
        except:
            theory_f_values.append(np.nan)
    theory_f_values = np.array(theory_f_values)
    
    # Pre-compute value range
    print("\n[2/3] Computing value range...")
    all_f_values = []
    for gamma in tqdm(gamma_values, desc="      LP Range"):
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
    print(f"      Matrix: N×{base_M} where N = γ×{base_M}")
    
    frames = []
    gamma_history = []
    f_history = []
    theory_history = []
    
    for i, gamma in enumerate(tqdm(gamma_values, desc="      Rendering")):
        M = base_M
        N = max(3, int(round(gamma * M)))
        gamma_eff = N / M
        
        C = C_full[:N, :M]
        
        # Solve LP
        lp = solve_minmax_lp(C, return_strategies=True)
        game_value = lp.value
        f_scaled = (N * M) ** 0.25 * game_value
        
        # RS Theory
        try:
            theory = solve_zeroT_rs(gamma_eff)
            theory_f = theory.f
        except:
            theory_f = f_scaled
        
        gamma_history.append(gamma_eff)
        f_history.append(f_scaled)
        theory_history.append(theory_f)
        
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
            theory_history=theory_history.copy(),
            all_gammas=theory_gammas,
            all_theory_f=theory_f_values,
            frame_idx=i,
            total_frames=len(gamma_values),
            f_min=f_min,
            f_max=f_max,
            base_M=base_M,
        )
        frames.append(frame)
    
    # Output path
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets",
        "nash_equilibrium_gamma_sweep.gif"
    )
    
    print(f"\n      Saving GIF...")
    
    # Add pause frames at start and end
    frames = [frames[0]] * 8 + frames + [frames[-1]] * 12
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=100,  # ms per frame
        loop=0,
        optimize=True
    )
    
    file_size = os.path.getsize(output_path) / 1024 / 1024
    
    print(f"\n" + "=" * 60)
    print(f"  ✓ Animation created successfully!")
    print(f"    - File: {output_path}")
    print(f"    - Size: {file_size:.2f} MB")
    print(f"    - Frames: {len(frames)}")
    print(f"    - Matrix range: {int(0.3*base_M)}×{base_M} to {int(1.7*base_M)}×{base_M}")
    print("=" * 60)


if __name__ == "__main__":
    main()
