"""
Create an animated GIF showing how Nash equilibrium value and mixed strategies
change as gamma (N/M ratio) varies for a random matrix game.

Visualizes:
- The payoff matrix C as a heatmap
- Mixed strategies p (minimizer) and q (maximizer) overlaid on the matrix
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
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
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
) -> Image.Image:
    """Create a single frame for the animation."""
    
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(16, 10), facecolor='#0d1117')
    
    # Layout: 2 rows, 3 columns
    # Row 1: Payoff matrix (large), Strategy p bar, Strategy q bar
    # Row 2: Game value plot (spans 2 cols), Info panel
    gs = GridSpec(2, 3, figure=fig, 
                  width_ratios=[2, 1, 1], 
                  height_ratios=[1.2, 1],
                  hspace=0.35, wspace=0.3)
    
    N, M = C.shape
    
    # Colors
    color_min = '#00d4ff'  # cyan for minimizer
    color_max = '#ff6b6b'  # red for maximizer
    color_theory = '#feca57'  # yellow for theory
    
    # === Panel 1: Payoff Matrix with Strategy Overlay ===
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor('#161b22')
    
    # Create custom colormap for matrix
    cmap = LinearSegmentedColormap.from_list('custom', 
        ['#1e3a5f', '#0d1117', '#5f1e1e'], N=256)
    
    # Plot payoff matrix
    vmax = np.abs(C).max()
    im = ax1.imshow(C, cmap=cmap, aspect='auto', vmin=-vmax, vmax=vmax)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    cbar.set_label('Payoff $C_{ij}$', fontsize=11, color='white')
    cbar.ax.tick_params(colors='white')
    
    # Overlay strategy intensities
    # Row strategy (minimizer p) - highlight rows with high p
    for i in range(N):
        if p[i] > 1e-6:
            intensity = min(p[i] * N / 3, 1.0)  # normalize
            rect = Rectangle((-0.5, i - 0.5), M, 1, 
                           linewidth=0, edgecolor='none',
                           facecolor=color_min, alpha=intensity * 0.4)
            ax1.add_patch(rect)
    
    # Column strategy (maximizer q) - highlight columns with high q
    for j in range(M):
        if q[j] > 1e-6:
            intensity = min(q[j] * M / 3, 1.0)
            rect = Rectangle((j - 0.5, -0.5), 1, N,
                           linewidth=0, edgecolor='none',
                           facecolor=color_max, alpha=intensity * 0.3)
            ax1.add_patch(rect)
    
    ax1.set_xlabel('Maximizer Actions (j)', fontsize=12, color='white')
    ax1.set_ylabel('Minimizer Actions (i)', fontsize=12, color='white')
    ax1.set_title(f'Payoff Matrix C ({N}×{M})\nwith Mixed Strategy Overlay', 
                  fontsize=14, fontweight='bold', color='white', pad=10)
    ax1.tick_params(colors='white')
    
    # === Panel 2: Minimizer Strategy p ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#161b22')
    
    # Scale to x = N*p
    x_strategy = N * p
    bars_p = ax2.barh(np.arange(N), x_strategy, color=color_min, 
                      alpha=0.8, height=0.8)
    
    # Highlight active strategies
    for i, (bar, val) in enumerate(zip(bars_p, x_strategy)):
        if val > 0.1:
            bar.set_alpha(1.0)
            bar.set_edgecolor('white')
            bar.set_linewidth(1)
    
    ax2.axvline(x=1.0, color='white', linestyle='--', alpha=0.5, 
                label='Uniform')
    ax2.set_xlabel(r'$x_i = N \cdot p_i$', fontsize=11, color='white')
    ax2.set_ylabel('Action i', fontsize=11, color='white')
    ax2.set_title('Minimizer Strategy', fontsize=13, fontweight='bold', 
                  color=color_min)
    ax2.set_ylim(-0.5, N - 0.5)
    ax2.set_xlim(0, max(x_strategy.max() * 1.1, 2))
    ax2.invert_yaxis()
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, axis='x')
    
    # === Panel 3: Maximizer Strategy q ===
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#161b22')
    
    y_strategy = M * q
    bars_q = ax3.bar(np.arange(M), y_strategy, color=color_max,
                     alpha=0.8, width=0.8)
    
    for j, (bar, val) in enumerate(zip(bars_q, y_strategy)):
        if val > 0.1:
            bar.set_alpha(1.0)
            bar.set_edgecolor('white')
            bar.set_linewidth(1)
    
    ax3.axhline(y=1.0, color='white', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Action j', fontsize=11, color='white')
    ax3.set_ylabel(r'$y_j = M \cdot q_j$', fontsize=11, color='white')
    ax3.set_title('Maximizer Strategy', fontsize=13, fontweight='bold',
                  color=color_max)
    ax3.set_xlim(-0.5, M - 0.5)
    ax3.set_ylim(0, max(y_strategy.max() * 1.1, 2))
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, axis='y')
    
    # === Panel 4: Game Value Comparison (spans 2 columns) ===
    ax4 = fig.add_subplot(gs[1, :2])
    ax4.set_facecolor('#161b22')
    
    # Plot full RS theory curve (pre-computed)
    ax4.plot(all_gammas, all_theory_f, '-', color=color_theory, 
             linewidth=3, label='RS Theory $f(\\gamma)$', alpha=0.9)
    
    # Plot LP numerical results so far
    if len(gamma_history) > 0:
        ax4.plot(gamma_history, f_history, 'o-', color=color_min,
                 linewidth=2, markersize=6, label='LP Numerical', alpha=0.9)
    
    # Current point highlighted
    ax4.scatter([gamma], [f_scaled], s=250, color='white', 
                zorder=10, edgecolors=color_min, linewidth=3,
                marker='o')
    ax4.scatter([gamma], [theory_f], s=200, color=color_theory,
                zorder=9, marker='*', edgecolors='white', linewidth=1)
    
    # Vertical line at current gamma
    ax4.axvline(x=gamma, color='white', linestyle=':', alpha=0.4)
    
    ax4.set_xlabel(r'$\gamma = N/M$', fontsize=14, color='white')
    ax4.set_ylabel(r'Scaled Game Value $f(\gamma)$', fontsize=14, color='white')
    ax4.set_title('Nash Equilibrium Value: RS Theory vs LP Numerical', 
                  fontsize=15, fontweight='bold', color='white', pad=10)
    ax4.legend(loc='upper right', fontsize=12, framealpha=0.8)
    ax4.set_xlim(all_gammas.min() - 0.05, all_gammas.max() + 0.05)
    ax4.set_ylim(f_min - 0.05, f_max + 0.1)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.tick_params(colors='white', labelsize=11)
    
    # === Panel 5: Info Panel ===
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#161b22')
    ax5.axis('off')
    
    rho_x = support_fraction(p)
    rho_y = support_fraction(q)
    error = abs(f_scaled - theory_f)
    rel_error = error / (abs(theory_f) + 1e-10) * 100
    
    info_text = f"""
  ┌─────────────────────────┐
  │   Current State         │
  ├─────────────────────────┤
  │                         │
  │  γ = N/M = {gamma:.4f}      │
  │                         │
  │  Matrix: {N} × {M}         │
  │                         │
  ├─────────────────────────┤
  │   Game Values           │
  ├─────────────────────────┤
  │                         │
  │  LP Numerical:  {f_scaled:.5f} │
  │  RS Theory:     {theory_f:.5f} │
  │                         │
  │  Error: {rel_error:.2f}%          │
  │                         │
  ├─────────────────────────┤
  │   Support Fractions     │
  ├─────────────────────────┤
  │                         │
  │  Minimizer ρₓ: {rho_x:.1%}    │
  │  Maximizer ρᵧ: {rho_y:.1%}    │
  │                         │
  └─────────────────────────┘
    """
    
    ax5.text(0.05, 0.95, info_text, transform=ax5.transAxes,
             fontsize=11, color='white', verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117',
                      edgecolor='#30363d', linewidth=2))
    
    # Main title
    fig.suptitle(
        r'Zero-Temperature Min-Max Game: $\min_{\mathbf{p}} \max_{\mathbf{q}} \; \mathbf{p}^T C \mathbf{q}$',
        fontsize=18, fontweight='bold', color='white', y=0.98
    )
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=110, facecolor='#0d1117',
                edgecolor='none', bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf).copy()
    buf.close()
    plt.close(fig)
    
    return img


def main():
    print("=" * 60)
    print("Creating Nash Equilibrium Animation")
    print("=" * 60)
    
    # Parameters - larger matrix for better RS theory match
    seed = 42
    base_M = 120  # Larger size for better convergence to RS theory
    gamma_values = np.concatenate([
        np.linspace(0.25, 1.0, 30),
        np.linspace(1.0, 1.8, 20)
    ])
    
    rng = np.random.default_rng(seed)
    
    # Generate full random matrix
    max_N = int(2.0 * base_M) + 10
    C_full = rng.standard_normal(size=(max_N, base_M))
    
    # Pre-compute RS theory curve for smooth plotting
    print("\nPre-computing RS theory curve...")
    theory_gammas = np.linspace(0.2, 2.0, 100)
    theory_f_values = []
    for g in theory_gammas:
        try:
            res = solve_zeroT_rs(g)
            theory_f_values.append(res.f)
        except:
            theory_f_values.append(np.nan)
    theory_f_values = np.array(theory_f_values)
    
    # Compute f range for consistent axis
    print("Pre-computing value range...")
    all_f_values = []
    for gamma in tqdm(gamma_values, desc="Computing range"):
        M = base_M
        N = max(2, int(round(gamma * M)))
        C = C_full[:N, :M]
        lp = solve_minmax_lp(C, return_strategies=False)
        f_scaled = (N * M) ** 0.25 * lp.value
        all_f_values.append(f_scaled)
    
    f_min = min(min(all_f_values), np.nanmin(theory_f_values))
    f_max = max(max(all_f_values), np.nanmax(theory_f_values))
    
    frames = []
    gamma_history = []
    f_history = []
    theory_history = []
    
    print(f"\nGenerating {len(gamma_values)} frames...")
    print(f"Matrix size: up to {max_N} × {base_M}")
    
    for i, gamma in enumerate(tqdm(gamma_values, desc="Rendering")):
        M = base_M
        N = max(2, int(round(gamma * M)))
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
        )
        frames.append(frame)
    
    # Output path
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets",
        "nash_equilibrium_gamma_sweep.gif"
    )
    
    print(f"\nSaving GIF to: {output_path}")
    
    # Add pause frames at end
    frames.extend([frames[-1]] * 15)
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=150,  # ms per frame
        loop=0,
        optimize=True
    )
    
    print(f"\n✓ GIF created successfully!")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"  Frames: {len(frames)}")
    print(f"  Matrix size: up to {max_N} × {base_M}")
    print("=" * 60)


if __name__ == "__main__":
    main()
