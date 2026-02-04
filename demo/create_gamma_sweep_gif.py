"""
Create an animated GIF showing how Nash equilibrium value and mixed strategies
change as gamma (N/M ratio) varies for a random matrix game.

This demonstrates the zero-temperature limit behavior of the two-player game.

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
    frame_idx: int,
    total_frames: int,
) -> Image.Image:
    """Create a single frame for the animation."""
    
    # Set up the figure with dark theme
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 8), facecolor='#1a1a2e')
    gs = GridSpec(2, 3, figure=fig, width_ratios=[1.2, 1, 1], height_ratios=[1, 1],
                  hspace=0.3, wspace=0.3)
    
    N, M = C.shape
    
    # Color scheme
    color_primary = '#00d4ff'
    color_secondary = '#ff6b6b'
    color_theory = '#feca57'
    color_accent = '#5f27cd'
    
    # === Panel 1: Game Value vs Gamma (top-left, spans 2 rows) ===
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_facecolor('#16213e')
    
    if len(gamma_history) > 1:
        ax1.plot(gamma_history, theory_history, '--', color=color_theory, 
                 linewidth=2.5, label='RS Theory', alpha=0.8)
        ax1.plot(gamma_history, f_history, '-', color=color_primary, 
                 linewidth=3, label='LP Numerical')
        ax1.scatter([gamma], [f_scaled], s=200, color=color_secondary, 
                    zorder=5, edgecolors='white', linewidth=2)
    else:
        ax1.scatter([gamma], [f_scaled], s=200, color=color_primary, 
                    label='LP Numerical', edgecolors='white', linewidth=2)
        ax1.scatter([gamma], [theory_f], s=150, color=color_theory, 
                    marker='*', label='RS Theory')
    
    ax1.set_xlabel(r'$\gamma = N/M$', fontsize=14, color='white')
    ax1.set_ylabel(r'Scaled Game Value $f(\gamma)$', fontsize=14, color='white')
    ax1.set_title('Nash Equilibrium Value', fontsize=16, fontweight='bold', 
                  color='white', pad=10)
    ax1.legend(loc='upper right', fontsize=11, framealpha=0.8)
    ax1.set_xlim(0.15, 2.05)
    ax1.set_ylim(-0.1, 1.5)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.tick_params(colors='white')
    
    # === Panel 2: Minimizer Strategy p (top-right) ===
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor('#16213e')
    
    # Sort strategies for better visualization
    p_sorted = np.sort(p)[::-1]
    x_indices = np.arange(len(p_sorted))
    
    # Create gradient colors based on value
    colors_p = plt.cm.cool(p_sorted / (p_sorted.max() + 1e-10))
    ax2.bar(x_indices, p_sorted * N, color=colors_p, width=1.0, edgecolor='none')
    ax2.axhline(y=1.0, color=color_secondary, linestyle='--', alpha=0.7, 
                label='Uniform level')
    
    ax2.set_xlabel('Strategy Index (sorted)', fontsize=11, color='white')
    ax2.set_ylabel(r'$x_i = N \cdot p_i$', fontsize=11, color='white')
    ax2.set_title(f'Minimizer Strategy (N={N})', fontsize=13, 
                  fontweight='bold', color='white')
    ax2.set_xlim(-0.5, len(p_sorted) - 0.5)
    ax2.tick_params(colors='white')
    ax2.grid(True, alpha=0.2, axis='y')
    
    # === Panel 3: Maximizer Strategy q (top-right second) ===
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor('#16213e')
    
    q_sorted = np.sort(q)[::-1]
    colors_q = plt.cm.autumn(q_sorted / (q_sorted.max() + 1e-10))
    ax3.bar(np.arange(len(q_sorted)), q_sorted * M, color=colors_q, 
            width=1.0, edgecolor='none')
    ax3.axhline(y=1.0, color=color_secondary, linestyle='--', alpha=0.7)
    
    ax3.set_xlabel('Strategy Index (sorted)', fontsize=11, color='white')
    ax3.set_ylabel(r'$y_j = M \cdot q_j$', fontsize=11, color='white')
    ax3.set_title(f'Maximizer Strategy (M={M})', fontsize=13, 
                  fontweight='bold', color='white')
    ax3.set_xlim(-0.5, len(q_sorted) - 0.5)
    ax3.tick_params(colors='white')
    ax3.grid(True, alpha=0.2, axis='y')
    
    # === Panel 4: Support fractions (bottom-middle) ===
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor('#16213e')
    
    rho_x = support_fraction(p)
    rho_y = support_fraction(q)
    
    bars = ax4.bar(['Min (ρₓ)', 'Max (ρᵧ)'], [rho_x, rho_y], 
                   color=[color_primary, color_secondary], 
                   edgecolor='white', linewidth=2, width=0.6)
    
    # Add value labels on bars
    for bar, val in zip(bars, [rho_x, rho_y]):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{val:.2%}', ha='center', va='bottom', 
                fontsize=14, color='white', fontweight='bold')
    
    ax4.set_ylabel('Support Fraction', fontsize=12, color='white')
    ax4.set_title('Active Strategies', fontsize=13, fontweight='bold', color='white')
    ax4.set_ylim(0, 1.1)
    ax4.tick_params(colors='white')
    ax4.grid(True, alpha=0.2, axis='y')
    
    # === Panel 5: Info panel (bottom-right) ===
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor('#16213e')
    ax5.axis('off')
    
    info_text = f"""
    Current Parameters
    ══════════════════
    
    γ = N/M = {gamma:.3f}
    
    Matrix Size: {N} × {M}
    
    ──────────────────
    
    Game Value: {game_value:.4f}
    
    Scaled Value f(γ): {f_scaled:.4f}
    
    RS Theory f(γ): {theory_f:.4f}
    
    ──────────────────
    
    Support (Min): {rho_x:.1%}
    Support (Max): {rho_y:.1%}
    """
    
    ax5.text(0.1, 0.95, info_text, transform=ax5.transAxes,
             fontsize=12, color='white', verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#0f3460', 
                      edgecolor=color_primary, linewidth=2))
    
    # Main title with gamma indicator
    progress = frame_idx / max(total_frames - 1, 1)
    fig.suptitle(
        f'Zero-Temperature Nash Equilibrium Analysis\n'
        f'γ sweep: {progress:.0%} complete',
        fontsize=18, fontweight='bold', color='white', y=0.98
    )
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100, facecolor='#1a1a2e', 
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
    
    # Parameters
    seed = 42
    base_M = 60
    gamma_values = np.concatenate([
        np.linspace(0.2, 1.0, 25),
        np.linspace(1.0, 2.0, 25)
    ])
    
    rng = np.random.default_rng(seed)
    
    # We'll use a fixed random matrix that we resize
    # by selecting submatrices
    max_N = int(2.0 * base_M) + 10
    C_full = rng.standard_normal(size=(max_N, base_M))
    
    frames = []
    gamma_history = []
    f_history = []
    theory_history = []
    
    print(f"\nGenerating {len(gamma_values)} frames...")
    
    for i, gamma in enumerate(tqdm(gamma_values, desc="Processing")):
        M = base_M
        N = max(2, int(round(gamma * M)))
        gamma_eff = N / M
        
        # Use submatrix
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
            theory_f = f_scaled  # fallback
        
        gamma_history.append(gamma_eff)
        f_history.append(f_scaled)
        theory_history.append(theory_f)
        
        # Create frame
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
            frame_idx=i,
            total_frames=len(gamma_values),
        )
        frames.append(frame)
    
    # Create GIF
    output_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "assets",
        "nash_equilibrium_gamma_sweep.gif"
    )
    
    print(f"\nSaving GIF to: {output_path}")
    
    # Add some pause frames at the end
    frames.extend([frames[-1]] * 10)
    
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=120,  # ms per frame
        loop=0,
        optimize=True
    )
    
    print(f"\n✓ GIF created successfully!")
    print(f"  Size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    print(f"  Frames: {len(frames)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
