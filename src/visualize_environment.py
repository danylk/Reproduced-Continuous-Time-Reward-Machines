import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from AutonomousVehicleCTRM import AutonomousVehicleCTRM

FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'figures')

def visualize_traffic_rates():
    """
    Visualize the traffic rates across the grid.
    Similar to Figure 1(a) in the paper with color-coded cells.
    """
    ctrm = AutonomousVehicleCTRM(grid_size=11)
    
    # Create rate matrix for visualization (average across all actions)
    rate_matrix_f1 = np.zeros((11, 11))
    rate_matrix_f2 = np.zeros((11, 11))
    
    for row in range(11):
        for col in range(11):
            # Average rate across all actions for f1 (normal traffic)
            rates_f1 = [ctrm.function1[(row, col, action)] for action in ctrm.actions]
            rate_matrix_f1[row, col] = np.mean(rates_f1)
            
            # Average rate across all actions for f2 (emergency)
            rates_f2 = [ctrm.function2[(row, col, action)] for action in ctrm.actions]
            rate_matrix_f2[row, col] = np.mean(rates_f2)
    
    # Create custom colormap (green=fast/low traffic, red=slow/high traffic)
    colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('traffic', colors, N=n_bins)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot f1 (normal traffic rates)
    im1 = ax1.imshow(rate_matrix_f1, cmap=cmap, origin='lower', aspect='auto')
    ax1.set_title('Traffic Rates - f₁ (Normal Traffic)\nq₀: Without Package, q₂: After Delivery', 
                  fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column (x)', fontsize=12)
    ax1.set_ylabel('Row (y)', fontsize=12)
    
    # Mark special locations on f1
    ax1.plot(0, 0, 'ws', markersize=15, label='Start (0,0)')
    ax1.plot(0, 10, 'w^', markersize=15, label='Package (10,0)')
    ax1.plot(10, 10, 'wP', markersize=15, label='Hospital (10,10)')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add colorbar for f1
    cbar1 = plt.colorbar(im1, ax=ax1)
    cbar1.set_label('Average Rate (higher = faster)', fontsize=11)
    
    # Plot f2 (emergency delivery rates)
    im2 = ax2.imshow(rate_matrix_f2, cmap=cmap, origin='lower', aspect='auto')
    ax2.set_title('Traffic Rates - f₂ (Emergency Delivery)\nq₁: With Package (10x faster)', 
                  fontsize=14, fontweight='bold')
    ax2.set_xlabel('Column (x)', fontsize=12)
    ax2.set_ylabel('Row (y)', fontsize=12)
    
    # Mark special locations on f2
    ax2.plot(0, 0, 'ws', markersize=15, label='Start (0,0)')
    ax2.plot(0, 10, 'w^', markersize=15, label='Package (10,0)')
    ax2.plot(10, 10, 'wP', markersize=15, label='Hospital (10,10)')
    ax2.legend(loc='lower right', fontsize=10)
    ax2.grid(True, alpha=0.3, color='white', linewidth=0.5)
    
    # Add colorbar for f2
    cbar2 = plt.colorbar(im2, ax=ax2)
    cbar2.set_label('Average Rate (higher = faster)', fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'traffic_rates_heatmap.png'), dpi=300, bbox_inches='tight')
    print("Saved: results/figures/traffic_rates_heatmap.png")
    plt.show()
    
    # Print statistics
    print("\n" + "="*60)
    print("TRAFFIC RATE STATISTICS")
    print("="*60)
    print(f"\nNormal Traffic (f₁):")
    print(f"  Min rate:  {rate_matrix_f1.min():.3f}")
    print(f"  Max rate:  {rate_matrix_f1.max():.3f}")
    print(f"  Mean rate: {rate_matrix_f1.mean():.3f}")
    
    print(f"\nEmergency Delivery (f₂):")
    print(f"  Min rate:  {rate_matrix_f2.min():.3f}")
    print(f"  Max rate:  {rate_matrix_f2.max():.3f}")
    print(f"  Mean rate: {rate_matrix_f2.mean():.3f}")
    
    print(f"\nRate Multiplier: {rate_matrix_f2.mean() / rate_matrix_f1.mean():.1f}x")
    print("="*60)
    
    # Create single combined visualization similar to paper Figure 1(a)
    fig, ax = plt.subplots(figsize=(10, 10))
    
    im = ax.imshow(rate_matrix_f1, cmap=cmap, origin='lower', aspect='auto')
    ax.set_title('Autonomous Vehicle Delivery Environment\n11×11 Gridworld with Traffic Rates', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Column (x)', fontsize=13)
    ax.set_ylabel('Row (y)', fontsize=13)
    
    # Mark special locations with larger, clearer markers
    ax.plot(0, 0, 'w', marker='s', markersize=20, markeredgewidth=2, 
            markeredgecolor='black', label='Distribution Center (Start)')
    ax.plot(0, 10, 'w', marker='^', markersize=20, markeredgewidth=2,
            markeredgecolor='black', label='Package Depot')
    ax.plot(10, 10, 'w', marker='P', markersize=20, markeredgewidth=2, 
            markeredgecolor='black', label='Hospital (Goal)')
    
    # Add text annotations
    ax.text(0, 0, 'START', ha='center', va='center', fontsize=8, 
            fontweight='bold', color='black')
    ax.text(0, 10, 'PKG', ha='center', va='center', fontsize=8,
            fontweight='bold', color='black')
    ax.text(10, 10, 'HOSP', ha='center', va='center', fontsize=8, 
            fontweight='bold', color='black')
    
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.4, color='white', linewidth=1)
    
    # Colorbar with label
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Average Traversal Rate\n(Red=Heavy Traffic, Green=Light Traffic)', 
                   fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'autonomous_vehicle_grid.png'), dpi=300, bbox_inches='tight')
    print("\nSaved: results/figures/autonomous_vehicle_grid.png")
    plt.show()

if __name__ == "__main__":
    print("Generating visualizations...")
    print("\n1. Traffic Rate Heatmaps")
    visualize_traffic_rates()
    
    print("\n" + "="*60)
    print("All visualizations complete!")
    print("="*60)
