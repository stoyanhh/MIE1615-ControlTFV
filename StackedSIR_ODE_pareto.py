import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

def load_ode_results(filepath):
    """Load ODE results from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def compute_ode_cumulative_metrics(ode_data):
    """
    Compute cumulative infections and violence from ODE solution.
    Uses trapezoidal integration over time.
    
    Returns:
        cumulative_infections: Integral of (IN + IV) over time (person-days)
        cumulative_violence: Integral of (SV + IV + RV) over time (person-days)
    """
    t = ode_data['t']
    x = ode_data['x']
    
    # Extract compartments
    IN = x[:, 1]
    IV = x[:, 4]
    SV = x[:, 3]
    RV = x[:, 5]
    
    # Integrate over time using trapezoidal rule
    cumulative_infections = np.trapz(IN + IV, t)
    cumulative_violence = np.trapz(SV + IV + RV, t)
    
    return cumulative_infections, cumulative_violence

def plot_ode_pareto_frontier(pkl_files, output_file='ode_pareto_frontier.png'):
    """
    Create Pareto frontier plot from multiple ODE results.
    
    Args:
        pkl_files: List of paths to .pkl files (or list of tuples (path, label))
        output_file: Output filename for the plot
    """
    print("Loading ODE results and computing metrics...")
    
    results = []
    
    for item in pkl_files:
        # Handle both string paths and (path, label) tuples
        if isinstance(item, tuple):
            filepath, custom_label = item
        else:
            filepath = item
            custom_label = None
        
        print(f"\nProcessing: {filepath}")
        ode_data = load_ode_results(filepath)
        
        alpha = ode_data['parameters']['alpha']
        J = ode_data['J']
        
        # Compute metrics
        infections, violence = compute_ode_cumulative_metrics(ode_data)
        
        # Use custom label if provided, otherwise use alpha
        label = custom_label if custom_label else f'α={alpha:.2f}'
        
        results.append({
            'alpha': alpha,
            'label': label,
            'infections': infections,
            'violence': violence,
            'cost': J,
            'filepath': filepath
        })
        
        print(f"  α={alpha:.2f}: Infections={infections:.1f} person-days, "
              f"Violence={violence:.1f} person-days, Cost={J:.2f}")
    
    # Sort by alpha for consistent ordering
    results.sort(key=lambda x: x['alpha'])
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Extract data
    infections = [r['infections'] for r in results]
    violence = [r['violence'] for r in results]
    labels = [r['label'] for r in results]
    alphas = [r['alpha'] for r in results]
    
    # Create colormap based on alpha values
    colors = plt.cm.viridis(np.array(alphas))
    
    # Plot points (no error bars for deterministic ODE)
    for i, r in enumerate(results):
        ax.plot(r['violence'], r['infections'], 
               'o', markersize=12, color=colors[i], 
               alpha=0.8, label=r['label'])
    
    # Connect points to show the frontier
    ax.plot(violence, infections, 'k--', alpha=0.3, linewidth=1, zorder=1)
    
    # Identify Pareto-optimal points
    pareto_points = []
    for i, r in enumerate(results):
        is_dominated = False
        for j, other in enumerate(results):
            if i != j:
                # Check if other dominates this point (lower in both objectives)
                if (other['infections'] <= r['infections'] and 
                    other['violence'] <= r['violence'] and
                    (other['infections'] < r['infections'] or 
                     other['violence'] < r['violence'])):
                    is_dominated = True
                    break
        
        if not is_dominated:
            pareto_points.append(i)
    
    # Highlight Pareto-optimal points
    if len(pareto_points) > 1:
        pareto_violence = [results[i]['violence'] for i in pareto_points]
        pareto_infections = [results[i]['infections'] for i in pareto_points]
        ax.plot(pareto_violence, pareto_infections, 'r-', 
               linewidth=3, alpha=0.8, label='Pareto Frontier', zorder=2)
    
    # Labels and formatting
    ax.set_xlabel('Cumulative Violence (person-days)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Infections (person-days)', fontsize=14, fontweight='bold')
    ax.set_title('ODE Pareto Frontier: Infection-Violence Tradeoff', 
                fontsize=16, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add diagonal reference lines
    ax.axhline(y=min(infections), color='gray', linestyle=':', alpha=0.5)
    ax.axvline(x=min(violence), color='gray', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✓ Saved Pareto frontier plot to '{output_file}'")
    
    # Print summary
    print("\n" + "="*60)
    print("ODE PARETO FRONTIER SUMMARY")
    print("="*60)
    print(f"{'Alpha':<10} {'Label':<15} {'Infections':<15} {'Violence':<15} {'Cost':<15} {'Pareto?'}")
    print("-"*70)
    for i, r in enumerate(results):
        pareto_str = "✓" if i in pareto_points else ""
        print(f"{r['alpha']:<10.2f} {r['label']:<15} "
              f"{r['infections']:<15.1f} {r['violence']:<15.1f} "
              f"{r['cost']:<15.2f} {pareto_str}")
    print("="*60)
    
    return results, pareto_points


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # List of ODE pickle files to compare
    pkl_files = [
        'ode_run1_alpha0.pkl',
        'ode_run1_alpha0.1.pkl',
        'ode_run1_alpha0.2.pkl',
        'ode_run1_alpha0.3.pkl',
        'ode_run1_alpha0.4.pkl',
        'ode_run1_alpha0.5.pkl',
        'ode_run1_alpha0.6.pkl',
        'ode_run1_alpha0.7.pkl',
        'ode_run1_alpha0.8.pkl',
        'ode_run1_alpha0.9.pkl',
        'ode_run1_alpha1.pkl',
    ]
    
    # Option 2: Provide (filename, custom_label) tuples
    # pkl_files = [
    #     ('ode_results_alpha_0.10.pkl', 'Violence Priority'),
    #     ('ode_results_alpha_0.50.pkl', 'Balanced'),
    #     ('ode_results_alpha_1.00.pkl', 'Infection Priority'),
    # ]
    
    # Generate Pareto frontier
    results, pareto_points = plot_ode_pareto_frontier(
        pkl_files=pkl_files,
        output_file='ODE_pareto_frontier.png'
    )