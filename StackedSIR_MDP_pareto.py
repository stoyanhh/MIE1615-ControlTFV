# pareto_frontier.py
import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

def load_mdp_results(filepath):
    """Load MDP results from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def compute_cumulative_metrics(mdp, policy, initial_state, n_runs=100, seed=42):
    """
    Compute cumulative infections and violence from simulations.
    
    Returns:
        mean_infections: Average cumulative infections (IN + IV) over all runs
        mean_violence: Average cumulative violence (SV + IV + RV) over all runs
        std_infections: Standard deviation of infections
        std_violence: Standard deviation of violence
    """
    np.random.seed(seed)
    trajectories = mdp.simulate(initial_state, policy, n_runs=n_runs)
    
    infections_per_run = []
    violence_per_run = []
    
    for traj in trajectories:
        total_infections = 0
        total_violence = 0
        
        for state, action, cost in traj:
            SN, IN, RN, SV, IV, RV = state
            total_infections += (IN + IV)
            total_violence += (SV + IV + RV)
        
        infections_per_run.append(total_infections)
        violence_per_run.append(total_violence)
    
    return (np.mean(infections_per_run), np.mean(violence_per_run),
            np.std(infections_per_run), np.std(violence_per_run))

def plot_pareto_frontier(pkl_files, n_runs=100, seed=42, output_file='pareto_frontier.png'):
    """
    Create Pareto frontier plot from multiple MDP results.
    
    Args:
        pkl_files: List of paths to .pkl files (or list of tuples (path, label))
        n_runs: Number of simulation runs per MDP
        seed: Random seed for reproducibility
        output_file: Output filename for the plot
    """
    print("Loading MDP results and computing metrics...")
    
    results = []
    
    for item in pkl_files:
        # Handle both string paths and (path, label) tuples
        if isinstance(item, tuple):
            filepath, custom_label = item
        else:
            filepath = item
            custom_label = None
        
        print(f"\nProcessing: {filepath}")
        mdp_data = load_mdp_results(filepath)
        
        mdp = mdp_data['mdp']
        policy = mdp_data['policy']
        initial_state = mdp_data['initial_state']
        alpha = mdp_data['parameters']['alpha']
        
        # Compute metrics
        mean_inf, mean_viol, std_inf, std_viol = compute_cumulative_metrics(
            mdp, policy, initial_state, n_runs=n_runs, seed=seed
        )
        
        # Use custom label if provided, otherwise use alpha
        label = custom_label if custom_label else f'α={alpha:.2f}'
        
        results.append({
            'alpha': alpha,
            'label': label,
            'infections': mean_inf,
            'violence': mean_viol,
            'infections_std': std_inf,
            'violence_std': std_viol,
            'filepath': filepath
        })
        
        print(f"  α={alpha:.2f}: Infections={mean_inf:.1f}±{std_inf:.1f}, "
              f"Violence={mean_viol:.1f}±{std_viol:.1f}")
    
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
    
    # Plot points with error bars
    for i, r in enumerate(results):
        ax.errorbar(r['violence'], r['infections'], 
                   xerr=r['violence_std'], yerr=r['infections_std'],
                   fmt='o', markersize=10, capsize=5, capthick=2,
                   color=colors[i], ecolor=colors[i], alpha=0.7,
                   label=r['label'])
    
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
    ax.set_xlabel('Cumulative Violence (SV + IV + RV)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Infections (IN + IV)', fontsize=14, fontweight='bold')
    ax.set_title('Pareto Frontier: Infection-Violence Tradeoff', 
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
    print("PARETO FRONTIER SUMMARY")
    print("="*60)
    print(f"{'Alpha':<10} {'Label':<15} {'Infections':<15} {'Violence':<15} {'Pareto?'}")
    print("-"*60)
    for i, r in enumerate(results):
        pareto_str = "✓" if i in pareto_points else ""
        print(f"{r['alpha']:<10.2f} {r['label']:<15} "
              f"{r['infections']:<15.1f} {r['violence']:<15.1f} {pareto_str}")
    print("="*60)
    
    return results, pareto_points


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # List of pickle files to compare
    # Option 1: Just provide filenames (will use alpha as label)
    pkl_files = [
        'mdp_run1_dt0.05_a0.pkl',        
        'mdp_run1_dt0.05_a0.1.pkl',
        'mdp_run1_dt0.05_a0.5.pkl',
        'mdp_run1_dt0.05_a0.9.pkl',
        'mdp_run1_dt0.05_a1.pkl',
    ]
    
    # Option 2: Provide (filename, custom_label) tuples
    # pkl_files = [
    #     ('mdp_results_alpha_0.0.pkl', 'Violence Priority'),
    #     ('mdp_results_alpha_0.5.pkl', 'Balanced'),
    #     ('mdp_results_alpha_1.0.pkl', 'Infection Priority'),
    # ]
    
    # Generate Pareto frontier
    results, pareto_points = plot_pareto_frontier(
        pkl_files=pkl_files,
        n_runs=500,           # Number of simulation runs per MDP
        seed=42,              # For reproducibility
        output_file='MDP_pareto_frontier.png'
    )