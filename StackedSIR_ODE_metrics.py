import matplotlib.pyplot as plt
from StackedSIR_ODE import StackedSIRODE
import numpy as np
import pandas as pd
import pickle

def load_ode_results(filepath):
    """Load ODE results from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def compute_cumulative_metrics(t, x):
    """
    Compute cumulative infections and violence from ODE solution.
    Uses trapezoidal integration over time.
    
    Args:
        t: time array
        x: state trajectory [S_N, I_N, R_N, S_V, I_V, R_V]
    
    Returns:
        cumulative_infections: Integral of (IN + IV) over time (person-days)
        cumulative_violence: Integral of (SV + IV + RV) over time (person-days)
    """
    # Extract compartments
    IN = x[:, 1]
    IV = x[:, 4]
    SV = x[:, 3]
    RV = x[:, 5]
    
    # Integrate over time using trapezoidal rule
    cumulative_infections = np.trapz(IN + IV, t)
    cumulative_violence = np.trapz(SV + IV + RV, t)
    
    return cumulative_infections, cumulative_violence


def compare_policies(model, initial_state, optimal_pkl_file=None, 
                     t_opt=None, x_opt=None, u_opt=None):
    """
    Compare optimal, no control (u=0), and full control (u=1) policies.
    
    Args:
        model: StackedSIRODE instance
        initial_state: initial state vector
        optimal_pkl_file: (optional) path to pickle file with optimal policy
        t_opt, x_opt, u_opt: (optional) directly provide optimal policy results
        
    Returns:
        results_dict: dictionary with all results
        metrics_df: pandas DataFrame with comparison metrics
    """
    print("=" * 60)
    print("POLICY COMPARISON")
    print("=" * 60)
    
    # Get optimal policy results
    if optimal_pkl_file is not None:
        print("\n[1/3] Loading Optimal Policy from file...")
        ode_data = load_ode_results(optimal_pkl_file)
        t_opt = ode_data['t']
        x_opt = ode_data['x']
        u_opt = ode_data['u']
        print(f"Loaded from: {optimal_pkl_file}")
    elif t_opt is not None and x_opt is not None:
        print("\n[1/3] Using provided Optimal Policy...")
    else:
        print("\n[1/3] Computing Optimal Policy...")
        t_opt, x_opt, u_opt, J_opt = model.forward_backward_sweep(
            initial_state, 
            n_iter=200, 
            n_points=500
        )
    
    cum_infections_opt, cum_violence_opt = compute_cumulative_metrics(t_opt, x_opt)
    
    # Policy 2: No control (u=0)
    print("\n[2/3] Simulating No Control Policy (u=0)...")
    t_zero, x_zero = model.simulate_with_constant_control(initial_state, u_value=0.0)
    cum_infections_zero, cum_violence_zero = compute_cumulative_metrics(t_zero, x_zero)
    
    # Policy 3: Full control (u=1)
    print("\n[3/3] Simulating Full Control Policy (u=1)...")
    t_full, x_full = model.simulate_with_constant_control(initial_state, u_value=1.0)
    cum_infections_full, cum_violence_full = compute_cumulative_metrics(t_full, x_full)
    
    # Create metrics table
    print("\n" + "=" * 60)
    print("METRICS TABLE")
    print("=" * 60)
    
    metrics_data = {
        'Policy': ['Optimal', 'No Control (u=0)', 'Full Control (u=1)'],
        'SV+IV+RV (cumulative violence)': [
            f"{cum_violence_opt:,.0f}",
            f"{cum_violence_zero:,.0f}",
            f"{cum_violence_full:,.0f}"
        ],
        'IN+IV (cumulative infections)': [
            f"{cum_infections_opt:,.0f}",
            f"{cum_infections_zero:,.0f}",
            f"{cum_infections_full:,.0f}"
        ]
    }
    
    df = pd.DataFrame(metrics_data)
    print("\n" + df.to_string(index=False))
    
    # Print raw values for analysis
    print("\n" + "=" * 60)
    print("NUMERICAL VALUES")
    print("=" * 60)
    print(f"Optimal Policy:")
    print(f"  Violence:   {cum_violence_opt:,.2f}")
    print(f"  Infections: {cum_infections_opt:,.2f}")
    print(f"\nNo Control (u=0):")
    print(f"  Violence:   {cum_violence_zero:,.2f}")
    print(f"  Infections: {cum_infections_zero:,.2f}")
    print(f"\nFull Control (u=1):")
    print(f"  Violence:   {cum_violence_full:,.2f}")
    print(f"  Infections: {cum_infections_full:,.2f}")
    
    # Store results
    results = {
        'optimal': {'t': t_opt, 'x': x_opt, 'u': u_opt, 
                   'cum_violence': cum_violence_opt, 'cum_infections': cum_infections_opt},
        'no_control': {'t': t_zero, 'x': x_zero, 'u': None,
                      'cum_violence': cum_violence_zero, 'cum_infections': cum_infections_zero},
        'full_control': {'t': t_full, 'x': x_full, 'u': None,
                        'cum_violence': cum_violence_full, 'cum_infections': cum_infections_full}
    }
    
    return results, df


def plot_policy_comparison(results, save_path='policy_comparison.png'):
    """
    Create visualization comparing all three policies.
    
    Args:
        results: dictionary from compare_policies()
        save_path: where to save the figure
    """
    t_opt = results['optimal']['t']
    x_opt = results['optimal']['x']
    u_opt = results['optimal']['u']
    
    t_zero = results['no_control']['t']
    x_zero = results['no_control']['x']
    
    t_full = results['full_control']['t']
    x_full = results['full_control']['x']
    
    cum_violence_opt = results['optimal']['cum_violence']
    cum_infections_opt = results['optimal']['cum_infections']
    cum_violence_zero = results['no_control']['cum_violence']
    cum_infections_zero = results['no_control']['cum_infections']
    cum_violence_full = results['full_control']['cum_violence']
    cum_infections_full = results['full_control']['cum_infections']
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Violence compartments over time
    ax = axes[0]
    ax.plot(t_opt, x_opt[:, 3] + x_opt[:, 4] + x_opt[:, 5], 
            'b-', label='Optimal', linewidth=2)
    ax.plot(t_zero, x_zero[:, 3] + x_zero[:, 4] + x_zero[:, 5], 
            'r--', label='No Control (u=0)', linewidth=2)
    ax.plot(t_full, x_full[:, 3] + x_full[:, 4] + x_full[:, 5], 
            'g-.', label='Full Control (u=1)', linewidth=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Total Violence (SV+IV+RV)', fontsize=11)
    ax.set_title('Violence Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Infections over time
    ax = axes[1]
    ax.plot(t_opt, x_opt[:, 1] + x_opt[:, 4], 
            'b-', label='Optimal', linewidth=2)
    ax.plot(t_zero, x_zero[:, 1] + x_zero[:, 4], 
            'r--', label='No Control (u=0)', linewidth=2)
    ax.plot(t_full, x_full[:, 1] + x_full[:, 4], 
            'g-.', label='Full Control (u=1)', linewidth=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Total Infections (IN+IV)', fontsize=11)
    ax.set_title('Infections Over Time', fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Control policies
    ax = axes[2]
    ax.plot(t_opt, u_opt, 'b-', label='Optimal', linewidth=2)
    ax.axhline(y=0, color='r', linestyle='--', label='No Control (u=0)', linewidth=2)
    ax.axhline(y=1, color='g', linestyle='-.', label='Full Control (u=1)', linewidth=2)
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Control u(t)', fontsize=11)
    ax.set_title('Control Policies', fontsize=12, fontweight='bold')
    ax.set_ylim([-0.1, 1.1])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cumulative metrics comparison
    # ax = axes[1, 1]
    # policies = ['Optimal', 'No Control\n(u=0)', 'Full Control\n(u=1)']
    # violence_values = [cum_violence_opt, cum_violence_zero, cum_violence_full]
    # infection_values = [cum_infections_opt, cum_infections_zero, cum_infections_full]
    
    # x_pos = np.arange(len(policies))
    # width = 0.35
    
    # bars1 = ax.bar(x_pos - width/2, violence_values, width, 
    #                label='Cumulative Violence', color='steelblue', alpha=0.8)
    # bars2 = ax.bar(x_pos + width/2, infection_values, width, 
    #                label='Cumulative Infections', color='coral', alpha=0.8)
    
    # ax.set_xlabel('Policy', fontsize=11)
    # ax.set_ylabel('Cumulative Count', fontsize=11)
    # ax.set_title('Cumulative Metrics Comparison', fontsize=12, fontweight='bold')
    # ax.set_xticks(x_pos)
    # ax.set_xticklabels(policies)
    # ax.legend(loc='best')
    # ax.grid(True, alpha=0.3, axis='y')
    
    # # Add value labels on bars
    # for bars in [bars1, bars2]:
    #     for bar in bars:
    #         height = bar.get_height()
    #         ax.text(bar.get_x() + bar.get_width()/2., height,
    #                f'{height:,.0f}',
    #                ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nFigure saved as '{save_path}'")
    plt.show()


# ===== USAGE EXAMPLE =====
if __name__ == "__main__":
    # Setup model
    population = 1000000
    model = StackedSIRODE(
        P=population,
        T=180,
        alpha=0.9,
        beta=0.4,
        gamma=0.1,
        k_N=2.0,
        k_V=0.5,
        mu=0.3,
        xi=0.2,
        a0=4.0,
        a1=0.01
    )
    
    # Initial state: mostly neutral susceptible, some infected
    initial_state = np.array([0.8*population, 0.1*population, 0, 0.1*population, 0, 0])
    
    # OPTION 1: Load optimal policy from pickle file
    # results, metrics_df = compare_policies(
    #     model, 
    #     initial_state, 
    #     optimal_pkl_file='ode_results_alpha_0.90.pkl'
    # )
    
    # List of ODE pickle files to compare
    pkl_files = [
        'ode_run1_alpha0.pkl',
        'ode_run1_alpha0.1.pkl',
        'ode_run1_alpha0.9.pkl',
        'ode_run1_alpha1.pkl',
    ]

    # OPTION 2: Compute optimal policy on the fly
    results, metrics_df = compare_policies(
        model, 
        initial_state,
        pkl_files[2]

    )
    
    # OPTION 3: Provide optimal policy directly (if you already have t, x, u)
    # t_opt, x_opt, u_opt, _ = model.forward_backward_sweep(initial_state)
    # results, metrics_df = compare_policies(
    #     model, 
    #     initial_state,
    #     t_opt=t_opt,
    #     x_opt=x_opt,
    #     u_opt=u_opt
    # )
    
    # Create visualization
    plot_policy_comparison(results)
    
    # Optionally save metrics to CSV
    print(metrics_df)
