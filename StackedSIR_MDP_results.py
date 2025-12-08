import numpy as np
import matplotlib.pyplot as plt
import pickle

# Load the saved MDP data
print("Loading MDP results...")
with open('mdp_run1_dt0.05_T360.pkl', 'rb') as f:
    mdp_data = pickle.load(f)

mdp = mdp_data['mdp']
policy = mdp_data['policy']
V = mdp_data['V']
initial_state = mdp_data['initial_state']
params = mdp_data['parameters']

print(f"✓ Loaded MDP with {len(mdp.states)} states")
print(f"✓ Time horizon: {params['T']} steps ({params['T'] * params['dt']:.1f} time units)")
print(f"✓ Initial state: {initial_state}")

# Set seed for reproducibility
np.random.seed(42)

# Run simulations
print("\nRunning simulations...")
n_runs = 500
trajectories = mdp.simulate(initial_state, policy, n_runs=n_runs)
print(f"✓ Completed {n_runs} simulation runs")

# Plot individual trajectories
# print("\nGenerating individual trajectory plots...")
# for i in range(min(3, n_runs)):  # Plot first 3
#     mdp.plot_trajectory(trajectories[i], filename=f'test_trajectory_{i}.png')

# Compute statistics across all runs
print("\nComputing statistics across runs...")
total_costs = [sum(cost for _, _, cost in traj) for traj in trajectories]
print(f"Average total cost: {np.mean(total_costs):.2f} ± {np.std(total_costs):.2f}")

# Extract compartment data from all trajectories
max_len = max(len(traj) for traj in trajectories)
all_SN = np.full((n_runs, max_len), np.nan)
all_IN = np.full((n_runs, max_len), np.nan)
all_RN = np.full((n_runs, max_len), np.nan)
all_SV = np.full((n_runs, max_len), np.nan)
all_IV = np.full((n_runs, max_len), np.nan)
all_RV = np.full((n_runs, max_len), np.nan)
all_actions = np.full((n_runs, max_len), np.nan)

for run_idx, traj in enumerate(trajectories):
    for t, (state, action, cost) in enumerate(traj):
        all_SN[run_idx, t] = state[0]
        all_IN[run_idx, t] = state[1]
        all_RN[run_idx, t] = state[2]
        all_SV[run_idx, t] = state[3]
        all_IV[run_idx, t] = state[4]
        all_RV[run_idx, t] = state[5]
        all_actions[run_idx, t] = action

# Plot mean trajectories with confidence bands
fig, axes = plt.subplots(3, 1, figsize=(12, 10))
t = np.arange(max_len) * params['dt']

# Plot 1: Neutral compartments
ax = axes[0]
ax.plot(t, np.nanmean(all_SN, axis=0), 'b-', label='$S_N$ (Susceptible Neutral)', linewidth=2)
#ax.fill_between(t, np.nanpercentile(all_SN, 25, axis=0), np.nanpercentile(all_SN, 75, axis=0), 
#                alpha=0.3, color='b')
ax.plot(t, np.nanmean(all_IN, axis=0), 'r-', label='$I_N$ (Infected Neutral)', linewidth=2)
#ax.fill_between(t, np.nanpercentile(all_IN, 25, axis=0), np.nanpercentile(all_IN, 75, axis=0), 
#                alpha=0.3, color='r')
ax.plot(t, np.nanmean(all_RN, axis=0), 'g-', label='$R_N$ (Recovered Neutral)', linewidth=2)
#ax.fill_between(t, np.nanpercentile(all_RN, 25, axis=0), np.nanpercentile(all_RN, 75, axis=0), 
#                alpha=0.3, color='g')
ax.set_ylabel('Population', fontsize=12)
ax.set_title('Neutral Compartments (Mean ± IQR)', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Plot 2: Violence compartments
ax = axes[1]
ax.plot(t, np.nanmean(all_SV, axis=0), 'b--', label='$S_V$ (Susceptible Violence)', linewidth=2)
#ax.fill_between(t, np.nanpercentile(all_SV, 25, axis=0), np.nanpercentile(all_SV, 75, axis=0), 
#                alpha=0.3, color='b')
ax.plot(t, np.nanmean(all_IV, axis=0), 'r--', label='$I_V$ (Infected Violence)', linewidth=2)
#ax.fill_between(t, np.nanpercentile(all_IV, 25, axis=0), np.nanpercentile(all_IV, 75, axis=0), 
#                alpha=0.3, color='r')
ax.plot(t, np.nanmean(all_RV, axis=0), 'g--', label='$R_V$ (Recovered Violence)', linewidth=2)
#ax.fill_between(t, np.nanpercentile(all_RV, 25, axis=0), np.nanpercentile(all_RV, 75, axis=0), 
#                alpha=0.3, color='g')
ax.set_ylabel('Population', fontsize=12)
ax.set_title('Violence Compartments (Mean ± IQR)', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

# Plot 3: Optimal control
ax = axes[2]
ax.plot(t, np.nanmean(all_actions, axis=0), 'k-', linewidth=2)
#ax.fill_between(t, np.nanpercentile(all_actions, 25, axis=0), 
#                np.nanpercentile(all_actions, 75, axis=0), alpha=0.3, color='gray')
ax.set_xlabel('Time', fontsize=12)
ax.set_ylabel('Control $u(t)$', fontsize=12)
ax.set_title('Optimal Control Policy (Mean ± IQR)', fontsize=14, fontweight='bold')
ax.set_ylim([-0.05, 1.05])
ax.grid(True, alpha=0.3)
ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
ax.axhline(y=1, color='k', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('MDP_T360_mean_trajectories.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n✓ Saved 'mean_trajectories.png'")
print("\nVisualization complete!")