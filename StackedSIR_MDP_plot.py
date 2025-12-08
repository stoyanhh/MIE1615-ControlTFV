import numpy as np
from itertools import product
from collections import defaultdict
import pickle
import matplotlib.pyplot as plt

class StackedSIRMDP:
    """
    MDP formulation of the Stacked SIR model with epidemic and violence dynamics.
    
    State: (SN, IN, RN, SV, IV, RV) where compartments sum to P (total population)
    Action: u in {0, 0.25, 0.5, 0.75, 1}
    Objective: Minimize alpha*(IN + IV) + (1-alpha)*(SV + IV + RV)
    """
    
    def __init__(self, P, T, alpha, beta, gamma, k_N, k_V, mu, xi, a0, a1, dt=0.01):
        """
        Parameters:
            P     : total population size
            T     : time horizon (number of steps)
            alpha : weight on infections vs violence in cost
            beta  : infection contact rate
            gamma : recovery rate
            k     : action-suppression parameter for infection
            mu    : base rate for sigmoid transitions
            a0    : sigmoid parameter
            dt    : time step for discrete approximation
        """
        self.P = P
        self.T = T
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k_N = k_N
        self.k_V = k_V
        self.mu = mu
        self.xi = xi
        self.a0 = a0
        self.a1 = a1
        self.dt = dt
        
        # Action space
        self.actions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        
        # Generate state space
        print("Generating state space...")
        self.states = self._generate_states()
        self.state_to_idx = {s: i for i, s in enumerate(self.states)}
        print(f"State space size: {len(self.states)}")
        
        # Precompute transition probabilities for each action
        print("Computing transition probabilities...")
        self.transitions = {}
        for u in self.actions:
            print(f"  Action u={u}")
            self.transitions[u] = self._compute_transitions(u)
    
    def _generate_states(self):
        """Generate all valid states (SN, IN, RN, SV, IV, RV) that sum to P."""
        states = []
        # Enumerate all non-negative integer tuples that sum to P
        for SN in range(self.P + 1):
            for IN in range(self.P + 1 - SN):
                for RN in range(self.P + 1 - SN - IN):
                    for SV in range(self.P + 1 - SN - IN - RN):
                        for IV in range(self.P + 1 - SN - IN - RN - SV):
                            RV = self.P - SN - IN - RN - SV - IV
                            if RV >= 0:
                                states.append((SN, IN, RN, SV, IV, RV))
        return states
    
    def sigmoid(self, z):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def _compute_transitions(self, u):
        """
        Compute transition probability matrix for action u.
        Returns a dictionary: state_idx -> [(next_state_idx, probability), ...]
        """
        transitions = {}
        
        for state_idx, state in enumerate(self.states):
            SN, IN, RN, SV, IV, RV = state
            P = self.P
            V = SV + IV + RV
            
            # Compute hazard rates (only non-zero if compartment non-empty)
            lambda_inf_N = self.beta * (IN + IV) * (SN / P) * np.exp(-self.k_N * u) if (SN > 0 and (IN + IV) > 0) else 0
            lambda_inf_V = self.beta * (IN + IV) * (SV / P) * np.exp(self.k_V * u) if (SV > 0 and (IN + IV) > 0) else 0
            
            lambda_viol = self.mu * max(0, self.sigmoid(self.a0 * u * (V/P) + self.a1 * u) - 0.5)
            
            Lambda_viol_SN = SN * lambda_viol
            Lambda_viol_IN = IN * lambda_viol
            Lambda_viol_RN = RN * lambda_viol
            
            lambda_calm = self.xi * (1 - u)
            
            Lambda_calm_SV = SV * lambda_calm
            Lambda_calm_IV = IV * lambda_calm
            Lambda_calm_RV = RV * lambda_calm
            
            Lambda_rec_N = self.gamma * IN
            Lambda_rec_V = self.gamma * IV
            
            # Build event dictionary
            events = {
                "infection_N": lambda_inf_N,
                "infection_V": lambda_inf_V,
                "recover_N": Lambda_rec_N,
                "recover_V": Lambda_rec_V,
                "violence_SN": Lambda_viol_SN,
                "violence_IN": Lambda_viol_IN,
                "violence_RN": Lambda_viol_RN,
                "calm_SV": Lambda_calm_SV,
                "calm_IV": Lambda_calm_IV,
                "calm_RV": Lambda_calm_RV
            }
            
            # Convert to probabilities
            probs = {event: rate * self.dt for event, rate in events.items()}
            total_prob = sum(probs.values())
            p_nothing = max(0, 1 - total_prob)
            
            # Normalize if needed
            if total_prob > 1:
                scale = 1 / (total_prob + p_nothing)
                probs = {event: p * scale for event, p in probs.items()}
                p_nothing = p_nothing * scale
            
            # Apply each event to get next states
            next_states_probs = []
            
            # Infection N: SN -> IN
            if probs["infection_N"] > 0 and SN > 0:
                next_state = (SN-1, IN+1, RN, SV, IV, RV)
                next_states_probs.append((next_state, probs["infection_N"]))
            
            # Infection V: SV -> IV
            if probs["infection_V"] > 0 and SV > 0:
                next_state = (SN, IN, RN, SV-1, IV+1, RV)
                next_states_probs.append((next_state, probs["infection_V"]))
            
            # Recovery N: IN -> RN
            if probs["recover_N"] > 0 and IN > 0:
                next_state = (SN, IN-1, RN+1, SV, IV, RV)
                next_states_probs.append((next_state, probs["recover_N"]))
            
            # Recovery V: IV -> RV
            if probs["recover_V"] > 0 and IV > 0:
                next_state = (SN, IN, RN, SV, IV-1, RV+1)
                next_states_probs.append((next_state, probs["recover_V"]))
            
            # Violence transitions
            if probs["violence_SN"] > 0 and SN > 0:
                next_state = (SN-1, IN, RN, SV+1, IV, RV)
                next_states_probs.append((next_state, probs["violence_SN"]))
            
            if probs["violence_IN"] > 0 and IN > 0:
                next_state = (SN, IN-1, RN, SV, IV+1, RV)
                next_states_probs.append((next_state, probs["violence_IN"]))
            
            if probs["violence_RN"] > 0 and RN > 0:
                next_state = (SN, IN, RN-1, SV, IV, RV+1)
                next_states_probs.append((next_state, probs["violence_RN"]))
            
            # Calming transitions
            if probs["calm_SV"] > 0 and SV > 0:
                next_state = (SN+1, IN, RN, SV-1, IV, RV)
                next_states_probs.append((next_state, probs["calm_SV"]))
            
            if probs["calm_IV"] > 0 and IV > 0:
                next_state = (SN, IN+1, RN, SV, IV-1, RV)
                next_states_probs.append((next_state, probs["calm_IV"]))
            
            if probs["calm_RV"] > 0 and RV > 0:
                next_state = (SN, IN, RN+1, SV, IV, RV-1)
                next_states_probs.append((next_state, probs["calm_RV"]))
            
            # Nothing happens
            if p_nothing > 0:
                next_states_probs.append((state, p_nothing))
            
            # Convert to indices and store - ALWAYS create entry even if empty
            transitions[state_idx] = []
            for next_state, prob in next_states_probs:
                if next_state in self.state_to_idx:
                    next_idx = self.state_to_idx[next_state]
                    transitions[state_idx].append((next_idx, prob))
            
            # Normalize probabilities to ensure they sum to 1
            if transitions[state_idx]:
                total_p = sum(p for _, p in transitions[state_idx])
                if total_p > 0:
                    transitions[state_idx] = [(idx, p/total_p) for idx, p in transitions[state_idx]]
        
        return transitions
    
    def cost(self, state):
        """
        Stage cost: alpha*(IN + IV) + (1-alpha)*(SV + IV + RV)
        """
        SN, IN, RN, SV, IV, RV = state
        infections = IN + IV
        violence = SV + IV + RV
        return self.alpha * infections + (1 - self.alpha) * violence
    
    def value_iteration(self, max_iter=1000, tol=1e-6):
        """
        Backward value iteration for finite horizon MDP.
        
        Returns:
            policy: list of length T, where policy[t][state_idx] = optimal action
            V: list of length T+1, where V[t][state_idx] = value function at time t
        """
        n_states = len(self.states)
        
        # Initialize value function at terminal time
        V = [np.zeros(n_states) for _ in range(self.T + 1)]
        policy = [np.zeros(n_states, dtype=int) for _ in range(self.T)]
        
        # Terminal cost
        for state_idx, state in enumerate(self.states):
            V[self.T][state_idx] = self.cost(state)
        
        # Backward iteration
        print("\nRunning value iteration...")
        for t in range(self.T - 1, -1, -1):
            if t % 10 == 0:
                print(f"  Time step {t}/{self.T}")
            
            for state_idx in range(n_states):
                state = self.states[state_idx]
                min_cost = float('inf')
                best_action_idx = 0
                
                for action_idx, u in enumerate(self.actions):
                    # Expected cost-to-go
                    expected_cost = self.cost(state)
                    
                    # Add expected future value
                    for next_idx, prob in self.transitions[u][state_idx]:
                        expected_cost += prob * V[t + 1][next_idx]
                    
                    if expected_cost < min_cost:
                        min_cost = expected_cost
                        best_action_idx = action_idx
                
                V[t][state_idx] = min_cost
                policy[t][state_idx] = best_action_idx
        
        print("Value iteration complete!")
        return policy, V
    
    def simulate(self, initial_state, policy, n_runs=100):
        """
        Simulate the MDP using the computed policy.
        
        Returns:
            trajectories: list of trajectories, each is a list of (state, action, cost)
        """
        trajectories = []
        
        for run in range(n_runs):
            trajectory = []
            state = initial_state
            
            for t in range(self.T):
                state_idx = self.state_to_idx.get(state)
                if state_idx is None:
                    break
                
                action_idx = policy[t][state_idx]
                action = self.actions[action_idx]
                cost = self.cost(state)
                
                trajectory.append((state, action, cost))
                
                # Sample next state
                if self.transitions[action][state_idx]:
                    next_indices, probs = zip(*self.transitions[action][state_idx])
                    next_idx = np.random.choice(next_indices, p=probs)
                    state = self.states[next_idx]
                else:
                    # Absorbing state - stay here
                    break
            
            trajectories.append(trajectory)
        
        return trajectories
    
    def plot_trajectory(self, trajectory, filename='stacked_sir_optimal_control_MDP.png'):
        """
        Plot the trajectory with compartments and control policy.
        
        Args:
            trajectory: A single trajectory from simulate()
            filename: Output filename for the plot
        """
        # Extract data from trajectory
        t = np.arange(len(trajectory))
        
        # Extract compartments (state) and control (action)
        states = np.array([state for state, _, _ in trajectory])
        actions = np.array([action for _, action, _ in trajectory])
        
        # Separate compartments
        SN = states[:, 0]
        IN = states[:, 1]
        RN = states[:, 2]
        SV = states[:, 3]
        IV = states[:, 4]
        RV = states[:, 5]
        
        # Visualization
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Neutral compartments
        ax = axes[0]
        ax.plot(t, SN, 'b-', label='$S_N$ (Susceptible Neutral)', linewidth=2)
        ax.plot(t, IN, 'r-', label='$I_N$ (Infected Neutral)', linewidth=2)
        ax.plot(t, RN, 'g-', label='$R_N$ (Recovered Neutral)', linewidth=2)
        ax.set_ylabel('Population', fontsize=12)
        ax.set_title('Neutral Compartments', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Violence compartments
        ax = axes[1]
        ax.plot(t, SV, 'b--', label='$S_V$ (Susceptible Violence)', linewidth=2)
        ax.plot(t, IV, 'r--', label='$I_V$ (Infected Violence)', linewidth=2)
        ax.plot(t, RV, 'g--', label='$R_V$ (Recovered Violence)', linewidth=2)
        ax.set_ylabel('Population', fontsize=12)
        ax.set_title('Violence Compartments', fontsize=14, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Optimal control
        ax = axes[2]
        ax.plot(t, actions, 'k-', linewidth=2)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Control $u(t)$', fontsize=12)
        ax.set_title('Optimal Control Policy', fontsize=14, fontweight='bold')
        ax.set_ylim([-0.05, 1.05])
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
        ax.axhline(y=1, color='k', linestyle=':', alpha=0.5)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"\nPlot saved to {filename}")


# -------------------------
# Example usage
# -------------------------

if __name__ == "__main__":
    # Create MDP with small population
    population = 20
    deltat = 0.05
    mdp = StackedSIRMDP(
        P=population,              # population size
        T= int(180 / deltat),             # time horizon - six months, one year
        alpha=0.9,         # weight on infections (vs violence)
        beta=0.4,          # infection rate
        gamma=0.1,         # recovery rate
        k_N=2.0,           # action neutral infection reduction parameter
        k_V = 0.5,         # action neutral violence increase parameter
        mu=0.3,            # violence rate
        xi = 0.2,          # calming rate
        a0=4.0,            # sigmoid action x violent people = violence
        a1 = 0.01,         # sigmoid action = violence
        dt=deltat             # time step
    )

    print("\n" + "="*60)
    print("DIAGNOSTIC: Computing maximum rates")
    print("="*60)
    
    max_rates = []
    for state in mdp.states:
        for u in mdp.actions:
            SN, IN, RN, SV, IV, RV = state
            P = mdp.P
            V = SV + IV + RV
            
            # Compute all rates for this state/action (same as in _compute_transitions)
            lambda_inf_N = mdp.beta * (IN + IV) * (SN / P) * np.exp(-mdp.k_N * u) if (SN > 0 and (IN + IV) > 0) else 0
            lambda_inf_V = mdp.beta * (IN + IV) * (SV / P) * np.exp(mdp.k_V * u) if (SV > 0 and (IN + IV) > 0) else 0
            
            lambda_viol = mdp.mu * max(0, mdp.sigmoid(mdp.a0 * u * (V/P) + mdp.a1 * u) - 0.5)
            lambda_calm = mdp.xi * (1 - u)
            
            # Total outgoing rates per compartment
            rate_SN = lambda_inf_N + lambda_viol*SN if SN > 0 else 0
            rate_IN = mdp.gamma*IN + lambda_viol*IN if IN > 0 else 0
            rate_RN = lambda_viol*RN if RN > 0 else 0
            rate_SV = lambda_inf_V + lambda_calm*SV if SV > 0 else 0
            rate_IV = mdp.gamma*IV + lambda_calm*IV if IV > 0 else 0
            rate_RV = lambda_calm*RV if RV > 0 else 0
            
            total_rate = max(rate_SN, rate_IN, rate_RN, rate_SV, rate_IV, rate_RV)
            max_rates.append(total_rate)
    
    max_rate_overall = max(max_rates)
    median_rate = np.median(max_rates)
    
    print(f"\nMax total rate across all states/actions: {max_rate_overall:.4f}")
    print(f"Median total rate: {median_rate:.4f}")
    print(f"Current dt: {mdp.dt}")
    print(f"Max rate * dt: {max_rate_overall * mdp.dt:.4f} (should be << 1 for Euler)")
    print(f"\nFor Euler accuracy, need dt < {1/max_rate_overall:.4f}")
    print(f"For conservative Euler, recommend dt < {0.5/max_rate_overall:.4f}")
    
    if max_rate_overall * mdp.dt > 0.5:
        print("\n⚠️  WARNING: dt is too large! Euler approximation is inaccurate.")
        print("   Consider reducing dt or using uniformization method.")
    elif max_rate_overall * mdp.dt > 0.2:
        print("\n⚠️  CAUTION: dt is marginal. Results may not be robust.")
    else:
        print("\n✓ dt appears acceptable for Euler method.")
    
    print("="*60 + "\n")
    
    # Analyze transition structure for initial state
    initial_state = (0.8*population, 0.1*population, 0, 0.1*population, 0, 0)
    state_idx = mdp.state_to_idx[initial_state]
    
    print(f"\nTransition analysis for initial state {initial_state}:")
    for u in mdp.actions[:3]:  # Just show a few actions
        transitions = mdp.transitions[u][state_idx]
        n_transitions = len(transitions)
        print(f"\nAction u={u:.2f}: {n_transitions} possible next states")
        if n_transitions > 0 and n_transitions <= 15:
            for next_idx, prob in sorted(transitions, key=lambda x: -x[1])[:10]:
                next_state = mdp.states[next_idx]
                print(f"  -> {next_state} with prob {prob:.4f}")
    
    # Solve MDP
    policy, V = mdp.value_iteration()

    np.random.seed(1012642847)
    
    # Simulate from initial state
    print(f"\nSimulating from initial state: {initial_state}")
    num_runs = 10
    trajectories = mdp.simulate(initial_state, policy, n_runs=num_runs)
    
    # Print sample trajectory
    print("\nSample trajectory (first 20 steps):")
    print("t\tState\t\t\t\t\tAction\tCost")
    for t, (state, action, cost) in enumerate(trajectories[0][:20]):
        print(f"{t}\t{state}\t{action:.2f}\t{cost:.2f}")
    
    # Compute average total cost
    total_costs = [sum(cost for _, _, cost in traj) for traj in trajectories]
    print(f"\nAverage total cost over {len(trajectories)} runs: {np.mean(total_costs):.2f} ± {np.std(total_costs):.2f}")
    
    # Analyze what actions are taken across different states
    print("\nPolicy distribution at t=0:")
    action_counts = {u: 0 for u in mdp.actions}
    for state_idx in range(len(mdp.states)):
        action_idx = policy[0][state_idx]
        action = mdp.actions[action_idx]
        action_counts[action] += 1
    
    for action, count in sorted(action_counts.items()):
        pct = 100 * count / len(mdp.states)
        print(f"  u={action:.2f}: {count} states ({pct:.1f}%)")
    
    # Plot the first trajectory
    print("\nGenerating plots...")
    for i in range(len(trajectories)):
        mdp.plot_trajectory(trajectories[i],filename=f'stacked_sir_optimal_control_MDP_{i}.png')

    # ### Check robustness to delta t ###

    # plt.figure(figsize=(10, 6))

    # for dt in [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 1.0]:
    #     # Create new MDP with this dt
    #     mdp_dt = StackedSIRMDP(
    #         P=population,
    #         T=300,
    #         alpha=0.9,
    #         beta=0.4,
    #         gamma=0.1,
    #         k_N=2.0,
    #         k_V=0.5,
    #         mu=0.3,
    #         xi=0.2,
    #         a0=4.0,
    #         a1=0.01,
    #         dt=dt
    #     )
        
    #     policy, V = mdp_dt.value_iteration()
        
    #     trajectory = mdp_dt.simulate(initial_state, policy, n_runs=1)[0]
        
    #     # Extract actions
    #     actions = [action for _, action, _ in trajectory]
    #     t = np.arange(len(actions))
        
    #     # Plot
    #     plt.plot(t, actions, linewidth=2, label=f'dt={dt}')

    # plt.xlabel('Time Step', fontsize=12)
    # plt.ylabel('Control $u(t)$', fontsize=12)
    # plt.title('Optimal Control Policy for Different dt Values', fontsize=14, fontweight='bold')
    # plt.ylim([-0.05, 1.05])
    # plt.legend(loc='best')
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # plt.savefig('dt_sensitivity_comparison.png', dpi=300, bbox_inches='tight')
    # plt.show()