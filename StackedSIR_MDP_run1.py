import numpy as np
import matplotlib.pyplot as plt
import pickle
from StackedSIR_MDP_plot import StackedSIRMDP 

population = 20
deltat = 0.05

for alpha_run in [0,0.1,0.5,0.9,1]:

    mdp = StackedSIRMDP(
        P=population,              # population size
        T= int(180 / deltat),             # time horizon - six months, one year
        alpha=alpha_run,         # weight on infections (vs violence)
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

    initial_state = (0.8*population, 0.1*population, 0, 0.1*population, 0, 0)
    state_idx = mdp.state_to_idx[initial_state]

    policy, V = mdp.value_iteration()

    # ===== SAVE EVERYTHING NEEDED =====
    print("\nSaving MDP results...")

    mdp_data = {
        # MDP object (contains states, transitions, actions, etc.)
        'mdp': mdp,
        
        # Policy and value function
        'policy': policy,
        'V': V,
        
        # Initial state for simulations
        'initial_state': initial_state,
        
        # Parameters for reference
        'parameters': {
            'P': population,
            'T': mdp.T,
            'alpha': mdp.alpha,
            'beta': mdp.beta,
            'gamma': mdp.gamma,
            'k_N': mdp.k_N,
            'k_V': mdp.k_V,
            'mu': mdp.mu,
            'xi': mdp.xi,
            'a0': mdp.a0,
            'a1': mdp.a1,
            'dt': mdp.dt
        },
    }

    # Save to file
    with open(f'mdp_run1_dt{deltat}_a{alpha_run}.pkl', 'wb') as f:
        pickle.dump(mdp_data, f)

    print("✓ Saved to 'mdp_results.pkl'")
    print(f"  - MDP object with {len(mdp.states)} states")
    print(f"  - Policy with {len(policy)} time steps")
    print(f"  - Value function")
    print(f"  - Initial state: {initial_state}")
    print(f"  - All parameters")