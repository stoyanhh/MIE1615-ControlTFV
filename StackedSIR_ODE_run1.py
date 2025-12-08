import numpy as np
import matplotlib.pyplot as plt
import pickle
from StackedSIR_ODE import StackedSIRODE

population = 1000000
alphas =[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

for alpha_run in alphas: 
    print("Solving model with alpha: ",alpha_run)
    ode = StackedSIRODE(
        P=population,            # population size
        T=180,             # time horizon
        alpha=alpha_run,         # weight on infections vs violence
        beta=0.4,          # infection rate
        gamma=0.1,         # recovery rate
        k_N=2.0,           # neutral infection suppression
        k_V=0.5,           # violence infection enhancement
        mu=0.3,            # violence transition rate
        xi=0.2,            # calming rate
        a0=4.0,             # sigmoid violence violence parameter
        a1 = 0.01           # sigmoid violence tweet parameter
    )

    # Initial state: mostly neutral susceptible, some infected
    initial_state = (0.8*population, 0.1*population, 0, 0.1*population, 0, 0)

    print("Solving optimal control problem...")
    t, x, u, J = ode.forward_backward_sweep(initial_state, n_iter=500, n_points=500)

    print(f"\nOptimal objective value: J = {J:.2f}")

    # ===== SAVE EVERYTHING NEEDED =====
    print("\nSaving ODE results...")

    ode_data = {
        # MDP object (contains states, transitions, actions, etc.)
        'ode': ode,
        
        # Optimal control solution
        't': t,              # Time array
        'x': x,              # State trajectories (n_points x 6)
        'u': u,              # Control trajectory (n_points,)
        'J': J,              # Total cost (scalar)
        
        # Initial state for simulations
        'initial_state': initial_state,
        
        # Parameters for reference
        'parameters': {
            'P': population,
            'T': ode.T,
            'alpha': ode.alpha,
            'beta': ode.beta,
            'gamma': ode.gamma,
            'k_N': ode.k_N,
            'k_V': ode.k_V,
            'mu': ode.mu,
            'xi': ode.xi,
            'a0': ode.a0,
            'a1': ode.a1      },
    }

    # Save to file
    with open(f'ode_run1_alpha{alpha_run}.pkl', 'wb') as f:
        pickle.dump(ode_data, f)

    print("✓ Saved", alpha_run)

