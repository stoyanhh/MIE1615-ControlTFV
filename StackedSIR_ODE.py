import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar
import matplotlib.pyplot as plt

# Uses an ODE to approximate the MDP at a large scale 
# lambda_V currently only u*V - so no violence = no new violence despite strongest action 
# open loop policy

class StackedSIRODE:
    """
    ODE formulation of the Stacked SIR model with optimal control.
    Uses Pontryagin's Maximum Principle for numerical solution.
    """
    
    def __init__(self, P, T, alpha, beta, gamma, k_N, k_V, mu, xi, a0, a1):
        """
        Parameters:
            P     : total population size
            T     : time horizon
            alpha : weight on infections vs violence in cost
            beta  : infection contact rate
            gamma : recovery rate
            k_N   : action suppression parameter for neutral infection
            k_V   : action enhancement parameter for violence infection
            mu    : base rate for sigmoid transitions (violence)
            xi    : calming rate
            a0    : sigmoid parameter
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
    
    def sigmoid(self, z):
        """Sigmoid function with clipping for numerical stability."""
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def running_cost(self, state):
        """Running cost L(x) = α(I_N + I_V) + (1-α)(S_V + I_V + R_V)"""
        SN, IN, RN, SV, IV, RV = state
        infections = IN + IV
        violence = SV + IV + RV
        return self.alpha * infections + (1 - self.alpha) * violence
    
    def dynamics(self, t, state, u):
        """
        State dynamics: dx/dt = f(x, u)
        state = [S_N, I_N, R_N, S_V, I_V, R_V]
        """
        SN, IN, RN, SV, IV, RV = state
        V = SV + IV + RV  # Total violence compartment
        
        # Infection rates
        lambda_inf_N = self.beta * (IN + IV) * (SN / self.P) * np.exp(-self.k_N * u)
        lambda_inf_V = self.beta * (IN + IV) * (SV / self.P) * np.exp(self.k_V * u)
        
        # Violence transition rate
        sigmoid_term = self.sigmoid(self.a0 * u * (V / self.P)+self.a1*u)
        lambda_viol = self.mu * max(0, sigmoid_term - 0.5)
        
        # Calming rate
        lambda_calm = self.xi * (1 - u)
        
        # Recovery rates
        lambda_rec_N = self.gamma * IN
        lambda_rec_V = self.gamma * IV
        
        # State derivatives
        dSN = lambda_calm * SV - lambda_inf_N - lambda_viol * SN
        dIN = lambda_inf_N + lambda_calm * IV - lambda_rec_N - lambda_viol * IN
        dRN = lambda_rec_N + lambda_calm * RV - lambda_viol * RN
        dSV = lambda_viol * SN - lambda_inf_V - lambda_calm * SV
        dIV = lambda_inf_V + lambda_viol * IN - lambda_rec_V - lambda_calm * IV
        dRV = lambda_rec_V + lambda_viol * RN - lambda_calm * RV
        
        return np.array([dSN, dIN, dRN, dSV, dIV, dRV])
    
    def costate_dynamics(self, t, state, costate, u):
        """
        Costate dynamics: dλ/dt = -∂H/∂x
        where H = L(x) + λᵀf(x,u)
        """
        SN, IN, RN, SV, IV, RV = state
        lSN, lIN, lRN, lSV, lIV, lRV = costate
        V = SV + IV + RV
        
        # Common terms
        infection_pressure = (IN + IV) / self.P
        sigmoid_term = self.sigmoid(self.a0 * u * (V / self.P)+self.a1*u)
        lambda_viol = self.mu * max(0, sigmoid_term - 0.5)
        lambda_calm = self.xi * (1 - u)
        
        # Partial derivatives of sigmoid term w.r.t. violence compartments
        if V > 0:
            dsigmoid_dV = self.a0 * u / self.P * sigmoid_term * (1 - sigmoid_term)
            dlambda_viol_dV = self.mu * dsigmoid_dV if sigmoid_term > 0.5 else 0
        else:
            dlambda_viol_dV = 0
        
        # Partial derivatives of running cost
        dL_dSN = 0
        dL_dIN = self.alpha
        dL_dRN = 0
        dL_dSV = (1 - self.alpha)
        dL_dIV = self.alpha + (1 - self.alpha)  # appears in both terms
        dL_dRV = (1 - self.alpha)
        
        # Costate equations: dλ/dt = -∂H/∂x = -∂L/∂x - λᵀ∂f/∂x
        
        # ∂f/∂S_N terms
        df1_dSN = -self.beta * infection_pressure * np.exp(-self.k_N * u) - lambda_viol
        df2_dSN = self.beta * infection_pressure * np.exp(-self.k_N * u)
        df3_dSN = 0
        df4_dSN = lambda_viol
        df5_dSN = 0
        df6_dSN = 0
        
        dlSN = -(dL_dSN + lSN*df1_dSN + lIN*df2_dSN + lRN*df3_dSN + 
                 lSV*df4_dSN + lIV*df5_dSN + lRV*df6_dSN)
        
        # ∂f/∂I_N terms
        df1_dIN = -self.beta * (SN/self.P) * np.exp(-self.k_N * u)
        df2_dIN = (self.beta * (SN/self.P) * np.exp(-self.k_N * u) - 
                   self.gamma - lambda_viol)
        df3_dIN = self.gamma
        df4_dIN = -self.beta * (SV/self.P) * np.exp(self.k_V * u)
        df5_dIN = (self.beta * (SV/self.P) * np.exp(self.k_V * u) + lambda_viol)
        df6_dIN = 0
        
        dlIN = -(dL_dIN + lSN*df1_dIN + lIN*df2_dIN + lRN*df3_dIN + 
                 lSV*df4_dIN + lIV*df5_dIN + lRV*df6_dIN)
        
        # ∂f/∂R_N terms
        df1_dRN = 0
        df2_dRN = 0
        df3_dRN = -lambda_viol
        df4_dRN = 0
        df5_dRN = 0
        df6_dRN = lambda_viol
        
        dlRN = -(dL_dRN + lSN*df1_dRN + lIN*df2_dRN + lRN*df3_dRN + 
                 lSV*df4_dRN + lIV*df5_dRN + lRV*df6_dRN)
        
        # ∂f/∂S_V terms (includes violence rate dependency)
        df1_dSV = lambda_calm - dlambda_viol_dV * SN
        df2_dSV = -dlambda_viol_dV * IN
        df3_dSV = -dlambda_viol_dV * RN
        df4_dSV = (dlambda_viol_dV * SN - 
                   self.beta * infection_pressure * np.exp(self.k_V * u) - lambda_calm)
        df5_dSV = (self.beta * infection_pressure * np.exp(self.k_V * u) + 
                   dlambda_viol_dV * IN)
        df6_dSV = dlambda_viol_dV * RN
        
        dlSV = -(dL_dSV + lSN*df1_dSV + lIN*df2_dSV + lRN*df3_dSV + 
                 lSV*df4_dSV + lIV*df5_dSV + lRV*df6_dSV)
        
        # ∂f/∂I_V terms
        df1_dIV = -self.beta * (SN/self.P) * np.exp(-self.k_N * u) - dlambda_viol_dV * SN
        df2_dIV = (self.beta * (SN/self.P) * np.exp(-self.k_N * u) + 
                   lambda_calm - dlambda_viol_dV * IN)
        df3_dIV = -dlambda_viol_dV * RN
        df4_dIV = (dlambda_viol_dV * SN - 
                   self.beta * (SV/self.P) * np.exp(self.k_V * u))
        df5_dIV = (self.beta * (SV/self.P) * np.exp(self.k_V * u) + 
                   dlambda_viol_dV * IN - self.gamma - lambda_calm)
        df6_dIV = self.gamma + dlambda_viol_dV * RN
        
        dlIV = -(dL_dIV + lSN*df1_dIV + lIN*df2_dIV + lRN*df3_dIV + 
                 lSV*df4_dIV + lIV*df5_dIV + lRV*df6_dIV)
        
        # ∂f/∂R_V terms
        df1_dRV = -dlambda_viol_dV * SN
        df2_dRV = -dlambda_viol_dV * IN
        df3_dRV = lambda_calm - dlambda_viol_dV * RN
        df4_dRV = dlambda_viol_dV * SN
        df5_dRV = dlambda_viol_dV * IN
        df6_dRV = dlambda_viol_dV * RN - lambda_calm
        
        dlRV = -(dL_dRV + lSN*df1_dRV + lIN*df2_dRV + lRN*df3_dRV + 
                 lSV*df4_dRV + lIV*df5_dRV + lRV*df6_dRV)
        
        return np.array([dlSN, dlIN, dlRN, dlSV, dlIV, dlRV])
    
    def hamiltonian(self, state, costate, u):
        """
        Hamiltonian: H = L(x) + λᵀf(x,u)
        """
        L = self.running_cost(state)
        f = self.dynamics(0, state, u)
        return L + np.dot(costate, f)
    
    def optimal_control(self, state, costate):
        """
        Find optimal control u* that minimizes H at given state and costate.
        """
        # Use bounded optimization
        result = minimize_scalar(
            lambda u: self.hamiltonian(state, costate, u),
            bounds=(0, 1),
            method='bounded'
        )
        return np.clip(result.x, 0, 1)
    
    def forward_backward_sweep(self, initial_state, n_iter=50, n_points=1000):
        """
        Forward-Backward Sweep Method to solve the optimal control problem.
        
        Returns:
            t: time points
            x: state trajectory
            u: control trajectory
            J: objective value
        """
        t = np.linspace(0, self.T, n_points)
        dt = t[1] - t[0]
        
        # Initialize control with a reasonable guess
        u = np.ones(n_points) * 0.5
        
        # Storage for state and costate
        x = np.zeros((n_points, 6))
        lam = np.zeros((n_points, 6))
        
        for iteration in range(n_iter):
            # Forward sweep: solve state equation with current control
            x[0] = initial_state
            for i in range(n_points - 1):
                k1 = self.dynamics(t[i], x[i], u[i])
                k2 = self.dynamics(t[i] + dt/2, x[i] + dt*k1/2, u[i])
                k3 = self.dynamics(t[i] + dt/2, x[i] + dt*k2/2, u[i])
                k4 = self.dynamics(t[i] + dt, x[i] + dt*k3, u[i])
                x[i+1] = x[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
                
                # Ensure non-negativity
                x[i+1] = np.maximum(x[i+1], 0)
            
            # Backward sweep: solve costate equation with current state and control
            lam[-1] = np.zeros(6)  # Terminal condition
            for i in range(n_points - 1, 0, -1):
                k1 = self.costate_dynamics(t[i], x[i], lam[i], u[i])
                k2 = self.costate_dynamics(t[i-1] + dt/2, x[i-1], lam[i] - dt*k1/2, u[i-1])
                k3 = self.costate_dynamics(t[i-1] + dt/2, x[i-1], lam[i] - dt*k2/2, u[i-1])
                k4 = self.costate_dynamics(t[i-1], x[i-1], lam[i] - dt*k3, u[i-1])
                lam[i-1] = lam[i] - dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            
            # Update control using optimality condition
            u_new = np.zeros(n_points)
            for i in range(n_points):
                u_new[i] = self.optimal_control(x[i], lam[i])
            
            # Convergence check
            max_change = np.max(np.abs(u_new - u))
            u = 0.5 * u + 0.5 * u_new  # Relaxation for stability
            
            if iteration % 10 == 0:
                J = np.trapz([self.running_cost(x[i]) for i in range(n_points)], t)
                print(f"Iteration {iteration}: J = {J:.2f}, max control change = {max_change:.6f}")
            
            if max_change < 1e-4:
                print(f"Converged at iteration {iteration}")
                break
        
        # Compute final objective
        J = np.trapz([self.running_cost(x[i]) for i in range(n_points)], t)
        
        return t, x, u, J
    
    def simulate_with_control(self, initial_state, control_func):
        """
        Simulate the system with a given control function.
        control_func: callable that takes (t, state) and returns u
        """
        def ode_system(t, state):
            u = control_func(t, state)
            return self.dynamics(t, state, u)
        
        sol = solve_ivp(
            ode_system,
            [0, self.T],
            initial_state,
            dense_output=True,
            max_step=0.1
        )
        
        return sol
    
    def simulate_with_constant_control(self, initial_state, u_value, n_points=1000):
        """
        Simulate the system with a constant control value.
        
        Args:
            initial_state: initial state vector
            u_value: constant control value (0 or 1)
            n_points: number of time points for evaluation
            
        Returns:
            t: time points
            x: state trajectory
        """
        t = np.linspace(0, self.T, n_points)
        dt = t[1] - t[0]
        
        x = np.zeros((n_points, 6))
        x[0] = initial_state
        
        for i in range(n_points - 1):
            k1 = self.dynamics(t[i], x[i], u_value)
            k2 = self.dynamics(t[i] + dt/2, x[i] + dt*k1/2, u_value)
            k3 = self.dynamics(t[i] + dt/2, x[i] + dt*k2/2, u_value)
            k4 = self.dynamics(t[i] + dt, x[i] + dt*k3, u_value)
            x[i+1] = x[i] + dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            
            # Ensure non-negativity
            x[i+1] = np.maximum(x[i+1], 0)
        
        return t, x


    def compute_cumulative_metrics(t, x):
        """
        Compute cumulative metrics for violence and infections.
        
        Args:
            t: time points
            x: state trajectory [S_N, I_N, R_N, S_V, I_V, R_V]
            
        Returns:
            cumulative_violence: integral of (S_V + I_V + R_V) over time
            cumulative_infections: integral of (I_N + I_V) over time
        """
        violence = x[:, 3] + x[:, 4] + x[:, 5]  # S_V + I_V + R_V
        infections = x[:, 1] + x[:, 4]  # I_N + I_V
        
        cumulative_violence = np.trapz(violence, t)
        cumulative_infections = np.trapz(infections, t)
        
        return cumulative_violence, cumulative_infections



# -------------------------
# Example usage and visualization
# -------------------------

if __name__ == "__main__":
    # Create ODE model
    population = 1000000
    model = StackedSIRODE(
        P=population,            # population size
        T=180,             # time horizon
        alpha=0.9,         # weight on infections vs violence
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
    t, x, u, J = model.forward_backward_sweep(initial_state, n_iter=100, n_points=500)
    
    print(f"\nOptimal objective value: J = {J:.2f}")
    
    # Visualization
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Neutral compartments
    ax = axes[0]
    ax.plot(t, x[:, 0], 'b-', label='$S_N$ (Susceptible Neutral)', linewidth=2)
    ax.plot(t, x[:, 1], 'r-', label='$I_N$ (Infected Neutral)', linewidth=2)
    ax.plot(t, x[:, 2], 'g-', label='$R_N$ (Recovered Neutral)', linewidth=2)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title('Neutral Compartments', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Violence compartments
    ax = axes[1]
    ax.plot(t, x[:, 3], 'b--', label='$S_V$ (Susceptible Violence)', linewidth=2)
    ax.plot(t, x[:, 4], 'r--', label='$I_V$ (Infected Violence)', linewidth=2)
    ax.plot(t, x[:, 5], 'g--', label='$R_V$ (Recovered Violence)', linewidth=2)
    ax.set_ylabel('Population', fontsize=12)
    ax.set_title('Violence Compartments', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Optimal control
    ax = axes[2]
    ax.plot(t, u, 'k-', linewidth=2)
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Control $u(t)$', fontsize=12)
    ax.set_title('Optimal Control Policy', fontsize=14, fontweight='bold')
    ax.set_ylim([-0.05, 1.05])
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle=':', alpha=0.5)
    ax.axhline(y=1, color='k', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('stacked_sir_optimal_control_ODE.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Additional analysis
    print("\n=== Summary Statistics ===")
    print(f"Final state: S_N={x[-1,0]:.0f}, I_N={x[-1,1]:.0f}, R_N={x[-1,2]:.0f}")
    print(f"            S_V={x[-1,3]:.0f}, I_V={x[-1,4]:.0f}, R_V={x[-1,5]:.0f}")
    print(f"Average control: {np.mean(u):.3f}")
    print(f"Peak infected (neutral): {np.max(x[:,1]):.0f} at t={t[np.argmax(x[:,1])]:.1f}")
    print(f"Peak infected (violence): {np.max(x[:,4]):.0f} at t={t[np.argmax(x[:,4])]:.1f}")
    print(f"Peak total violence: {np.max(x[:,3:6].sum(axis=1)):.0f} at t={t[np.argmax(x[:,3:6].sum(axis=1))]:.1f}")