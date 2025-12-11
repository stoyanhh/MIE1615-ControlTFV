# MIE1615-ControlTFV
Code for "To Tweet or not to Tweet? Optimal Control of Technology Facilitated Violence During a Pandemic"

This repo includes all code for running the MDP models, running the ODE models, and generating the graphs. For ease of access, the completed ODE models are included in the repo. However, the completed MDP models are excluded because each model run is 3-6 GB large. 

To replicate the results: 

1. Clone the repo
2. Install all dependencies (numpy, matplotlib, pandas, scipy)
3. Run `python3 StackedSIR_MDP_run1.py` to run the MDP models for varying alpha values
4. Run `python3 StackedSIR_MDP_run2.py` to run the MDP models for varying time horizons
5. Run `python3 StackedSIR_ODE_run1.py`to run the ODE models for varying alpha values
6. Run `python3 StackedSIR_MDP_pareto.py` to generate a Pareto frontier with varying alpha in the MDP results
7. Run `python3 StackedSIR_MDP_results.py` to generate 10 sample trajectories of a chosen MDP model, exploring the policy, infections (S, I, R) and violence (N, V)
8. Run `python3 StackedSIR_ODE_results.py` to compare the meanfield optimal policy against the full-control and no-control alternatives
