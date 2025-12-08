import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

a0 = 4 
mu = 2
beta = 1
kN = 2
kV = 0.5

u_vals = np.linspace(0, 1, 400)
vp_values = [0, 0.25, 0.5, 0.75, 1.0]

plt.figure(figsize=(8,6))


f = 1 / (1+np.exp(-u_vals)) - 0.5
f = mu*np.maximum(0, f)
plt.plot(u_vals,f)

plt.xlabel("u (action)")
plt.ylabel("max(0, sigmoid(a0 * u * (V/P)) - 0.5)")
plt.title("Violence rate vs control action")
plt.legend()
plt.grid(True)
plt.show()

# plt.figure(figsize=(8,6))

# for vp in vp_values:
#     f = sigmoid(a0 * u_vals * vp) - 0.5
#     f = mu*np.maximum(0, f)
#     plt.plot(u_vals, f, label=f"V/P = {vp}")

# plt.xlabel("u (action)")
# plt.ylabel("max(0, sigmoid(a0 * u * (V/P)) - 0.5)")
# plt.title("Violence rate vs control action")
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8,6))

# for vp in vp_values:
#     f = beta*vp*np.exp(-kN*u_vals)
#     plt.plot(u_vals, f, label=f"I/P = {vp}")

# plt.xlabel("u (action)")
# plt.ylabel("beta*(IN+IV)/P*e^(-ku)")
# plt.title("Neutral: Infection rate vs control action")
# plt.legend()
# plt.grid(True)
# plt.show()

# plt.figure(figsize=(8,6))

# for vp in vp_values:
#     f = beta*vp*np.exp(kV*u_vals)
#     plt.plot(u_vals, f, label=f"I/P = {vp}")

# plt.xlabel("u (action)")
# plt.ylabel("beta*(IN+IV)/P*e^(ku)")
# plt.title("Violent: Infection rate vs control action")
# plt.legend()
# plt.grid(True)
# plt.show()