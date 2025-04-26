import numpy as np
import matplotlib.pyplot as plt

# System parameters
x_length = 1000  # meters
y_length = 1000  # meters
n_aps = 5        # Number of APs
n_users = 20     # Number of users
M_ant = 100      # Antennas per AP
p_p = 0.2        # Pilot power (W)
p_d = 0.2        # Data power (W)
tau_c = 200      # Coherence length (symbols)
tau_p = 20       # Pilot length (symbols)
gamma = 3.76     # Path loss exponent
shadow_std = 5   # Shadow fading standard deviation (dB)
beta_0db = -35.3  # Reference path loss at 1m (dB)
n_realizations = 500  # Number of Monte-Carlo simulations

np.random.seed(42)

# Precompute AP and user positions once (for consistency across realizations)
x_pos_aps = np.random.uniform(0, x_length, n_aps)
y_pos_aps = np.random.uniform(0, y_length, n_aps)
x_pos_users = np.random.uniform(0, x_length, n_users)
y_pos_users = np.random.uniform(0, y_length, n_users)

def compute_distances(X_users, X_aps):
    distances = np.sqrt(
        (X_users[:, 0, np.newaxis] - X_aps[np.newaxis, :, 0])**2 +
        (X_users[:, 1, np.newaxis] - X_aps[np.newaxis, :, 1])**2
    )
    return distances

def compute_beta(distances):
    n_users, n_aps = distances.shape
    beta_linear = np.zeros((n_users, n_aps))
    for k in range(n_users):
        for l in range(n_aps):
            d_kl = max(distances[k, l], 1)  # Avoid log(0)
            path_loss_db = beta_0db - 10 * gamma * np.log10(d_kl)
            shadow_fading = np.random.normal(0, shadow_std)
            total_loss_db = path_loss_db + shadow_fading
            beta_linear[k, l] = 10 ** (total_loss_db / 10)
    return beta_linear

def compute_eta(beta):
    return (p_p * tau_p * beta**2) / (tau_p * p_p * beta + 1)

def compute_self_price(eta):
    return np.sum(eta**2, axis=1)

def compute_effective_gain(user_id, ap_id, eta, beta, assignments):
    # Calculate interference from users already assigned to this AP
    existing_users = [u for u, a in enumerate(assignments) if a == ap_id]
    interference = np.sum([eta[u, ap_id] * beta[user_id, ap_id] for u in existing_users])
    effective_gain = eta[user_id, ap_id]**2 - interference
    return effective_gain

def CAPS(beta, eta):
    assignments = np.full(n_users, -1)  # -1 indicates unassigned
    delta = compute_self_price(eta)
    user_order = np.argsort(-delta)  # Sort users from highest to lowest delta
    
    for k in user_order:
        effective_gains = []
        for l in range(n_aps):
            effective_gains.append(compute_effective_gain(k, l, eta, beta, assignments))
        best_ap = np.argmax(effective_gains)
        assignments[k] = best_ap
    return assignments

def calculate_se(assignments, beta, eta):
    se = np.zeros(n_users)
    for k in range(n_users):
        l = assignments[k]
        numerator = M_ant**2 * p_d * eta[k, l]**2
        interference = 0
        for j in range(n_aps):
            users_in_ap = np.where(assignments == j)[0]
            for i in users_in_ap:
                if i == k:
                    continue  # Skip self-interference
                interference += M_ant * p_d * eta[i, j] * beta[k, j]
        sinr = numerator / (interference + 1)  # Noise normalized to 1
        se[k] = ((tau_c - tau_p) / tau_c) * np.log2(1 + sinr)
    return np.sum(se)

# Main simulation loop
sum_se = []
X_aps = np.column_stack((x_pos_aps, y_pos_aps))
X_users = np.column_stack((x_pos_users, y_pos_users))
distances = compute_distances(X_users, X_aps)

for _ in range(n_realizations):
    beta = compute_beta(distances)
    eta = compute_eta(beta)
    assignments = CAPS(beta, eta)
    sum_se.append(calculate_se(assignments, beta, eta))

# Generate CDF
sum_se = np.sort(sum_se)
cdf = np.arange(1, len(sum_se) + 1) / len(sum_se)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(sum_se, cdf, 'b-', linewidth=2, label='Proposed CAPS')
plt.xlabel('Sum Spectral Efficiency (bits/s/Hz)')
plt.ylabel('CDF')
plt.title('CDF of Sum Spectral Efficiency')
plt.grid(True)
plt.legend()
plt.show()