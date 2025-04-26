# import numpy as np
# import matplotlib.pyplot as plt

# # System parameters
# x_length = 100  # meters
# y_length = 100  # meters
# z_length = 20   # meters
# n_aps = 5     # Number of APs
# n_users = 10    # Number of users
# M = 100         # Antennas per AP
# p_p = 0.2       # Pilot power (W)
# p_d = 0.2       # Data power (W)
# tau_c = 200     # Coherence length (symbols)
# tau_p = 30      # Pilot length (symbols)
# gamma = 3.76    # Path loss exponent
# shadow_std = 5  # Shadow fading standard deviation (dB)

# # Set random seed for reproducibility
# np.random.seed(42)

# # Generate AP and user positions
# x_pos_aps = np.random.uniform(0, x_length, n_aps)
# y_pos_aps = np.random.uniform(0, y_length, n_aps)
# z_pos_aps = np.random.normal(z_length/2, z_length/4, n_aps)
# z_pos_aps = np.clip(z_pos_aps, 0, z_length)

# x_pos_users = np.random.uniform(0, x_length, n_users)
# y_pos_users = np.random.uniform(0, y_length, n_users)
# z_pos_users = np.random.normal(z_length/2, z_length/6, n_users)
# z_pos_users = np.clip(z_pos_users, 0, z_length)

# X_aps = np.column_stack((x_pos_aps, y_pos_aps, z_pos_aps))
# X_users = np.column_stack((x_pos_users, y_pos_users, z_pos_users))

# # Random AP assignment function
# def random_ap_assignment(n_users, n_aps):
#     """Randomly assign each user to one AP."""
#     return np.random.randint(0, n_aps, size=n_users)

# # Compute distances
# def compute_distances(X_users, X_aps):
#     """Compute Euclidean distances between users and APs."""
#     n_users, n_aps = len(X_users), len(X_aps)
#     distances = np.zeros((n_users, n_aps))
#     for k in range(n_users):
#         for l in range(n_aps):
#             distances[k, l] = np.sqrt(np.sum((X_users[k] - X_aps[l])**2))
#     return distances

# # Compute large-scale fading coefficients
# def compute_beta(distances, gamma, shadow_std):
#     """Compute large-scale fading coefficients using 3GPP LTE model."""
#     n_users, n_aps = distances.shape
#     beta_dB = np.zeros((n_users, n_aps))
#     beta_linear = np.zeros((n_users, n_aps))
#     for k in range(n_users):
#         for l in range(n_aps):
#             d_kl = max(distances[k, l], 1)  # Avoid division by zero
#             path_loss = -10 - gamma * np.log10(d_kl)  # Adjusted reference path loss
#             shadow_fading = np.random.normal(0, shadow_std)
#             beta_dB[k, l] = path_loss + shadow_fading
#             beta_linear[k, l] = 10 ** (beta_dB[k, l] / 10)
#     return beta_linear

# # Compute eta_kl
# def compute_eta(beta, p_p, tau_p):
#     """Compute channel estimation variance eta_kl."""
#     n_users, n_aps = beta.shape
#     eta = np.zeros((n_users, n_aps))
#     for k in range(n_users):
#         for l in range(n_aps):
#             beta_kl = beta[k, l]
#             eta[k, l] = (p_p * tau_p * beta_kl**2) / (tau_p * p_p * beta_kl + 1)
#     return eta

# # Compute spectral efficiency
# def compute_spectral_efficiency(assignments, beta, eta, M, p_d, tau_c, tau_p):
#     """Compute spectral efficiency for each user."""
#     n_users, n_aps = beta.shape
#     S = np.zeros(n_users)
    
#     # Construct U_j sets
#     U = [[] for _ in range(n_aps)]
#     for k in range(n_users):
#         l_k = assignments[k]
#         U[l_k].append(k)
    
#     for k in range(n_users):
#         l_k = assignments[k]  # AP assigned to user k
#         # Numerator
#         numerator = M**2 * p_d * eta[k, l_k]**2
#         # Denominator (interference + noise)
#         interference = 0
#         for j in range(n_aps):
#             for i in U[j]:
#                 interference += M * p_d * eta[i, j] * beta[k, j]
#         denominator = interference + 1  # Normalized noise
#         # SINR
#         SINR_k = numerator / denominator
#         # Spectral efficiency
#         S[k] = ((tau_c - tau_p) / tau_c) * np.log2(1 + SINR_k)
    
#     return S

# # Monte-Carlo simulation for CDF
# n_realizations = 500
# sum_se = []

# for _ in range(n_realizations):
#     assignments = random_ap_assignment(n_users, n_aps)
#     distances = compute_distances(X_users, X_aps)
#     beta = compute_beta(distances, gamma, shadow_std)
#     eta = compute_eta(beta, p_p, tau_p)
#     S = compute_spectral_efficiency(assignments, beta, eta, M, p_d, tau_c, tau_p)
#     sum_se.append(np.sum(S))

# # Print user-to-AP assignments and SE for the last realization
# print("User-to-AP Assignments and Spectral Efficiency (bits/s/Hz):")
# for k in range(n_users):
#     print(f"User {k} assigned to AP {assignments[k]}: SE = {S[k]:.4f}")

# # Compute empirical CDF
# sum_se = np.sort(sum_se)
# cdf = np.arange(1, len(sum_se) + 1) / len(sum_se)

# # Plot CDF
# plt.figure(figsize=(8, 6))
# plt.plot(sum_se, cdf, 'b-', label='Random AP Selection')
# plt.xlim([0,30])
# plt.title('CDF of Sum Spectral Efficiency')
# plt.xlabel('Sum Spectral Efficiency (bits/s/Hz)')
# plt.ylabel('CDF')
# plt.grid(True)
# plt.legend()
# # plt.savefig('sum_se_cdf.png')
# plt.show()


import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# System parameters
x_length = 1000  # meters
y_length = 1000  # meters
n_aps = 5        # Number of APs
n_users = 10     # Number of users
M = 100          # Antennas per AP
p_p = 0.2        # Pilot power (W)
p_d = 0.2        # Data power (W)
tau_c = 100      # Coherence length (symbols)
tau_p = 10       # Pilot length (symbols)
gamma = 3.76     # Path loss exponent
shadow_std = 5   # Shadow fading standard deviation (dB)
beta_0db = 8.53  # Reference path loss

np.random.seed(42)

# Generate AP and user positions (2D)
x_pos_aps = np.random.uniform(0, x_length, n_aps)
y_pos_aps = np.random.uniform(0, y_length, n_aps)
x_pos_users = np.random.uniform(0, x_length, n_users)
y_pos_users = np.random.uniform(0, y_length, n_users)

X_aps = np.column_stack((x_pos_aps, y_pos_aps))
X_users = np.column_stack((x_pos_users, y_pos_users))

# Cluster APs using K-means++ (done once since positions are fixed)
C = 2  # Number of clusters
kmeans = KMeans(n_clusters=C, init='k-means++', random_state=42)
ap_labels = kmeans.fit_predict(X_aps)

# Compute distances (2D)
def compute_distances(X_users, X_aps):
    n_users, n_aps = len(X_users), len(X_aps)
    distances = np.zeros((n_users, n_aps))
    for k in range(n_users):
        for l in range(n_aps):
            distances[k, l] = np.sqrt((X_users[k, 0] - X_aps[l, 0])**2 +
                                      (X_users[k, 1] - X_aps[l, 1])**2)
    return distances

# Compute large-scale fading coefficients
def compute_beta(distances, gamma, shadow_std):
    n_users, n_aps = distances.shape
    beta_linear = np.zeros((n_users, n_aps))
    for k in range(n_users):
        for l in range(n_aps):
            d_kl = max(distances[k, l], 1)  # Avoid log(0)
            path_loss = -beta_0db - gamma * np.log10(d_kl)
            shadow_fading = np.random.normal(0, shadow_std)
            beta_dB = path_loss + shadow_fading
            beta_linear[k, l] = 10 ** (beta_dB / 10)
    return beta_linear

# Compute eta_kl
def compute_eta(beta, p_p, tau_p):
    eta = (p_p * tau_p * beta**2) / (tau_p * p_p * beta + 1)
    return eta

# Compute spectral efficiency
def compute_spectral_efficiency(assignments, beta, eta, M, p_d, tau_c, tau_p):
    n_users, n_aps = beta.shape
    S = np.zeros(n_users)
    U = [[] for _ in range(n_aps)]
    for k in range(n_users):
        l_k = assignments[k]
        U[l_k].append(k)
    for k in range(n_users):
        l_k = assignments[k]
        numerator = M**2 * p_d * eta[k, l_k]**2
        interference = 0
        for j in range(n_aps):
            for i in U[j]:
                interference += M * p_d * eta[i, j] * beta[k, j]
        denominator = interference + 1
        SINR_k = numerator / denominator
        S[k] = ((tau_c - tau_p) / tau_c) * np.log2(1 + SINR_k)
    return S

# Monte-Carlo simulation for CDF
n_realizations = 500
sum_se = []

for iter in range(n_realizations):
    distances = compute_distances(X_users, X_aps)
    beta = compute_beta(distances, gamma, shadow_std)
    eta = compute_eta(beta, p_p, tau_p)
    
    # CAPS algorithm assignment
    assignments = np.zeros(n_users, dtype=int)
    for k in range(n_users):
        # Compute distances to cluster centroids
        distances_to_centroids = np.linalg.norm(X_users[k] - kmeans.cluster_centers_, axis=1)
        cluster_idx = np.argmin(distances_to_centroids)
        # APs in this cluster
        ap_in_cluster = np.where(ap_labels == cluster_idx)[0]
        if len(ap_in_cluster) > 0:
            # Select AP with largest beta[k, l] in the cluster
            beta_k_cluster = beta[k, ap_in_cluster]
            best_ap_idx = ap_in_cluster[np.argmax(beta_k_cluster)]
            assignments[k] = best_ap_idx
        else:
            assignments[k] = np.random.randint(0, n_aps)  # Fallback

    S = compute_spectral_efficiency(assignments, beta, eta, M, p_d, tau_c, tau_p)
    sum_se.append(np.sum(S))

# Print assignments and SE for the last realization
print("User-to-AP Assignments and Spectral Efficiency (bits/s/Hz):")
for k in range(n_users):
    print(f"User {k} assigned to AP {assignments[k]}: SE = {S[k]:.4f}")

# Compute empirical CDF
sum_se = np.sort(sum_se)
cdf = np.arange(1, len(sum_se) + 1) / len(sum_se)

# Plot CDF
plt.figure(figsize=(8, 6))
plt.plot(sum_se, cdf, 'b-', label='CAPS Algorithm')
plt.xlim([0, 30])
plt.title('CDF of Sum Spectral Efficiency (CAPS Algorithm, 2D Deployment)')
plt.xlabel('Sum Spectral Efficiency (bits/s/Hz)')
plt.ylabel('CDF')
plt.grid(True)
plt.legend()
plt.savefig('caps_cdf.png')