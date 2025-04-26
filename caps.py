import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
import math


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
beta_0db = 8.53    # Reference path loss

np.random.seed(42)

# Generate AP and user positions (only x and y now, no z)
x_pos_aps = np.random.uniform(0, x_length, n_aps)
y_pos_aps = np.random.uniform(0, y_length, n_aps)

x_pos_users = np.random.uniform(0, x_length, n_users)
y_pos_users = np.random.uniform(0, y_length, n_users)

X_aps = np.column_stack((x_pos_aps, y_pos_aps))
X_users = np.column_stack((x_pos_users, y_pos_users))


def CAPS(n_users):

    n_clusters = 5
    def clusters():
        kmeans_aps = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=42)
        cluster_labels_aps = kmeans_aps.fit_predict(X_aps)
        # print("cluster_labels_aps: ",cluster_labels_aps)
        clusterAP = [[] for _ in range(n_clusters)]

        for i in range(n_aps):
            clusterAP[cluster_labels_aps[i]].append(i)

        return kmeans_aps.cluster_centers_ , clusterAP

    centroids_aps,clusterAP = clusters()

    def dist_calc():
        distances = np.zeros((n_users, n_clusters))

    # Calculate distances between each user and each cluster centroid
        for i in range(n_users):
            for j in range(n_clusters):
                # Calculate Euclidean distance in 3D space
                distances[i, j] = np.sqrt(np.sum((X_users[i] - centroids_aps[j])**2))

        # sorted_distances = np.zeros_like(distances)
        # sorted_indices = np.zeros_like(distances, dtype=int)
        clusterX=np.zeros(n_users,dtype=int)
        for i in range(n_users):
            clusterX[i] = np.argsort(distances[i])[0]
        return clusterX
    
    clusterX = dist_calc()

    def assign_nearest_ap():
        M = [[] for _ in range(n_users)]
        for i in range(n_users):
            M[i]=clusterAP[clusterX[i]]
        return M

    M = assign_nearest_ap()

    user_ap_se = {}  # Dictionary to store SE values for each user's APs
    user_assigned_aps = {i: [] for i in range(n_users)}  # Track assigned APs for each user

    iteration = 1
    # Iterate until all users have at least one AP assigned
    while any(len(user_assigned_aps[i]) == 0 for i in range(n_users)):
        print(f"\nIteration {iteration} - Least SE Values:")
        print("===============================")
        
        for i in range(n_users):
            if len(user_assigned_aps[i]) > 0:  # Skip users who already have an AP
                continue
                
            user_ap_se[i] = {}
            cluster_idx = clusterX[i]
            
            # For each AP in the user's assigned cluster (Mi)
            available_aps = [ap for ap in M[i] if not any(ap in assigned_aps for assigned_aps in user_assigned_aps.values())]
            
            if not available_aps:  # If no APs available in preferred cluster, look in other clusters
                for other_cluster in range(n_clusters):
                    if other_cluster != cluster_idx:
                        available_aps.extend([ap for ap in clusterAP[other_cluster] 
                                        if not any(ap in assigned_aps for assigned_aps in user_assigned_aps.values())])
            
            # for ap_idx in available_aps:
            #     # Calculate distance from user to this specific AP
            #     ap_pos = X_aps[ap_idx]
            #     user_pos = X_users[i]
            #     d = max(np.sqrt(np.sum((ap_pos - user_pos)**2)), 1)
                
            #     # Calculate path loss and channel quality
            #     # path_loss_dB = 30.6 + 36.7 * np.log10(d)
            #     # beta = 10**(-path_loss_dB/10)
            #     # eta = 1 / (1 + 0.1 * d)
                
            #     # Calculate SE
            #     se = calc_spectral_efficiency(
            #         tau_c=tau_c,
            #         tau_p=tau_p,
            #         M_k=20,
            #         p_d=p_d,
            #         eta_kl=eta,
            #         eta_matrix=eta_matrix,
            #         beta_matrix=beta_matrix,
            #         L=n_clusters,
            #         U_j=clusterAP
            #     )
            #     user_ap_se[i][ap_idx] = se

            compute_spectral_efficiency(assignments, beta, eta, M, p_d, tau_c, tau_p)
            
            if user_ap_se[i]:  # If we found any available APs
                # Sort and assign the best available AP to this user
                best_ap = max(user_ap_se[i].items(), key=lambda x: x[1])[0]
                user_assigned_aps[i].append(best_ap)
        
        # Print least SE values for all users after this iteration
        for user_idx in range(n_users):
            if user_idx in user_ap_se and user_ap_se[user_idx]:
                min_se = min(user_ap_se[user_idx].values())
                print(f"User {user_idx}: {min_se:.4f} bits/s/Hz")
            else:
                print(f"User {user_idx}: No SE values calculated yet")
        
        iteration += 1

    # Print final assignment results
    print("\nFinal AP Assignments and Spectral Efficiency Results:")
    print("====================================================")
    for user_idx in range(n_users):
        print(f"\nUser {user_idx}:")
        print(f"Assigned AP(s): {user_assigned_aps[user_idx]}")
        if user_idx in user_ap_se:
            assigned_se = [user_ap_se[user_idx][ap] for ap in user_assigned_aps[user_idx]]
            print(f"Spectral Efficiency: {[f'{se:.4f}' for se in assigned_se]} bits/s/Hz")
        # print(f"Average SE: {np.mean(assigned_se):.4f} bits/s/Hz")
    print("--------------------------------------------")

# Calculate and plot final average spectral efficiencies



final_user_averages = []
for user_idx in range(n_users):
    if user_idx in user_ap_se:
        assigned_se = [user_ap_se[user_idx][ap] for ap in user_assigned_aps[user_idx]]
        final_user_averages.append(np.mean(assigned_se))
    else:
        final_user_averages.append(0)

plt.figure(figsize=(10, 6))
plt.bar(range(len(final_user_averages)), final_user_averages)
plt.xlabel('User Index')
plt.ylabel('Average Spectral Efficiency (bits/s/Hz)')
plt.title('Final Average Spectral Efficiency per User')
plt.grid(True)
plt.show()