import pickle
import numpy as np
import matplotlib.pyplot as plt

# load  images
f = open('T1.p', 'rb')
T1 = np.load(f)
f2 = open('T2.p', 'rb')
T2 = np.load(f2)

# display images
plt.figure(figsize = [10,4])
plt.set_cmap('gray')
plt.subplot(121)
plt.imshow(T1)
plt.title('T1w - bright mWM', fontsize = 14)
plt.subplot(122)
plt.title('T2w - dark mWM', fontsize = 14)
plt.imshow(T2)
------
# Extract non-zero pixels
non_zero_indices = (T1 > 0) & (T2 > 0)
T1_non_zero = T1[non_zero_indices]
T2_non_zero = T2[non_zero_indices]

# Plot the normalized 2D histogram
plt.figure(figsize=(8, 6))
hist = plt.hist2d(
    T1_non_zero, T2_non_zero, bins=100, cmap='jet', density=True
)
plt.colorbar(label='Normalized Intensity')
plt.title('Normalized Joint Intensity Distribution', fontsize=14)
plt.xlabel('T1 Intensity', fontsize=12)
plt.ylabel('T2 Intensity', fontsize=12)
plt.show()
------
from sklearn.mixture import GaussianMixture

# Create 2D feature space (T1 and T2 intensities)
features = np.stack((T1_non_zero, T2_non_zero), axis=1)
print("Number of samples:", features.shape[0])
print("Number of features:", features.shape[1])

# Perform GMM clustering with 3 components
gmm = GaussianMixture(n_components=3, random_state=42)
gmm.fit(features)

# Predict cluster labels for the data
labels = gmm.predict(features)

# Map the labels back to the original image shape
segmentation = np.zeros_like(T1, dtype=int)
segmentation[non_zero_indices] = labels + 1  # Label classes starting from 1

# Display the segmentation
plt.figure(figsize=(8, 6))
plt.imshow(segmentation, cmap='viridis')
plt.title('GMM Segmentation (3 Classes)', fontsize=14)
plt.colorbar(label='Class Label')
plt.show()
------
# Predict posterior probabilities
posterior_probs = gmm.predict_proba(features)

# Map posterior probabilities back to the original image shape
prob_maps = [np.zeros_like(T1, dtype=float) for _ in range(3)]
for i in range(3):
    prob_maps[i][non_zero_indices] = posterior_probs[:, i]

# Determine which class corresponds to mWM
# mWM is bright in T1 and dark in T2
mean_intensities = gmm.means_
mWM_class = np.argmax(mean_intensities[:, 0] - mean_intensities[:, 1])

tissue_names = ["Grey Matter", "White Matter", "Cerebrospinal Fluid"]
tissue_names[mWM_class] = "Myelinated White Matter (mWM)"

# Plot probability maps
plt.figure(figsize=(15, 5))
for i in range(3):
    plt.subplot(1, 3, i + 1)
    plt.imshow(prob_maps[i], cmap='jet', aspect='auto')
    plt.title(f"Probability Map: {tissue_names[i]}", fontsize=14)
    plt.colorbar(label='Probability')
plt.tight_layout()
plt.show()
-------
# Calculate the average T1w and T2w intensities of myelinated WM

mWM_mask = np.zeros_like(T1, dtype=bool)
mWM_mask[non_zero_indices] = gmm.predict(features) == mWM_class

# Calculate average intensities for mWM
avg_T1_mWM = T1[mWM_mask].mean()
avg_T2_mWM = T2[mWM_mask].mean()

# Exclude CSF
csf_mask = np.zeros_like(T1, dtype=bool)
csf_mask[non_zero_indices] = np.sum(posterior_probs, axis=1) < 0.5

# Total brain tissue excluding CSF
brain_tissue_mask = non_zero_indices & ~csf_mask

# Percentage of mWM in brain tissue
percentage_mWM = (np.sum(mWM_mask) / np.sum(brain_tissue_mask)) * 100

# Print results
print("Average T1w intensity of myelinated WM: {:.2f}".format(avg_T1_mWM))
print("Average T2w intensity of myelinated WM: {:.2f}".format(avg_T2_mWM))
print("Percentage of brain tissue (excluding CSF) that is myelinated WM: {:.2f}%".format(percentage_mWM))
-------
# Intensity range for T1 and T2 images
T1_range = np.linspace(T1.min(), T1.max(), 200)
T2_range = np.linspace(T2.min(), T2.max(), 200)

# Grid of intensity values
T1_grid, T2_grid = np.meshgrid(T1_range, T2_range)
grid_features = np.column_stack((T1_grid.ravel(), T2_grid.ravel()))

likelihoods = np.exp(gmm.score_samples(grid_features)).reshape(T1_grid.shape)  # Likelihood function
labels = gmm.predict(grid_features).reshape(T1_grid.shape)  # Cluster labels

# Plot results
plt.figure(figsize=(18, 6))

#Joint distribution (2D histogram)
plt.subplot(1, 3, 1)
plt.hist2d(T1_non_zero, T2_non_zero, bins=100, cmap='jet', density=True)
plt.colorbar(label="Density")
plt.title("Joint Distribution", fontsize=14)
plt.xlabel("T1 Intensity")
plt.ylabel("T2 Intensity")

#Likelihood function
plt.subplot(1, 3, 2)
plt.imshow(likelihoods, extent=[T1.min(), T1.max(), T2.min(), T2.max()],
           origin='lower', cmap='jet', aspect='auto')
plt.colorbar(label="Likelihood")
plt.title("Likelihood Function $p(y|\\phi)$", fontsize=14)
plt.xlabel("T1 Intensity")
plt.ylabel("T2 Intensity")

#Decision boundaries
plt.subplot(1, 3, 3)
plt.imshow(labels, extent=[T1.min(), T1.max(), T2.min(), T2.max()],
           origin='lower', cmap='jet', aspect='auto')
plt.colorbar(label="Cluster Labels")
plt.title("Decision Boundaries", fontsize=14)
plt.xlabel("T1 Intensity")
plt.ylabel("T2 Intensity")

plt.tight_layout()
plt.show()
