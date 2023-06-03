
import numpy as np
from scipy.stats import gamma
from Helpers import helpers 

def generate_unit_sphere(): 
    vector = np.random.randn(3)
    vector /= np.linalg.norm(vector)

    #polar_angle = np.arccos(vector[2])
    #azimuth = np.arctan2(vector[1], vector[0])
    theta = 2 * np.random.uniform(0, np.pi)
    psi = np.arccos(2*np.random.uniform() - 1)
    return theta, psi, vector

def generate_3D_noise_for_dataset(X, epsilon):
    Z = []
    X = np.array(X)
    for x in X: 
        noise = generate_3D_noise(epsilon)
        z = x + noise
        Z.append(z)
    return Z
def generate_3D_noise(epsilon): 
    polar_angle, azimuth, _ = generate_unit_sphere() # theta, psi
    r = gamma.rvs(3, scale=1/epsilon)
    # theta = 2 * np.pi * u[0]
    #theta = np.random.rand() * np.pi
    #phi = np.arccos(2 * u[1] - 1)
    #phi = np.random.rand() * np.pi*2 # 
    # https://mathworld.wolfram.com/SphericalCoordinates.html formula 4/5/6
    x = r * np.sin(polar_angle) * np.sin(azimuth)
    y = r * np.sin(polar_angle) * np.cos(azimuth)
    z = r * np.cos(polar_angle)
    return x, y, z

def remap_to_closted(perturbed_dataset, original_dataset, grid): 
    X, Y, Z = grid

        # Define the domain of the original dataset
    X_min, X_max = original_dataset[:, 0].min(), original_dataset[:, 0].max()
    Y_min, Y_max = original_dataset[:, 1].min(), original_dataset[:, 1].max()
    Z_min, Z_max = original_dataset[:, 2].min(), original_dataset[:, 2].max()
    domain_X = ((X_min, X_max), (Y_min, Y_max), (Z_min, Z_max))

    # Find the indices of the closest points in the original dataset for each point in the perturbed dataset
    indices_X = np.argmin(np.sum((perturbed_dataset[:, np.newaxis] - original_dataset[np.newaxis, :]) ** 2, axis=-1), axis=-1)

    # Check which points in the perturbed dataset are outside the domain of the original dataset
    outside_domain_X = np.logical_or(perturbed_dataset[:, 0] < domain_X[0][0], perturbed_dataset[:, 0] > domain_X[0][1])
    outside_domain_X = np.logical_or(outside_domain_X, perturbed_dataset[:, 1] < domain_X[1][0])
    outside_domain_X = np.logical_or(outside_domain_X, perturbed_dataset[:, 1] > domain_X[1][1])
    outside_domain_X = np.logical_or(outside_domain_X, perturbed_dataset[:, 2] < domain_X[2][0])
    outside_domain_X = np.logical_or(outside_domain_X, perturbed_dataset[:, 2] > domain_X[2][1])

    # Find the indices of the closest points in the meshgrid for each point in the perturbed dataset
    indices_M = np.argmin(np.sum((perturbed_dataset[:, np.newaxis] - np.array([X.ravel(), Y.ravel(), Z.ravel()]).T[np.newaxis, :]) ** 2, axis=-1), axis=-1)
    indices_M = np.unravel_index(indices_M, X.shape)

    # Check which points in the perturbed dataset are outside the domain of the meshgrid
    outside_domain_M = np.logical_or(perturbed_dataset[:, 0] < X_min, perturbed_dataset[:, 0] > X_max)
    outside_domain_M = np.logical_or(outside_domain_M, perturbed_dataset[:, 1] < Y_min)
    outside_domain_M = np.logical_or(outside_domain_M, perturbed_dataset[:, 1] > Y_max)
    outside_domain_M = np.logical_or(outside_domain_M, perturbed_dataset[:, 2] < Z_min)
    outside_domain_M = np.logical_or(outside_domain_M, perturbed_dataset[:, 2] > Z_max)

    # Remap the points outside the domain of the original dataset to the closest points in the original dataset
    remapped_dataset = perturbed_dataset.copy()
    remapped_dataset[outside_domain_X, :] = original_dataset[indices_X[outside_domain_X], :]

    # Remap the points outside the domain of the meshgrid to the closest points in the meshgrid
    remapped_dataset[outside_domain_M, 0] = X[indices_M][outside_domain_M]
    remapped_dataset[outside_domain_M, 1] = Y[indices_M][outside_domain_M]
    remapped_dataset[outside_domain_M, 2] = Z[indices_M][outside_domain_M]
    return remapped_dataset

def generate_truncated_perturbed_dataset(X, epsilon):
    X = np.array(X)
    # meshgrid = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), num=6), np.linspace(X[:, 1].min(), X[:, 1].max(), num=6), np.linspace(X[:, 2].min(), X[:, 2].max(), num=6), indexing='ij')

    Z = generate_3D_noise_for_dataset(X, epsilon)
    Z = np.array(Z)

    return helpers.truncate_n_dimensional_laplace_noise(Z, X, 12)