import math
import random
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from Helpers import threed_laplace, twod_laplace, nd_laplace
from scipy import spatial

from scipy.stats import gamma
from scipy.special import logsumexp


class ldp_mechanism:
    # def __init__(self, epsilon: float = 0.5):
    # self.epsilon = epsilon
    """
    Truncate dataset based on a meshgrid
    """

    def grid_remap(self, non_private_dataset: pd.DataFrame, private_dataset: pd.DataFrame, grid_size: int = 10,
                   include_indicator: bool = False, columns=['x', 'y']):
        mesh = [np.linspace(non_private_dataset[:, i].min(), non_private_dataset[:, i].max(), num=grid_size) for i in
                range(non_private_dataset.shape[1])]
        meshgrid = np.meshgrid(*mesh, indexing='ij')
        tree = spatial.KDTree(non_private_dataset)
        _, closest_indices = tree.query(private_dataset)

        distances = np.linalg.norm(private_dataset - non_private_dataset[closest_indices], axis=1)

        # Reshape the meshgrid array
        meshgrid_reshaped = np.stack(meshgrid, axis=-1)

        # Create a KDTree from meshgrid
        meshgrid_tree = spatial.KDTree(meshgrid_reshaped.reshape(-1, meshgrid_reshaped.shape[-1]))

        # Query the KDTree with private dataset  to find the closest points in meshgrid
        _, closest_meshgrid_indices = meshgrid_tree.query(private_dataset)

        # Calculate the distances between private dataset  and closest points in meshgrid
        meshgrid_distances = np.linalg.norm(
            private_dataset - meshgrid_reshaped.reshape(-1, meshgrid_reshaped.shape[-1])[closest_meshgrid_indices],
            axis=1)

        # Check if each point in private dataset is within the domain of non private dataset
        outside_domain_mask = self.get_outside_domain_mask(private_dataset, non_private_dataset)

        # Create a mask for points outside the domain and closer to meshgrid points
        outside_domain_and_closer_mask = np.logical_or(outside_domain_mask, meshgrid_distances < distances)

        # Remap points outside the domain and closer to meshgrid points to the closest meshgrid points
        remapped_dataset = private_dataset.copy()
        remapped_dataset[outside_domain_and_closer_mask] = \
            meshgrid_reshaped.reshape(-1, meshgrid_reshaped.shape[-1])[closest_meshgrid_indices][
                outside_domain_and_closer_mask]
        remapped_dataset = pd.DataFrame(remapped_dataset, columns=columns)
        if (include_indicator):
            remapped_dataset['is_remapped'] = False
            remapped_dataset.loc[outside_domain_mask, 'is_remapped'] = True
            print("Points outside domain for grid remapping:", remapped_dataset.loc[outside_domain_mask].shape[0])
        return remapped_dataset

    def get_outside_domain_mask(self, private_dataset: pd.DataFrame, non_private_dataset: pd.DataFrame):
        # Create a mask for points outside the domain of dataset2
        in_domain = np.logical_and.reduce([np.logical_and(private_dataset[:, dim] > non_private_dataset[:, dim].min(),
                                                          private_dataset[:, dim] < non_private_dataset[:, dim].max())
                                           for dim in range(non_private_dataset.shape[1])])
        outside_domain_mask = np.logical_not(in_domain)
        return outside_domain_mask

    def Q_r(self, x, points, radius, kd_tree=None):
        kdtree = spatial.KDTree(points) if kd_tree is None else kd_tree
        indices = kdtree.query_ball_point(x, radius, p=2)
        return [points[i] for i in indices]

    def calculate_distance(self, point1, point2):
        # return spatial.distance.euclidean(point1, point2)  # Euclidean distance as an example
        return np.linalg.norm(np.array(point1) - np.array(point2))

    """
    x = point in the radius around the non-private data point
    z = private data point
    real_data = non-private data
    radius = the original radius that was used to generate z (so radial distance x - z)
    epsilon = privacy budget
    plain_tree = KDTree of the non-private data
    w_x = the points in the radius around the non-private data point
    """
    def remap_point(self, x, z, real_data, radius, epsilon, plain_tree, w_x):
        w_x_len = len(w_x)
        distances_q = []
        # get all points in the radius around point that was in the radius of the original point x.
        q_r = self.Q_r(x, real_data, radius, plain_tree)
        # get w(q) which is just the length of q_r
        w_q = len(q_r)
        # for each q_r we want to know the distance from q to z.
        for q in q_r:
            distance_q_r = self.calculate_distance(q, z)
            distances_q.append(distance_q_r)
        distances_q_sum = np.sum(distances_q) # calculate the sum of distances
        calculate_q_sum = (w_q * logsumexp(-epsilon * distances_q_sum)) # we use the logsumexp trick to avoid underflow (to low values).
        # always receive positive value and the sum.
        q_r_sum = np.abs(np.sum(calculate_q_sum))
        # calculate the distance between x and z
        distance_xz = self.calculate_distance(x, z)
        # calculate the remapped value
        remapped_value = (w_x_len * math.exp(-epsilon * distance_xz)) / q_r_sum if q_r_sum > 0 else 0

        return remapped_value

    def optimal_remap(self, non_private_df: pd.DataFrame, perturbed_df: pd.DataFrame, max_iterations=100):
        tree = spatial.KDTree(non_private_df)
        perturbed_data_copy = perturbed_df.copy()
        perturbed_data_outside_domain = perturbed_data_copy.copy()
        if perturbed_df.columns.isin(['is_remapped']).any():
            print("Found is_remapped column, so we use that to filter")
            perturbed_data_outside_domain = perturbed_data_copy.loc[perturbed_data_copy['is_remapped']]
            perturbed_data_outside_domain = perturbed_data_outside_domain.drop(columns=['is_remapped'])
            perturbed_data_copy.drop(columns=['is_remapped'], inplace=True)
        if not perturbed_data_copy.columns.isin(['is_remapped']).any():
            print("No is_remapped column found, so we find them ourself")
            points_outside_domain = self.get_outside_domain_mask(perturbed_data_copy.values, non_private_df.values)
            perturbed_data_outside_domain = perturbed_data_copy.copy().iloc[points_outside_domain]
        if not perturbed_data_copy.columns.isin(['r']).any():
            raise ValueError("Perturbed data should have a column named 'r' which is the radius of the grid.")
        if not perturbed_data_copy.shape[1] - 1 is non_private_df.shape[1]:
            raise ValueError("Perturbed data should have the same number of columns as plain data.")
        if not perturbed_data_copy.columns.drop(labels=['r']).isin(non_private_df.columns).all():
            raise ValueError("Perturbed data should have the same columns as plain data.")

        print("Points outside domain....", perturbed_data_outside_domain.shape)

        for index, private_data_point in perturbed_data_outside_domain.iterrows():  # loop through each point outside the domain
            non_private_data_point = non_private_df.iloc[index]  # get the corresponding plain data point
            list_sigma = []
            # calculate w_x
            c_ball = self.Q_r(non_private_data_point.values, non_private_df.values, private_data_point['r'], tree)

            popularity_x_memory_opt = np.array(c_ball)
            if len(c_ball) > max_iterations:
                # avoid memory issues and just select a random subset of the points in the radius
                random_indices = np.random.choice(len(c_ball), size=max_iterations)
                popularity_x_memory_opt = np.array(c_ball)[random_indices]
            # for every point in a radius r around the non-private data point, calculate the new r.
            for x_q in popularity_x_memory_opt:
                list_sigma.append(self.remap_point(x_q, private_data_point[non_private_df.columns].values,
                                                   non_private_df.values, private_data_point['r'], self.epsilon, tree,
                                                   w_x=c_ball))
            # calculate the coefficient
            coefficients = [x_new * non_private_data_point for x_new in list_sigma]
            sum_coefficients = sum(coefficients)
            # get the probabilities for each value
            probabilities = np.array([coeff / sum_coefficients for coeff in
                                      coefficients])  # calculate the probabilities using the coefficients
            if sum_coefficients > 0 if isinstance(sum_coefficients, int) else sum_coefficients[
                sum_coefficients > 0].all():
                # calculate the new value for the point based on the average with weighted probabilities.
                averaged_remap = np.average(popularity_x_memory_opt, axis=0,
                                            weights=probabilities)
                # assign the new points to the perturbed dataset
                perturbed_data_copy.loc[index, non_private_df.columns] = averaged_remap if not np.isnan(
                    averaged_remap).any() else private_data_point[non_private_df.columns]

        return perturbed_data_copy

    def generate_2d_noise_for_point(self, non_private_row: np.array):
        p = random.random()
        theta = np.random.rand() * np.pi * 2
        r = twod_laplace.inverseCumulativeGamma(self.epsilon, p)  # draw radius distance
        private_point = twod_laplace.addVectorToPoint(non_private_row, r, theta)
        return private_point[0], private_point[1], r

    def generate_2d_noise_for_dataset(self, non_private_data: pd.DataFrame):
        Z = []
        R = []
        X = np.array(non_private_data)
        for row in non_private_data.values:
            private_data_point = self.generate_2d_noise_for_point(row)
            Z.append(private_data_point[:2])
            R.append(private_data_point[2])
        return pd.concat((pd.DataFrame(Z, columns=non_private_data.columns), pd.DataFrame(R, columns=['r'])), axis=1)

    """
    Generate noise for a point in 3-dimensional space
    @returns: x, y, z, r, where r is the radial distance from the center.
    """
    def generate_3d_noise_for_point(self):
        polar_angle, azimuth, _ = threed_laplace.generate_unit_sphere()  # theta, psi
        r = threed_laplace.generate_r(self.epsilon)
        # theta = 2 * np.pi * u[0]
        # theta = np.random.rand() * np.pi
        # phi = np.arccos(2 * u[1] - 1)
        # phi = np.random.rand() * np.pi*2 #
        # https://mathworld.wolfram.com/SphericalCoordinates.html formula 4/5/6
        x = r * np.sin(polar_angle) * np.sin(azimuth)
        y = r * np.sin(polar_angle) * np.cos(azimuth)
        z = r * np.cos(polar_angle)
        return x, y, z, r

    """
    Generate noise for a dataset in 3-dimensional space
    """
    def generate_3d_noise_for_dataset(self, non_private_dataset: pd.DataFrame):
        Z = []
        R = []
        X = np.array(non_private_dataset)
        for x in X:
            noise = self.generate_3d_noise_for_point()
            z = x + noise[:3]
            Z.append(z)
            R.append(noise[3])
        return pd.concat((pd.DataFrame(Z, columns=non_private_dataset.columns), pd.DataFrame(R, columns=['r'])), axis=1)

    def generate_nd_noise_for_dataset(self, non_private_dataset: pd.DataFrame):
        Z = []
        R = []
        X = np.array(non_private_dataset)
        for x in X:
            z, r = self.generate_nd_noise_for_point(x)
            Z.append(z)
        return pd.concat((pd.DataFrame(Z, columns=non_private_dataset.columns), pd.DataFrame(R, columns=['r'])), axis=1)

    def generate_nd_noise_for_point(self, x):
        n = len(x)
        sphere_noise = nd_laplace.spherepicking(n)
        r = gamma.rvs(n, scale=1 / self.epsilon)
        u = nd_laplace.ct(r, sphere_noise)
        z = x + u
        return z, r

    def mechanism_factory(self, dimensions: int, non_private_dataset: pd.DataFrame):
        if dimensions is 2:
            print('Run 2D-Laplace mechanism...')
            return self.generate_2d_noise_for_dataset(non_private_dataset)
        elif dimensions is 3:
            print('Run 3D-Laplace mechanism...')
            return self.generate_3d_noise_for_dataset(non_private_dataset)
        else:
            return self.generate_nd_noise_for_dataset(non_private_dataset)

    """
    Put everything together
    """

    def generate_nd_laplace_for_dataset(self, non_private_dataset: pd.DataFrame):
        dimensions = len(non_private_dataset.columns)
        return self.mechanism_factory(dimensions, non_private_dataset)

    """
    Epsilon was added to have the same format as the other mechanisms.
    """

    def randomise(self, non_private_dataset: pd.DataFrame, epsilon, grid_size=6, plot_validation: bool = False, max_iterations=50, apply_normalization=False):
        self.epsilon = epsilon
        scaler = MinMaxScaler()
        print('Run appropiate mechanism to generate a private dataset...')
        columns = non_private_dataset.columns
        if apply_normalization:
            private_dataframe = self.generate_nd_laplace_for_dataset(pd.DataFrame(scaler.fit_transform(non_private_dataset), columns=columns))
        else:
            private_dataframe = self.generate_nd_laplace_for_dataset(non_private_dataset)

        # perturbed_df_find_grid_remappings_with_r = pd.concat([private_dataframe['r'], perturbed_df_with_grid_remapping], axis=1)
        # print(perturbed_df_with_grid_remapping)
        print('All data that was remapped using a grid, is optimally remapped...')
        perturbed_df_optimal_remapping = self.optimal_remap(non_private_dataset.copy(), private_dataframe.copy(), max_iterations)
        # remap any points that are outside the domain of the non-private dataset after optimal remapping
        print('Shapes', perturbed_df_optimal_remapping.shape, private_dataframe.shape)
        if (plot_validation):
            print(
                'Approximate the private dataset outside the domain to be inside the domain of the non-private dataset '
                'using a grid...')
            perturbed_df_with_grid_remapping = self.grid_remap(non_private_dataset.values,
                                                               private_dataframe.drop(columns=['r']).values,
                                                               grid_size=grid_size, columns=columns,
                                                               include_indicator=True)
            self.validate_randomisation(non_private_dataset, perturbed_df_optimal_remapping,
                                        perturbed_df_with_grid_remapping, private_dataframe)

        return perturbed_df_optimal_remapping.drop(columns=['r'])

    def validate_randomisation(self, non_private_dataset: pd.DataFrame, remapped_private_dataset: pd.DataFrame,
                               private_dataset_grid_remap: pd.DataFrame, private_dataset: pd.DataFrame, columns=None):
        dimensions = len(non_private_dataset.columns)
        if dimensions is 2:
            self.validate_2d_plot(non_private_dataset, remapped_private_dataset, private_dataset_grid_remap,
                                  private_dataset)
        elif dimensions is 3:
            self.validate_3d_plot(non_private_dataset, remapped_private_dataset, private_dataset_grid_remap,
                                  private_dataset)
        else:
            self.validate_3d_plot(non_private_dataset, remapped_private_dataset, private_dataset_grid_remap,
                                  private_dataset, columns=columns)

    def validate_2d_plot(self, non_private_dataset: pd.DataFrame, remapped_private_dataset: pd.DataFrame,
                         private_dataset_grid_remap: pd.DataFrame, original_private_dataset: pd.DataFrame):
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(18, 5))
        columns = non_private_dataset.columns
        ax = fig.add_subplot(1, 4, 1)
        ax.scatter(non_private_dataset[columns[0]], non_private_dataset[columns[1]], c='blue', marker='o', alpha=0.1)
        ax.legend(['Non-private data'])
        ax = fig.add_subplot(1, 4, 2)
        ax.scatter(original_private_dataset[columns[0]], original_private_dataset[columns[1]], c='green', marker='x',
                   alpha=0.1)
        ax.legend(['2D-Laplace private data'])
        ax = fig.add_subplot(1, 4, 3)
        ax.scatter(private_dataset_grid_remap[columns[0]], private_dataset_grid_remap[columns[1]], c='red', marker='o',
                   alpha=0.1)
        ax.legend(['Grid-mapped'])
        ax = fig.add_subplot(1, 4, 4)
        ax.scatter(remapped_private_dataset[columns[0]], remapped_private_dataset[columns[1]], c='green', marker='o',
                   alpha=0.1)
        ax.legend(['Optimal remapped'])
        plt.show()

    def validate_3d_plot(self, non_private_dataset: pd.DataFrame, remapped_private_dataset: pd.DataFrame,
                         private_dataset_grid_remap: pd.DataFrame, original_private_dataset: pd.DataFrame,
                         columns=None):
        import matplotlib.pyplot as plt
        columns_to_plot = non_private_dataset.columns if columns is None else columns
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 4, 1, projection='3d')
        ax.scatter(non_private_dataset[columns_to_plot[0]], non_private_dataset[columns_to_plot[1]],
                   non_private_dataset[columns_to_plot[2]], c='blue', marker='o', alpha=0.1)
        ax.legend(['Non-private data'])
        ax = fig.add_subplot(1, 4, 2, projection='3d')
        ax.scatter(original_private_dataset[columns_to_plot[0]], original_private_dataset[columns_to_plot[1]],
                   original_private_dataset[columns_to_plot[2]], c='green', marker='x', alpha=0.1)
        ax.legend(['3D-Laplace private data'])
        ax = fig.add_subplot(1, 4, 3, projection='3d')
        ax.scatter(private_dataset_grid_remap[columns_to_plot[0]], private_dataset_grid_remap[columns_to_plot[1]],
                   private_dataset_grid_remap[columns_to_plot[2]], c='red', marker='o', alpha=0.1)
        ax.legend(['Grid-mapped'])
        ax = fig.add_subplot(1, 4, 4, projection='3d')
        ax.scatter(remapped_private_dataset[columns_to_plot[0]], remapped_private_dataset[columns_to_plot[1]],
                   remapped_private_dataset[columns_to_plot[2]], c='green', marker='o', alpha=0.1)
        ax.legend(['Optimal remapped'])
        plt.show()
