import math
import random
import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist
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
    def prepare_grid_remapping(self, non_private_dataset: pd.DataFrame, columns, grid_unit_size = 10):
        mesh = [np.linspace(non_private_dataset.iloc[:, i].min(), non_private_dataset.iloc[:, i].max(), num=grid_unit_size) for i in
                range(non_private_dataset.shape[1])]
        meshgrid = np.meshgrid(*mesh)
        # Convert the meshgrid to a DataFrame
        mesh_grid = pd.DataFrame({f'x{i}': meshgrid[i].flatten() for i in range(len(meshgrid))})

        kd_tree = spatial.KDTree(non_private_dataset)
        df_data = pd.DataFrame(non_private_dataset, columns=columns).copy()
        results = []
        # Calculate the upper bound and lower bound of the meshgrid
        upper_bound = mesh_grid.max()
        lower_bound = mesh_grid.min()

        for center_index, center in df_data.iterrows():
            # Points within the grid unit around center using euclidean distance
            points_within_grid_unit = kd_tree.query(center, grid_unit_size)
            # Filter df_data to only contain points within the grid unit
            points_within_grid_unit2 = df_data.iloc[points_within_grid_unit[1]]
            # Calculate the diameter of the points within the grid unit
            diameter = pdist(points_within_grid_unit2, 'euclidean').max() if len(
                points_within_grid_unit2) > 0 else grid_unit_size
            diameters = []
            # Check each dimension separately for being outside the boundary
            for dim in df_data.columns:
                dim_index = df_data.columns.get_loc(dim)
                dim_value = center[dim_index]

                # Check if the center minus the grid_unit_size is outside the boundary for this dimension
                if center[dim_index] - grid_unit_size < lower_bound[dim_index]:
                    # Update the diameter for this dimension to the distance between the center and the lower bound
                    dist_to_lower_bound = np.linalg.norm(center[dim_index] - lower_bound[dim_index])
                    diameters.append(dist_to_lower_bound)

                # Check if the center plus the grid_unit_size is outside the boundary for this dimension
                if center[dim_index] + grid_unit_size > upper_bound[dim_index]:
                    # Update the diameter for this dimension to the distance between the center and the upper bound
                    dist_to_upper_bound = np.linalg.norm(center[dim_index] - upper_bound[dim_index])
                    diameters.append(dist_to_upper_bound)

            # Calculate the radius r_prime
            r_prime = min(diameters) if len(diameters) > 0 else diameter

            results.append(r_prime)

        return results

    def grid_remap(self, non_private_dataset: pd.DataFrame, private_dataset: pd.DataFrame, columns, grid_unit_size = 10):
        non_private_df_copy = non_private_dataset.copy()
        private_df = private_dataset.copy()
        private_df['r_prime'] = self.prepare_grid_remapping(non_private_df_copy, columns, grid_unit_size)

        if not private_df.columns.isin(['r']).any():
            ValueError("Did not find r column, so we cannot grid remap")
        for index, row in non_private_df_copy.iterrows():
            # Get r inside private_df on same index
            r = private_df.at[index, 'r']
            # Get r_prime inside private_df on same index
            r_prime = private_df.at[index, 'r_prime']
            # If the distance is greater than r_prime, adjust the private point to lie on the boundary
            if r > r_prime:
                # convert x1, y1 to polar coordinates
                #angle = np.arctan2(y1, x1)
                spherical_row = self.cartesian_to_spherical(row[columns])
                cartesian_noise = nd_laplace.ct(r_prime, spherical_row)
                # convert r_prime to polar rho
                # rho = np.sqrt(x1**2 + y1**2)
                # set value to each column
                for i, column in enumerate(columns):
                    z_prime = row[column] + cartesian_noise[i]
                    private_df.at[index, column] = z_prime

                private_df.at[index, 'is_remapped'] = True
            else:
                private_df.at[index, 'is_remapped'] = False

        return private_df

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
        if not perturbed_data_copy.columns.isin(['is_remapped']).any():
            print("No is_remapped column found, so we find them ourself")
            points_outside_domain = self.get_outside_domain_mask(perturbed_data_copy.values, non_private_df.values)
            perturbed_data_outside_domain = perturbed_data_copy.copy().loc[points_outside_domain]
        if perturbed_df.columns.isin(['is_remapped']).any():
            print("Found is_remapped column, so we use that to filter")
            perturbed_data_outside_domain = perturbed_data_copy.loc[perturbed_data_copy['is_remapped']]
            perturbed_data_outside_domain = perturbed_data_outside_domain.drop(columns=['is_remapped'])
            perturbed_data_copy.drop(columns=['is_remapped'], inplace=True)
        if not perturbed_data_copy.columns.isin(['r']).any():
            raise ValueError("Perturbed data should have a column named 'r' which is the radius of the grid.")
        #if not perturbed_data_copy.columns.drop(labels=['r', 'is_remapped']).isin(non_private_df.columns).all():
        #    raise ValueError("Perturbed data should have the same columns as plain data.")

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

    def cartesian_to_spherical(self, cartesian_coords):
        # Initialize an array to store the polar angles (θ1, θ2, ...)
        polar_angles = []
        r = np.sqrt(np.sum(np.square(cartesian_coords)))

        # Calculate the polar angles θ1, θ2, ...
        for i in range(len(cartesian_coords) - 1):
            angle_i = np.arccos(cartesian_coords[i] / (r * np.prod(np.sin(polar_angles))))
            polar_angles.append(angle_i)

        # Combine r and polar angles into a spherical coordinate tuple
        spherical_coords = (r,) + tuple(polar_angles)

        return spherical_coords

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

    def randomise(self, non_private_dataset: pd.DataFrame, epsilon, grid_size=12, plot_validation: bool = False, max_iterations=50, apply_normalization=False):
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
        print('Approximate the private dataset outside the domain to be inside the domain of the non-private dataset ', 'using a grid...')
        perturbed_df_with_grid_remapping = self.grid_remap(non_private_dataset,
                                                               private_dataframe,
                                                               grid_unit_size=grid_size, columns=columns)

        print('All data that was remapped using a grid, is optimally remapped...')
        perturbed_df_optimal_remapping = self.optimal_remap(non_private_dataset.copy(), perturbed_df_with_grid_remapping.copy(), max_iterations)
        # remap any points that are outside the domain of the non-private dataset after optimal remapping
        print('Shapes', perturbed_df_optimal_remapping.shape, private_dataframe.shape)
        if (plot_validation):

            self.validate_randomisation(non_private_dataset, perturbed_df_optimal_remapping,
                                        perturbed_df_with_grid_remapping, private_dataframe)

        columns_to_remove = ['r', 'r_prime', 'is_remapped'] if perturbed_df_optimal_remapping.columns.isin(['is_remapped']).any() else ['r', 'r_prime']
        return perturbed_df_optimal_remapping.drop(columns=columns_to_remove)

    def randomise_with_grid(self, non_private_dataset: pd.DataFrame, epsilon, grid_size=12):
        self.epsilon = epsilon
        columns = non_private_dataset.columns
        private_dataframe = self.generate_nd_laplace_for_dataset(non_private_dataset)
        perturbed_df_with_grid_remapping = self.grid_remap(non_private_dataset,
                                                           private_dataframe,
                                                           grid_unit_size=grid_size, columns=columns)
        return perturbed_df_with_grid_remapping.drop(columns=['r', 'r_prime', 'is_remapped'])

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
