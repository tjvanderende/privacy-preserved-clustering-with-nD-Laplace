# Define methods
import math
import random
import numpy as np


#Define Lambda distribution
def LambertW(x):
	#Min diff decides when the while loop ends
	min_diff = 1e-10
	if (x == -1 / np.e):
		return -1
	elif ((x < 0) and (x > -1/np.e)):
		q = np.log(-x)
		p = 1
		while (abs(p-q) > min_diff):
			p = (q * q + x / np.exp(q)) / (q + 1)
			q = (p * p + x / np.exp(p)) / (p + 1)
		#determine the precision of the float number to be returned
		return (np.round(1000000 * q) / 1000000)
	else:
		return 0

def inverseCumulativeGamma (eps, p): 
  x = (p - 1) / np.e
  return -(LambertW(x) + 1)/eps

def addVectorToPoint(point, distance, angle):
	x1, y1 = point
	x2 = x1 + (distance * np.cos(angle))
	y2 = y1 + (distance * np.sin(angle))
	return x2, y2
    
def generate_laplace_noise(eps, x, y): 
    theta = np.random.rand()*np.pi*2
    p = random.random()
    r = inverseCumulativeGamma(eps, p) # draw radius distance
    return addVectorToPoint([x, y], r, theta)


def calculate_radius_with_noise(x0, n, epsilon): 
    """
        x0: Point to perturb
        n: amount of points to generate
        epsilon: privacy budget
    """
    Z = []
    total_dis = 0
    for nm in range(0, n):
        x1, y1 = x0
        noise = generate_laplace_noise(epsilon, x1, y1)
        x2, y2 = noise
        total_dis = total_dis + math.dist(x0, noise)
        Z.append(noise)

    R = total_dis / n
    return np.array(Z), R
    

# ---------------------------------------------
# ----------------- TRUNCATION ----------------
def truncate(x_max, x_min, x0, z, epsilon): 
    """
    x_max: max domain point (x, y)
    x_min: min domain point (x, y)
    x0: point to truncate (radius centre)
    z: x0 + noise
    epsilon: privacy budget
    """
    x2, y2 = x_max
    x1, y1 = x_min

    zx, zy = z
    if(x1 < zx < x2 and y1 < zy < y2): 
        # print('inside', x, y)
        return z
    else:
        x, y = x0
        z2 = generate_laplace_noise(epsilon, x, y)
        return truncate(x_max, x_min, x0, z2, epsilon)

def truncate_array(x0, X, Z, epsilon): 
    truncatedZ1 = []
    x_max = [np.max(X[:, 0]), np.max(X[:, 1])]
    x_min = [np.min(X[:, 0]), np.min(X[:, 1])]
    for z in Z:
        truncatedZ1.append(truncate(x_max, x_min, x0, z, epsilon))

    return np.array(truncatedZ1)

def generate_laplace_noise_for_point(x0, epsilon, X, doTruncate = True): 
    x_max = [np.max(X[:, 0]), np.max(X[:, 1])]
    x_min = [np.min(X[:, 0]), np.min(X[:, 1])]
    z, R = calculate_radius_with_noise(x0, 1, epsilon)
    if(doTruncate):
        z = truncate(x_max, x_min, x0, z[0], epsilon)
    return z

def generate_truncated_laplace_noise(X, epsilon): 
    Z = []
    X = np.array(X)
    x_max = [np.max(X[:, 0]), np.max(X[:, 1])]
    x_min = [np.min(X[:, 0]), np.min(X[:, 1])]
    for x0 in X:
        z, R = calculate_radius_with_noise(x0, 1, epsilon)
        z = truncate(x_max, x_min, x0, z[0], epsilon)
        Z.append([z[0], z[1]])
    return Z

def generate_laplace_noise_for_dataset(X, epsilon):
    Z = []
    X = np.array(X)
    for x0 in X:
        z, R = calculate_radius_with_noise(x0, 1, epsilon)
        Z.append(z[0])
    return Z