"""
We create a simulation of a soft geometric random graph. First we define
suitable parameters and a connection function. Then we generate the particles
and define an adjacency matrix. Then we calculate desired values and plot.
"""
import numpy as np

from scipy.linalg import null_space
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from numpy import random as npr

import matplotlib.pyplot as plt
from matplotlib import colors

import time



window_size = 60
intensity = 4.0
conn_prob = 0.8

# Requires computation of Laplacian (slow)
plot_spectral_graph = False


grp_colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

NUMBER_OF_PARTICLES = npr.poisson((window_size**2) * intensity)
print("Number of particles:", NUMBER_OF_PARTICLES)
print("Connection probability:", conn_prob)


square_num = window_size
hash_mat = [[[] for j in range(square_num)] for i in range(square_num)]


def hash_func(x):
    return int(x[0]), int(x[1])


def get_hash_nbhd(hk):
    nbhd = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if 0 <= hk[0] + i < window_size and 0 <= hk[1] + j < window_size:
                nbhd = nbhd + hash_mat[(hk[0] + i)][(hk[1] + j)]
    return nbhd

def connection_function(x, y):
    norm_dist = np.linalg.norm(x - y)
    unit_vec = np.abs((x-y)/norm_dist)
    return conn_prob * (norm_dist < 1.0) * (unit_vec[0] < 0.5)


def will_connect(x, y):
    return npr.random() < connection_function(x, y)


""" Initialize all the particles """
all_particles = npr.uniform(size=(NUMBER_OF_PARTICLES, 2)) * window_size
connection_matrix = np.zeros((NUMBER_OF_PARTICLES, NUMBER_OF_PARTICLES), dtype=np.int8)

for i, x in enumerate(all_particles):
    hash_mat[int(x[0])][int(x[1])].append((i, x))

number_of_edges = 0

""" Set connections """
start_time = time.time()

for i, x in enumerate(all_particles):
    hk = hash_func(x)  # hash key
    nbhd = get_hash_nbhd(hk)
    for j, y in nbhd:
        if i < j and will_connect(all_particles[i], all_particles[j]):
            number_of_edges += 1
            connection_matrix[i, j] = 1

end_time = time.time() - start_time
print("Time for connection matrix:", end_time)

connection_matrix = connection_matrix + connection_matrix.T

""" Calculate connected components """


start_time = time.time()
csr_adjacency = csr_matrix(connection_matrix)
num_components, component_adherence = connected_components(
    csgraph=csr_adjacency, directed=False, return_labels=True
)
end_time = time.time() - start_time
print("Time to compute components(SCP): ", end_time)
component_adherence = component_adherence.tolist()


size_of_comps = np.bincount(component_adherence)
largest_component = np.argmax(size_of_comps)
largest_component_size = np.max(size_of_comps)
print("number of connected components: ", num_components)

sorted_soc = np.flip(np.sort(size_of_comps))
print("top 5 comps: ", sorted_soc[:5])

plt.plot(sorted_soc)
plt.title("Components")
plt.show()

""" Connection matrix operations """
degrees = np.sum(connection_matrix, axis=0)


print("Number of isolated verticies is: ", np.count_nonzero(degrees == 0), sep="")
print(
    "Percentage of isolated verticies is: ",
    np.count_nonzero(degrees == 0) / NUMBER_OF_PARTICLES,
    sep="",
)
print("Number of connected components is: ", num_components, sep="")
print("size of largest component:", largest_component_size)
print("Percent of largest component:", largest_component_size / NUMBER_OF_PARTICLES)


""" A histogram plot """
print("Average degree:", np.mean(degrees))
plt.hist(degrees, 50, density=True)
plt.title(
    "Degree distribution with lam={0:0.2f}, p={1:0.2f}".format(intensity, conn_prob)
)
plt.show()

""" Plotting the graph """
plt.xlim([0.0, window_size])
plt.ylim([0.0, window_size])


plt.scatter(all_particles[:, 0], all_particles[:, 1], s=1.0, c="black", alpha=0.5)

for i, x in enumerate(all_particles):
    hk = hash_func(x)  # hash key
    nbhd = get_hash_nbhd(hk)
    for j, y in nbhd:
        if i < j and connection_matrix[i, j] == 1:
            if component_adherence[i] == largest_component:
                col_choice = "black"
            else:
                col_choice = grp_colors[component_adherence[i] % len(grp_colors)]
            plt.plot(
                [x[0], y[0]],
                [x[1], y[1]],
                c=col_choice,
                alpha=0.75,
            )


plt.title("RCM with lam={0:0.2f}, p={1:0.2f}".format(intensity, conn_prob))
plt.show()


""" Plotting the spectral drawing """
# This only works for connected graphs, so I will take the
# largest connected component and then proceed

# We start by getting the second and third eigenvector (rem: first is constant)

if plot_spectral_graph:

    laplacian = np.diag(degrees) - connection_matrix

    start_time = time.time()
    lap_null = null_space(laplacian)
    end_time = time.time() - start_time

    print("Time for null space:", end_time)

    print("---")

    # Get the largest component and its laplacian
    indicies_large = [x == largest_component for x in component_adherence]
    large_laplace = laplacian[np.ix_(indicies_large, indicies_large)]
    eigvals, eigvecs = np.linalg.eigh(large_laplace)

    LARGE_SIZE = large_laplace.shape[0]

    v1 = eigvecs[:, 0]
    v2 = eigvecs[:, 1]
    v3 = eigvecs[:, 2]
    v4 = eigvecs[:, 3]

    plt.scatter(v2, v3, s=1.0, c="black")
    for i in range(LARGE_SIZE):
        for j in range(i + 1, LARGE_SIZE):
            if large_laplace[i, j] != 0:
                plt.plot(
                    [v2[i], v2[j]],
                    [v3[i], v3[j]],
                    c="blue",
                    alpha=0.55,
                )

        plt.title(
            "RCM spectral plot lam={0:0.2f}, p={1:0.2f}".format(intensity, conn_prob)
        )
    plt.show()
