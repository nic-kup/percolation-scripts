import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
import matplotlib.image as matim
from matplotlib import animation
import random
from itertools import product
from functools import partial

from joblib import Parallel, delayed

X_SIZE = 512
Y_SIZE = 512
p = 0.5

SEED = 5

fig = plt.figure()
ax = plt.axes(xlim=(0, X_SIZE), ylim=(0, Y_SIZE))
ax.set_axis_off()


grid = np.zeros((X_SIZE, Y_SIZE), dtype="int32")
grid[X_SIZE // 2, Y_SIZE // 2] = 1

active_verts = [(X_SIZE // 2, Y_SIZE // 2)]


def get_neighbors(list_of_verts):
    new_list = []
    for i, j in list_of_verts:
        new_list.append(((i - 1) % X_SIZE, j))
        new_list.append(((i + 1) % X_SIZE, j))
        new_list.append((i, (j - 1) % Y_SIZE))
        new_list.append((i, (j + 1) % Y_SIZE))
    return list(set(new_list))


def grid_update(old_grid, i, j):
    if i == X_SIZE or j == Y_SIZE or old_grid[i, j] != 0:
        return None
    else:
        if i > 0 and old_grid[i - 1, j] == 1 and npr.random() < p:
            return (i, j)
        if j > 0 and old_grid[i, j - 1] == 1 and npr.random() < p:
            return (i, j)
        if i < X_SIZE - 1 and old_grid[i + 1, j] == 1 and npr.random() < p:
            return (i, j)
        if j < Y_SIZE - 1 and old_grid[i, j + 1] == 1 and npr.random() < p:
            return (i, j)
    return None

time_step = 0
while len(active_verts) > 0:
    time_step += 1
    if time_step % 10 == 0:
        print(f"Step: {time_step:4}; active {len(active_verts)}")
    new_verts = get_neighbors(active_verts)

    new_verts = Parallel(n_jobs=-1)(
        delayed(grid_update)(grid, i, j) for i, j in new_verts
    )
    for i, j in active_verts:
        grid[i, j] = -time_step - 100

    active_verts = list(filter(lambda x: x is not None, new_verts))

    for i, j in active_verts:
        grid[i, j] = 1

min_grid = np.min(grid)

print(min_grid)

im = plt.imshow(
    grid, interpolation="none", cmap="gist_ncar"
)
plt.show()
grid = im.get_array()

filename = "seed" + str(SEED) + "pic_p=" + str(p)
matim.imsave("pics/"+filename+".png", grid.T, cmap="gist_ncar")
matim.imsave("pics/"+filename+"g.png", grid.T, cmap="gray")

