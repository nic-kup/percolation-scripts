import numpy as np
import matplotlib.pyplot as plt

# Use tuples for directions to avoid the overhead of NumPy arrays for small operations
directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
all_colors = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:brown",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

STEPS = 20_000


def generate_lerw_length_optimized(N, verbose=False):
    total_steps = 0
    walk_history = [(0, 0)]
    loop_soup = []
    visited = set(walk_history)

    while len(walk_history) < N:
        total_steps += 1
        current = walk_history[-1]
        direction = directions[
            np.random.randint(4)
        ]  # Directly use integers for random choice
        step = (current[0] + direction[0], current[1] + direction[1])

        if verbose and (total_steps % 5000 == 0):
            print(f"\r{len(walk_history)}  ", flush=True, end="")

        if step not in visited:
            walk_history.append(step)
            visited.add(step)
        else:
            col_ind = walk_history.index(step)
            loop_soup.append(walk_history[col_ind:])
            walk_history = walk_history[: col_ind + 1]
            visited = set(walk_history)  # Update visited to reflect the new history

    if verbose:
        print(f"Total Steps = {total_steps}")
    return walk_history, loop_soup


def plot_walks_and_loops(walk, loops):
    plt.figure(figsize=(8, 8))
    # Plot walk
    x, y = zip(*walk)
    plt.plot(x, y, color="tab:blue")
    plt.axis("equal")
    plt.show()

    # Plot loops with different colors
    for i, loop in enumerate(loops):
        x, y = zip(*(loop + [loop[0]]))
        plt.plot(x, y, color=all_colors[i % len(all_colors)])
    plt.axis("equal")
    plt.show()

    for i, loop in enumerate(loops):
        x, y = zip(*(loop + [loop[0]]))
        plt.plot(x, y, color=all_colors[i % len(all_colors)])
        plt.title(f"Length = {len(loop)}")
        plt.axis("equal")
        plt.show()


# Example usage
walks = generate_lerw_length_optimized(STEPS, True)
walks[1].sort(key=len)
plot_walks_and_loops(walks[0], walks[1][-10:])
