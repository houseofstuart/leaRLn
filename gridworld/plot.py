import matplotlib.pyplot as plt
import numpy as np


def plot_grid(q_values, grid_size):
    # Get the maximum Q-value for each state
    q_values_grid = np.max(q_values, axis=1).reshape(grid_size, grid_size)
    arrow_grid = np.argmax(q_values, axis=1).reshape(grid_size, grid_size)

    plt.figure(figsize=(6, 6))
    plt.imshow(q_values_grid, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Q-value')
    plt.title('Learned Q-values for Each State')

    plt.xticks(np.arange(grid_size), np.arange(grid_size))
    plt.yticks(np.arange(grid_size), np.arange(grid_size))
    plt.gca().invert_yaxis()
    plt.grid(True)

    # Right, Up, Left Down (0, 1, 2, 3)
    arrows = {
        "0": '→',
        "1": '↑',
        "2": '←',
        "3": '↓'
    }

    for i in range(grid_size ** 2):
        row, col = divmod(i, grid_size)
        q_value = q_values_grid[row, col]
        action = arrow_grid[row, col]

        plt.text(
            col, row, f'{q_value:.2f}:{arrows[str(action)]}',
            # col, row, f'{q_value:.2f}',
            ha='center', va='center', color='black'
        )

    plt.show()
