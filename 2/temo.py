import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Example: your matrix
# Z = np.random.rand(50, 50)

def surface(Z):
    # Create X and Y coordinates
    x = np.arange(Z.shape[1])
    y = np.arange(Z.shape[0])
    X, Y = np.meshgrid(x, y)

    # Plot surface
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    plt.show()
