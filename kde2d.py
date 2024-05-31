import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import pandas as pd

def kde_2d(data, grid_size=100, bandwidth=None):
    # Create a Gaussian KDE object
    kde = gaussian_kde(data.T, bw_method=bandwidth)
    
    # Create a grid of points
    x_min, y_min = data.min(axis=0)
    x_max, y_max = data.max(axis=0)
    x = np.linspace(x_min, x_max, grid_size)
    y = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x, y)
    
    # Evaluate the KDE on the grid
    positions = np.vstack([xx.ravel(), yy.ravel()])
    zz = np.reshape(kde(positions).T, xx.shape)
    
    return xx, yy, zz

np.random.seed(123)

# read data from csv
df = pd.read_csv("data_gravitational_lensing.csv")
data = df.to_numpy()

# bandwidth values
bandwidths = [0.5, 1.0, 2.0]

fig, axes = plt.subplots(3, 2, figsize=(18, 24))

for i, h in enumerate(bandwidths):
    xx, yy, zz = kde_2d(data, grid_size=100, bandwidth=h)
    
    # plot the heatmap
    ax = axes[i, 0]
    heatmap = ax.contourf(xx, yy, zz, levels=20, cmap='viridis')
    ax.scatter(data[:, 0], data[:, 1], s=8, color='red')
    ax.set_title(f'2D KDE Heatmap (h={h})')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    fig.colorbar(heatmap, ax=ax, label='Density')
    
    # plot the 3d surface plot
    ax = fig.add_subplot(3, 2, 2 * i + 2, projection='3d')
    ax.plot_surface(xx, yy, zz, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
    ax.set_title(f'3D KDE Surface Plot (h={h})')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel(r"$\hat{f}_{h}^{(ker)}(x)$")

plt.tight_layout()
plt.savefig("./kde2d_gravitaoinal_lensing_data.pdf")
plt.show()


