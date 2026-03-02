import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-6, 6, 30) 
y = np.linspace(-5, 5, 20)

X, Y = np.meshgrid(x, y)

z_function = lambda X, Y: np.sin(np.sqrt(X**4 + Y**3))
Z = z_function(X, Y)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')

surface = ax.plot_surface(X, Y, Z, cmap='plasma', edgecolor='k',  linewidth=0.3, alpha=0.9)

ax.set_xlabel('X-axis', fontsize=12)
ax.set_ylabel('Y-axis', fontsize=12)
ax.set_zlabel('Z-axis', fontsize=12)
ax.set_title('3D Surface Plot of z = sin(sqrt(x^4 + y^3))', fontsize=14)

ax.set_xlim(-6, 6)
ax.set_ylim(-5, 5)
ax.set_zlim(-1, 1)

fig.colorbar(surface, shrink=0.5, aspect=10, label='Z value')

ax.view_init(elev=30, azim=235)

plt.show()
