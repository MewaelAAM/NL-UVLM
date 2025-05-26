import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Fuselage:
    def __init__(self, length, width, height, num_points=30):
        """
        Initialize the fuselage with dimensions.

        Args:
            length (float): Length of the fuselage (x-direction)
            width (float): Width of the fuselage (y-direction)
            height (float): Height of the fuselage (z-direction)
            num_points (int): Number of points to use for the mesh
        """
        self.length = length
        self.width = width
        self.height = height
        self.num_points = num_points
        self.mesh = self._generate_mesh()

    def _generate_mesh(self):
        """Generate the ellipsoid mesh for the fuselage."""
        # Create parametric surface points
        u = np.linspace(0, 2 * np.pi, self.num_points)
        v = np.linspace(0, np.pi, self.num_points)
        u, v = np.meshgrid(u, v)

        # Generate ellipsoid surface
        x = (self.length/2) * np.cos(u) * np.sin(v)
        y = (self.width/2) * np.sin(u) * np.sin(v)
        z = (self.height/2) * np.cos(v)

        return {'X': x, 'Y': y, 'Z': z}

    def get_mesh(self):
        """Return the fuselage mesh."""
        return self.mesh