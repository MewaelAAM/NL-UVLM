import numpy as np
import matplotlib.pyplot as plt
from propeller import PropellerGeometry
from mesh import PropellerMesh

class CylindricalArms:
    def __init__(self, fuselage_position, propeller_mesh, radius=0.01, resolution=30, fuselage_radii=(0.1, 0.05, 0.03)):
        """
        Initialize the CylindricalArms class.

        Args:
            fuselage_position (array-like): The position of the fuselage COM (center of mass).
            propeller_mesh (PropellerMesh): The propeller mesh instance to extract hub positions.
            radius (float): Radius of the arms (cylinders).
            resolution (int): Resolution of the cylinder surface mesh.
            fuselage_radii (tuple): Radii of the fuselage ellipsoid in x, y, z directions.
        """
        self.fuselage_position = np.array(fuselage_position)
        self.hub_positions = propeller_mesh._generate_hub_points()  # Generate hub positions dynamically
        self.radius = radius
        self.resolution = resolution
        self.fuselage_radii = np.array(fuselage_radii)  # Ellipsoid radii (x, y, z)

    def adjust_start_point(self, end):
        direction = end - self.fuselage_position
        normalized_direction = direction / np.linalg.norm(direction)
        scale_factors = self.fuselage_radii / np.abs(normalized_direction)
        scale = np.min(scale_factors)
        return self.fuselage_position + normalized_direction * scale

    def generate_arm(self, start, end):
        vector = end - start
        length = np.linalg.norm(vector)
        vector /= length  # Normalize the vector

        theta = np.linspace(0, 2 * np.pi, self.resolution)
        circle_x = self.radius * np.cos(theta)
        circle_y = self.radius * np.sin(theta)

        t = np.linspace(0, length, 2)
        X = np.outer(circle_x, np.ones_like(t)) + np.outer(np.ones_like(circle_x), t * vector[0]) + start[0]
        Y = np.outer(circle_y, np.ones_like(t)) + np.outer(np.ones_like(circle_y), t * vector[1]) + start[1]
        Z = np.outer(np.zeros_like(circle_x), np.ones_like(t)) + np.outer(np.ones_like(circle_x), t * vector[2]) + start[2]

        return X, Y, Z

    def generate_perpendicular_cylinder(self, hub_position):
        perpendicular_vector = np.array([0, 0, 1])  # Fixed direction: Upwards

        length = self.radius * 1  # Small length for the perpendicular cylinder
        theta = np.linspace(0, 2 * np.pi, self.resolution)
        circle_x = self.radius * np.cos(theta)
        circle_y = self.radius * np.sin(theta)

        t = np.linspace(0, length, 2)
        X = np.outer(circle_x, np.ones_like(t)) + np.outer(np.ones_like(circle_x), t * perpendicular_vector[0]) + hub_position[0]
        Y = np.outer(circle_y, np.ones_like(t)) + np.outer(np.ones_like(circle_y), t * perpendicular_vector[1]) + hub_position[1]
        Z = np.outer(np.zeros_like(circle_x), np.ones_like(t)) + np.outer(np.ones_like(circle_x), t * perpendicular_vector[2]) + hub_position[2]

        return X, Y, Z

    def plot(self, ax):
        for hub_position in self.hub_positions:
            start = self.adjust_start_point(hub_position)
            X, Y, Z = self.generate_arm(start, hub_position)
            ax.plot_surface(X, Y, Z, color='gray', alpha=0.7)

            # Generate and plot the perpendicular cylinder
            X_perp, Y_perp, Z_perp = self.generate_perpendicular_cylinder(hub_position)
            ax.plot_surface(X_perp, Y_perp, Z_perp, color='red', alpha=0.7)  # Red for the perpendicular cylinder


class FuselageGeometry:
    def __init__(self, radii, position=(0, 0, 0)):
        self.radii = np.array(radii)
        self.position = np.array(position)

    def generate_ellipsoid(self, resolution=30):
        u = np.linspace(0, 2 * np.pi, resolution)
        v = np.linspace(0, np.pi, resolution)
        x = self.radii[0] * np.outer(np.cos(u), np.sin(v)) + self.position[0]
        y = self.radii[1] * np.outer(np.sin(u), np.sin(v)) + self.position[1]
        z = self.radii[2] * np.outer(np.ones_like(u), np.cos(v)) + self.position[2]
        return x, y, z



