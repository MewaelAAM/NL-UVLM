import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PropellerGeometry:
    def __init__(self, airfoil_distribution_file, chorddist_file, pitchdist_file, sweepdist_file=None, heightdist_file=None, R_tip=None, R_hub=None, num_blades=1):
        """
        Initialize the PropellerGeometry with required data files and blade dimensions.

        Args:
            airfoil_distribution_file (str): Path to airfoil distribution CSV.
            chorddist_file (str): Path to chord distribution CSV.
            pitchdist_file (str): Path to pitch distribution CSV.
            sweepdist_file (str): Path to sweep distribution CSV.
            heightdist_file (str): Path to height distribution CSV.
            R_tip (float): Radius of the blade tip (maximum radius).
            R_hub (float): Radius of the blade hub (minimum radius). Default is 0.
        """
        # Add number of blades parameter
        self.num_blades = num_blades

        # Assign tip and hub radii
        self.R_tip = R_tip
        self.R_hub = R_hub

        # Load airfoil contour and distribution data
        self.airfoil_distribution = pd.read_csv(airfoil_distribution_file)
        self.airfoil_contours = self._load_airfoil_contours()

        # Load chord, pitch, sweep, and height distribution data
        self.chorddist = pd.read_csv(chorddist_file)
        self.pitchdist = pd.read_csv(pitchdist_file)
        self.sweepdist = pd.read_csv(sweepdist_file) if sweepdist_file else None
        self.heightdist = pd.read_csv(heightdist_file) if heightdist_file else None

        # Create splines for interpolating blade geometry
        self._create_interpolation_splines()

    def apply_rotation(self, X, Y, Z, rotation_matrix):
        """
        Apply a rotation matrix to 3D coordinates.

        Args:
            X, Y, Z (numpy.ndarray): Original coordinates (arrays of the same shape).
            rotation_matrix (numpy.ndarray): 3x3 rotation matrix.

        Returns:
            tuple: Rotated coordinates (X_rotated, Y_rotated, Z_rotated).
        """
        original_coords = np.stack((X, Y, Z), axis=-1)  # Combine into (N, 3)
        rotated_coords = np.dot(original_coords, rotation_matrix.T)  # Apply rotation
        return rotated_coords[..., 0], rotated_coords[..., 1], rotated_coords[..., 2]

    def rotation_matrix(self, theta_x, theta_y, theta_z):
        """
        Generate a 3D rotation matrix for rotation around X, Y, and Z axes.

        Args:
            theta_x (float): Rotation angle (radians) around the X-axis.
            theta_y (float): Rotation angle (radians) around the Y-axis.
            theta_z (float): Rotation angle (radians) around the Z-axis.

        Returns:
            numpy.ndarray: Combined rotation matrix (3x3).
        """
        # Rotation matrices for each axis
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])
        R_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])
        R_z = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])

        # Combined rotation matrix
        return R_z @ R_y @ R_x

    def _create_interpolation_splines(self):
        """Create smooth splines for blade geometry distributions."""
        # Required splines
        self.r_R_chord = self.chorddist['r/R']
        self.c_R = self.chorddist['c/R']
        self.r_R_pitch = self.pitchdist['r/R']
        self.pitch_angle = self.pitchdist['twist (deg)']
        
        # Optional splines
        if self.sweepdist is not None:
            self.r_R_sweep = self.sweepdist['r/R']
            self.sweep_offset = self.sweepdist['y/R (y-distance of LE from the middle point of hub)']
            self.sweep_spline = UnivariateSpline(self.r_R_sweep, self.sweep_offset, k=5, s=1e-8)
        else:
            self.sweep_spline = None
            
        if self.heightdist is not None:
            self.r_R_height = self.heightdist['r/R']
            self.height_offset = self.heightdist['z/R  (height of leading edge from top face of hub)']
            self.height_spline = UnivariateSpline(self.r_R_height, self.height_offset, k=5, s=1e-8)
        else:
            self.height_spline = None
            
        self.chord_spline = UnivariateSpline(self.r_R_chord, self.c_R, k=5, s=1e-8)
        self.pitch_spline = UnivariateSpline(self.r_R_pitch, self.pitch_angle, k=5, s=1e-8)

    def _load_airfoil_contours(self):
        """Load airfoil shapes for each r/R."""
        contours = {}
        for _, row in self.airfoil_distribution.iterrows():
            r_R = row['r/R']
            contour_file = row['Contour file']
            airfoil_data = pd.read_csv(contour_file)
            contours[r_R] = airfoil_data
        return contours

    def _interpolate_airfoil(self, r_R_target, n_points=83):
        """
        Interpolate airfoil shape at the given spanwise location (r/R) with consistent point count.
        
        Args:
            r_R_target (float): The target radial position (r/R)
            n_points (int): Number of points to use for all airfoil sections
            
        Returns:
            pandas.DataFrame: Interpolated airfoil coordinates with consistent point count
        """
        r_R_values = np.array(sorted(self.airfoil_contours.keys()))
        lower_idx = np.searchsorted(r_R_values, r_R_target) - 1
        upper_idx = lower_idx + 1

        # Handle edge cases
        if lower_idx < 0:
            lower_idx = 0
        if upper_idx >= len(r_R_values):
            upper_idx = len(r_R_values) - 1

        r_R_lower = r_R_values[lower_idx]
        r_R_upper = r_R_values[upper_idx]

        # Get the two nearest airfoil sections
        airfoil_lower = self.airfoil_contours[r_R_lower]
        airfoil_upper = self.airfoil_contours[r_R_upper]
        
        # Function to resample an airfoil to n_points
        def resample_airfoil(airfoil_df, n_points):
            # Create normalized position array (0 to 1)
            t = np.linspace(0, 1, len(airfoil_df))
            # Create new normalized positions
            t_new = np.linspace(0, 1, n_points)
            # Interpolate x/c and y/c
            x_new = np.interp(t_new, t, airfoil_df['x/c'].values)
            y_new = np.interp(t_new, t, airfoil_df['y/c'].values)
            return pd.DataFrame({'x/c': x_new, 'y/c': y_new})

        # Resample both airfoils to have the same number of points
        airfoil_lower_resampled = resample_airfoil(airfoil_lower, n_points)
        airfoil_upper_resampled = resample_airfoil(airfoil_upper, n_points)

        # Linear interpolation between the two resampled airfoils
        if r_R_lower == r_R_upper:
            return airfoil_lower_resampled

        weight = (r_R_target - r_R_lower) / (r_R_upper - r_R_lower)
        x_interp = (airfoil_lower_resampled['x/c'] * (1 - weight) + 
                    airfoil_upper_resampled['x/c'] * weight)
        y_interp = (airfoil_lower_resampled['y/c'] * (1 - weight) + 
                    airfoil_upper_resampled['y/c'] * weight)

        return pd.DataFrame({'x/c': x_interp, 'y/c': y_interp})

    def generate_blade_geometry(self):
        """Generate 3D blade geometry based on distributions."""
        r_R_interpolated = np.linspace(
            self.R_hub / self.R_tip,
            1.0,
            100
        )
        
        X_list, Y_list, Z_list = [], [], []
        for r_R in r_R_interpolated:
            r_actual = r_R * self.R_tip
            chord_length = self.chord_spline(r_R) * self.R_tip
            twist_angle = -np.radians(self.pitch_spline(r_R))
            
            # Get optional height and sweep values (default to 0 if not provided)
            height = self.height_spline(r_R) * self.R_tip if self.height_spline else 0
            sweep = self.sweep_spline(r_R) * self.R_tip if self.sweep_spline else 0
            
            # Interpolate airfoil shape
            airfoil = self._interpolate_airfoil(r_R)
            x_scaled = airfoil['x/c'].values * chord_length
            z_scaled = airfoil['y/c'].values * chord_length
            
            # Apply twist (rotation)
            x_rotated = x_scaled * np.cos(twist_angle) - z_scaled * np.sin(twist_angle)
            z_rotated = x_scaled * np.sin(twist_angle) + z_scaled * np.cos(twist_angle)
            
            # Final positions (including sweep and height adjustments if available)
            x_final = x_rotated - sweep
            z_final = z_rotated + height
            y_final = np.full_like(x_final, r_actual)
            
            X_list.append(x_final)
            Y_list.append(y_final)
            Z_list.append(z_final)
            
        theta_x, theta_y, theta_z = np.radians([0, 0, 180])
        rotation_mat = self.rotation_matrix(theta_x=theta_x, theta_y=theta_y, theta_z=theta_z)
        X = np.array(X_list)
        Y = np.array(Y_list)
        Z = np.array(Z_list)
        X2, Y2, Z2 = self.apply_rotation(X, Y, Z, rotation_mat)
        
        return X, Y, Z, X2, Y2, Z2
    
    def create_blade_surface(self, X, Y, Z):
        """
        Creates a smooth surface for the blade by interpolating between spanwise sections.

        Args:
            X, Y, Z: Spanwise coordinates (arrays of the same shape).

        Returns:
            Meshgrid arrays for X, Y, and Z to represent the blade surface.
        """
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)

        chordwise_points = X.shape[1]  # Number of points per section (chordwise)
        spanwise_points = X.shape[0]   # Number of sections (spanwise)

        # Create interpolation grids
        chord_grid = np.linspace(0, 1, chordwise_points)  # Normalized chordwise
        span_grid = np.linspace(0, 1, spanwise_points)    # Normalized spanwise

        # Interpolate along the span
        X_surface = np.zeros((spanwise_points, chordwise_points))
        Y_surface = np.zeros((spanwise_points, chordwise_points))
        Z_surface = np.zeros((spanwise_points, chordwise_points))

        for j in range(chordwise_points):
            # Interpolate along the span for each chordwise point
            X_surface[:, j] = np.interp(span_grid, span_grid, X[:, j])
            Y_surface[:, j] = np.interp(span_grid, span_grid, Y[:, j])
            Z_surface[:, j] = np.interp(span_grid, span_grid, Z[:, j])

        return X_surface, Y_surface, Z_surface
   
    def save_blade_surfaces_to_mat(self,file_name, X1, Y1, Z1, X2, Y2, Z2):
        """
        Save the blade surface coordinates to a .mat file for MATLAB visualization.

        Args:
            file_name (str): Name of the .mat file to save.
            X1, Y1, Z1: Surface coordinates for the first blade.
            X2, Y2, Z2: Surface coordinates for the second blade.
        """
        data = {
            'X1': X1,
            'Y1': Y1,
            'Z1': Z1,
            'X2': X2,
            'Y2': Y2,
            'Z2': Z2
        }
        savemat(file_name, data)
        print(f"Blade surface data saved to {file_name}")

    def compute_pairwise_midpoints_unsorted(self, r_R_target):
        """10
        Compute the pairwise midpoints of the y/c values:
        - First x/c's y/c pairs with the last x/c's y/c.
        - Second x/c's y/c pairs with the second-last x/c's y/c, and so on.

        Args:
            r_R_target (float): The spanwise location to compute midpoints.

        Returns:
            pandas.DataFrame: A DataFrame with columns ['x/c_pair1', 'x/c_pair2', 'y_mid/c'].
        """
        # Interpolate the airfoil shape at the specified r/R location
        airfoil = self._interpolate_airfoil(r_R_target)

        # Extract x/c and y/c values without sorting
        x_c = airfoil['x/c'].values
        y_c = airfoil['y/c'].values

        # Initialize storage for results
        x_c_pair1 = []
        x_c_pair2 = []
        y_mid = []

        # Pair the points symmetrically
        n = len(x_c)
        for i in range((n + 1) // 2):  # Loop until the middle (inclusive for odd n)
            # Pair the first with the last, second with second-last, etc.
            x_c_pair1.append(x_c[i])
            x_c_pair2.append(x_c[n - 1 - i])
            y_mid.append((y_c[i] + y_c[n - 1 - i]) / 2)

        # Create a DataFrame to store results
        midpoints_df = pd.DataFrame({
            'x/c': x_c_pair1,
            'x/c_pair2': x_c_pair2,
            'y_c': y_mid
        })

        return midpoints_df
 
    def generate_flat_blade_surface(self, span_resolution, chord_resolution):
        """
        Generate a flat blade surface by accounting for optional height, pitch, sweep, and chord.
        The surface follows the mean camber line.
        """
        r_R_interpolated = np.linspace(
            self.R_hub / self.R_tip,
            1.0,
            span_resolution
        )
        
        X_flat, Y_flat, Z_flat = [], [], []
        
        for r_R in r_R_interpolated:
            r_actual = r_R * self.R_tip
            chord_length = self.chord_spline(r_R) * self.R_tip
            twist_angle = -np.radians(self.pitch_spline(r_R))
            
            # Get optional height and sweep values (default to 0 if not provided)
            height = self.height_spline(r_R) * self.R_tip if self.height_spline else 0
            sweep = self.sweep_spline(r_R) * self.R_tip if self.sweep_spline else 0
            
            # Compute mean camber line
            airfoil = self._interpolate_airfoil(r_R)
            midpoints_df = self.compute_pairwise_midpoints_unsorted(r_R)
            
            x_c_original = midpoints_df['x/c'].values
            y_mid_original = midpoints_df['y_c'].values
            
            x_c_resampled = np.linspace(x_c_original.min(), x_c_original.max(), chord_resolution)
            y_mid_resampled = np.interp(x_c_resampled, x_c_original, y_mid_original)
            
            x_scaled = x_c_resampled * chord_length
            z_scaled = y_mid_resampled * chord_length
            
            x_rotated = x_scaled * np.cos(twist_angle) - z_scaled * np.sin(twist_angle)
            z_rotated = x_scaled * np.sin(twist_angle) + z_scaled * np.cos(twist_angle)
            
            x_final = x_rotated - sweep
            y_final = np.full_like(x_final, r_actual)
            z_final = z_rotated + height
            
            X_flat.append(x_final)
            Y_flat.append(y_final)
            Z_flat.append(z_final)
            
        X_flat = np.array(X_flat)
        Y_flat = np.array(Y_flat)
        Z_flat = np.array(Z_flat)
        return X_flat, Y_flat, Z_flat

    def generate_flat_blade_surfaces(self, span_resolution=14, chord_resolution=6):
        """
        Generate flat blade surfaces for both blades by considering height, pitch, sweep, and chord.
        The second blade is generated by rotating the first blade 180 degrees.

        Args:
            span_resolution (int): Number of sections along the span.
            chord_resolution (int): Number of points along each chord.

        Returns:
            tuple: Meshgrid arrays (X1_flat, Y1_flat, Z1_flat, X2_flat, Y2_flat, Z2_flat) 
                for both blades' flat surfaces.
        """
        # Generate the flat surface for the first blade
        X1_flat, Y1_flat, Z1_flat = self.generate_flat_blade_surface(span_resolution, chord_resolution)

        # Initialize lists to store all blade surfaces
        X_flats = [X1_flat]
        Y_flats = [Y1_flat]
        Z_flats = [Z1_flat]

        # Calculate angular spacing between blades
        angle_spacing = 2 * np.pi / self.num_blades

        # Generate remaining blades by rotating the first blade
        for i in range(1, self.num_blades):
            rotation_angle = i * angle_spacing  # Angle for current blade
            
            # Create rotation matrix for current blade
            rotation_matrix = np.array([
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle),  np.cos(rotation_angle), 0],
                [0,                      0,                     1]
            ])

            # Flatten the arrays for transformation
            points = np.stack([X1_flat.ravel(), Y1_flat.ravel(), Z1_flat.ravel()], axis=1)

            # Apply the rotation
            points_rotated = points @ rotation_matrix.T

            # Reshape back to original dimensions and store
            X_flats.append(points_rotated[:, 0].reshape(X1_flat.shape))
            Y_flats.append(points_rotated[:, 1].reshape(Y1_flat.shape))
            Z_flats.append(points_rotated[:, 2].reshape(Z1_flat.shape))

        return X_flats, Y_flats, Z_flats

    def plot_flat_blades_surface(self, X_flats, Y_flats, Z_flats):
        """
        Plot the flat blade surfaces for all blades with uniform scaling.

        Args:
            X_flats, Y_flats, Z_flats: Lists of arrays representing blade surfaces.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Colors for different blades
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_blades))

        # Plot each blade surface
        for i in range(self.num_blades):
            ax.plot_surface(X_flats[i], Y_flats[i], Z_flats[i], 
                          color=colors[i], alpha=0.8, edgecolor='none', 
                          label=f'Blade {i+1}')

        # Set labels and title
        ax.set_title(f'{self.num_blades}-Blade Propeller Surfaces', fontsize=14)
        ax.set_xlabel('Chordwise Direction (X)', fontsize=12)
        ax.set_ylabel('Spanwise Direction (Y)', fontsize=12)
        ax.set_zlabel('Vertical Direction (Z)', fontsize=12)

        # Set uniform scaling
        x_range = [min(X.min() for X in X_flats), max(X.max() for X in X_flats)]
        y_range = [min(Y.min() for Y in Y_flats), max(Y.max() for Y in Y_flats)]
        z_range = [min(Z.min() for Z in Z_flats), max(Z.max() for Z in Z_flats)]

        max_range = max(max(x_range) - min(x_range),
                       max(y_range) - min(y_range),
                       max(z_range) - min(z_range)) / 2.0

        mid_x = sum(x_range) / 2.0
        mid_y = sum(y_range) / 2.0
        mid_z = sum(z_range) / 2.0

        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        ax.set_box_aspect([1, 1, 1])  # Force uniform scaling
        ax.view_init(elev=90, azim=90)
        
        plt.show()

# # Initialize the propeller geometry
# propeller = PropellerGeometry(
#     airfoil_distribution_file='DJI9443_airfoils.csv',
#     chorddist_file='DJI9443_chorddist.csv',
#     pitchdist_file='DJI9443_pitchdist.csv',
#     sweepdist_file='DJI9443_sweepdist.csv',
#     heightdist_file='DJI9443_heightdist.csv',
#     R_tip=0.12,  # Tip radius in meters
#     R_hub=0.0064   # Hub radius in meters
# )

# X, Y, Z, X2, Y2, Z2 = propeller._generate_blade_geometry()

# # Create blade surfaces for both blades
# X1_surf, Y1_surf, Z1_surf = propeller.create_blade_surface(X, Y, Z)
# X2_surf, Y2_surf, Z2_surf = propeller.create_blade_surface(X2, Y2, Z2)

# # print(f"Spanwise extent (actual): {X.min()} to {X.max()} meters")
# # # propeller.plot_blades_surface(X1_surf, Y1_surf, Z1_surf, X2_surf, Y2_surf, Z2_surf)

# # Save blade surfaces to a .mat file
# propeller.save_blade_surfaces_to_mat('blade_surfaces.mat', X1_surf, Y1_surf, Z1_surf, X2_surf, Y2_surf, Z2_surf)


# Generate flat blade geometry
# X1_flat, Y1_flat, Z1_flat, X2_flat, Y2_flat, Z2_flat = PropellerGeometry.generate_flat_blade_surfaces()

# propeller.plot_flat_blades_surface(X1_flat, Y1_flat, Z1_flat, X2_flat, Y2_flat, Z2_flat)
# Generate flat blade surfaces
# X1_flat, Y1_flat, Z1_flat, X2_flat, Y2_flat, Z2_flat = PropellerGeometry.generate_flat_blade_surfaces()

# # Save flat blade surfaces to a .mat file
# propeller.save_flat_blade_surfaces_to_mat('flat_blade_surfaces.mat', X1_flat, Y1_flat, Z1_flat, X2_flat, Y2_flat, Z2_flat)

