import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline, LinearNDInterpolator, NearestNDInterpolator
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
            num_blades (int): Number of blades to generate.
        """
        # Store basic parameters
        self.num_blades = num_blades
        self.R_tip = R_tip
        self.R_hub = R_hub

        # Load required data
        self.airfoil_distribution = pd.read_csv(airfoil_distribution_file)
        # Load airfoils at each cross section
        self.airfoil_contours = self._load_airfoil_contours()
        # Load polar data (cl, cd, cd vs aoa) for aerodynamic analysis
        self.airfoil_polars = self._load_polar_data()
        
        # Load geometric properties of blade
        self.chorddist = pd.read_csv(chorddist_file)
        self.pitchdist = pd.read_csv(pitchdist_file)
            # Load optional data
        self.sweepdist = pd.read_csv(sweepdist_file) if sweepdist_file else None
        self.heightdist = pd.read_csv(heightdist_file) if heightdist_file else None

        # Create interpolation splines for smooth geoemtric properties of blade: pitch, spline, chord, sweep, height
        self._create_interpolation_splines()
        
        # Initialize polar interpolation points and values
        self._initialize_polar_interpolation()
    
    def _load_airfoil_contours(self):
        """Load airfoil shapes for each r/R."""
        contours = {}
        for _, row in self.airfoil_distribution.iterrows():
            r_R = row['r/R']
            contour_file = row['Contour file']
            airfoil_data = pd.read_csv(contour_file)
            contours[r_R] = airfoil_data
        return contours
   
    def _load_polar_data(self):
        """Load polar data for each r/R from the Aero file column."""
        polars = {}
        for _, row in self.airfoil_distribution.iterrows():
            r_R = row['r/R']
            polar_file = row['Aero file']

            # Load polar data with columns for alpha, cl, cd, cm
            polar_data = pd.read_csv(polar_file)
            polars[r_R] = polar_data
            # except Exception as e:
            #     print(f"Warning: Could not load polar file {polar_file} for r/R = {r_R}: {e}")
        return polars
    
    def _create_interpolation_splines(self):
        """Create smooth splines for blade geometry distributions."""
        # Required splines
        self.chord_spline = UnivariateSpline(
            self.chorddist['r/R'], 
            self.chorddist['c/R'], 
            k=5, s=1e-8
        )
        
        self.pitch_spline = UnivariateSpline(
            self.pitchdist['r/R'], 
            self.pitchdist['twist (deg)'], 
            k=5, s=1e-8
        )
        
        # Optional splines
        if self.sweepdist is not None:
            self.sweep_spline = UnivariateSpline(
                self.sweepdist['r/R'],
                self.sweepdist['y/R (y-distance of LE from the middle point of hub)'],
                k=5, s=1e-8
            )
        else:
            self.sweep_spline = None
            
        if self.heightdist is not None:
            self.height_spline = UnivariateSpline(
                self.heightdist['r/R'],
                self.heightdist['z/R  (height of leading edge from top face of hub)'],
                k=5, s=1e-8
            )
        else:
            self.height_spline = None

    def _initialize_polar_interpolation(self):
        """Initialize data structures for polar interpolation."""
        # if not self.airfoil_polars or len(self.airfoil_polars) == 0:
        #     self.points = None
        #     self.cl_interp = None
        #     self.cd_interp = None
        #     self.cm_interp = None
        #     return
            
        # Initialize data structures
        points = []  # Will hold (r/R, alpha) pairs
        cl_values = []
        cd_values = []
        cm_values = []
        
        # Extract all data points from the polar data
        for r_R, polar_df in self.airfoil_polars.items():
            for _, row in polar_df.iterrows():
                # if 'Alpha' in row and isinstance(row['Alpha'], (int, float)) and not np.isnan(row['Alpha']):
                alpha = float(row['Alpha'])
                
                # Only add points if we have valid coefficient data
                cl_val = row.get('Cl')
                cd_val = row.get('Cd')
                cm_val = row.get('Cm')
                
                # # Convert None to NaN
                # cl_val = np.nan if cl_val is None else cl_val
                # cd_val = np.nan if cd_val is None else cd_val
                # cm_val = np.nan if cm_val is None else cm_val
                
                # Only add if at least one coefficient is valid
                # if not (np.isnan(cl_val) and np.isnan(cd_val) and np.isnan(cm_val)):
                points.append([float(r_R), alpha])
                cl_values.append(cl_val)
                cd_values.append(cd_val)
                cm_values.append(cm_val)
        
        # # If we have no valid points, don't create interpolators
        # if len(points) == 0:
        #     self.points = None
        #     self.cl_interp = None
        #     self.cd_interp = None
        #     self.cm_interp = None
        #     return
            
        # Convert to numpy arrays
        self.points = np.array(points)
        self.cl_values = np.array(cl_values, dtype=float)
        self.cd_values = np.array(cd_values, dtype=float)
        self.cm_values = np.array(cm_values, dtype=float)
        
        # Create interpolators
        self.cl_interp = self._create_coefficient_interpolator(self.cl_values)
        self.cd_interp = self._create_coefficient_interpolator(self.cd_values)
        self.cm_interp = self._create_coefficient_interpolator(self.cm_values)
    
    def _create_coefficient_interpolator(self, values):
        
        """Create an interpolator for coefficient values, handling NaN values."""
        # Filter out NaN values
        valid = ~np.isnan(values)
        
        # Check if we have valid points
        if np.sum(valid) < 3:  # Need at least 3 points for reasonable interpolation
            # If we have at least one valid point, use nearest neighbor
            if np.sum(valid) >= 1:
                return NearestNDInterpolator(self.points[valid], values[valid])
            else:
                # No valid points
                return None
        else:
            # Use linear interpolation with nearest for extrapolation
            try:
                return LinearNDInterpolator(self.points[valid], values[valid], fill_value=np.nan)
            except Exception:
                # If linear interpolation fails, try nearest neighbor
                return NearestNDInterpolator(self.points[valid], values[valid]) if np.sum(valid) >= 1 else None

    def _interpolate_airfoil(self, r_R_target, n_points=83):
        """
        Interpolate airfoil shape at the given spanwise location (r/R).
        
        Args:
            r_R_target (float): The target radial position (r/R)
            n_points (int): Number of points to use for all airfoil sections
            
        Returns:
            pandas.DataFrame: Interpolated airfoil coordinates
        """
        r_R_values = np.array(sorted(self.airfoil_contours.keys()))
        
        # Find the two nearest r/R values
        lower_idx = np.searchsorted(r_R_values, r_R_target) - 1
        lower_idx = max(0, lower_idx)  # Ensure not below 0
        upper_idx = min(lower_idx + 1, len(r_R_values) - 1)  # Ensure not above max
        
        r_R_lower = r_R_values[lower_idx]
        r_R_upper = r_R_values[upper_idx]
        
        # Handle the case where target falls exactly on a known airfoil
        if r_R_lower == r_R_upper:
            airfoil = self.airfoil_contours[r_R_lower]
            t = np.linspace(0, 1, len(airfoil))
            t_new = np.linspace(0, 1, n_points)
            x_new = np.interp(t_new, t, airfoil['x/c'].values)
            y_new = np.interp(t_new, t, airfoil['y/c'].values)
            return pd.DataFrame({'x/c': x_new, 'y/c': y_new})
        
        # Get and resample the two nearest airfoil sections
        airfoil_lower = self.airfoil_contours[r_R_lower]
        airfoil_upper = self.airfoil_contours[r_R_upper]
        
        # Resample both airfoils to the same number of points
        t_lower = np.linspace(0, 1, len(airfoil_lower))
        t_upper = np.linspace(0, 1, len(airfoil_upper))
        t_new = np.linspace(0, 1, n_points)
        
        x_lower = np.interp(t_new, t_lower, airfoil_lower['x/c'].values)
        y_lower = np.interp(t_new, t_lower, airfoil_lower['y/c'].values)
        
        x_upper = np.interp(t_new, t_upper, airfoil_upper['x/c'].values)
        y_upper = np.interp(t_new, t_upper, airfoil_upper['y/c'].values)
        
        # Linear interpolation between the two airfoils
        weight = (r_R_target - r_R_lower) / (r_R_upper - r_R_lower)
        x_interp = x_lower * (1 - weight) + x_upper * weight
        y_interp = y_lower * (1 - weight) + y_upper * weight
        
        return pd.DataFrame({'x/c': x_interp, 'y/c': y_interp})
        
    def _find_nearest_polar_stations(self, r_R_target):
        """
        Find the two nearest polar data stations for interpolation.
        
        Args:
            r_R_target (float): Target radial position (r/R)
            
        Returns:
            tuple: (r_R_lower, r_R_upper, weight) - The two nearest stations and interpolation weight
        """
        r_R_values = np.array(sorted(self.airfoil_polars.keys()))
        
        # Find the two nearest r/R values
        lower_idx = np.searchsorted(r_R_values, r_R_target) - 1
        lower_idx = max(0, lower_idx)  # Ensure not below 0
        upper_idx = min(lower_idx + 1, len(r_R_values) - 1)  # Ensure not above max
        
        r_R_lower = r_R_values[lower_idx]
        r_R_upper = r_R_values[upper_idx]
        
        # Calculate interpolation weight
        if r_R_lower == r_R_upper:
            weight = 0.0
        else:
            weight = (r_R_target - r_R_lower) / (r_R_upper - r_R_lower)
            
        return r_R_lower, r_R_upper, weight
        
    def get_polar_data(self, r_R_target, alpha=None):
        """
        Get interpolated polar data for the given r/R position.
        
        Args:
            r_R_target (float): Target radial position (r/R)
            alpha (float, optional): If provided, returns interpolated coefficients 
                                    at this specific angle of attack.
                                    
        Returns:
            pandas.DataFrame or dict: If alpha is None, returns full interpolated polar data.
                                     If alpha is provided, returns dict with cl, cd, cm values.
        """
        if not self.airfoil_polars or not hasattr(self, 'points') or self.points is None:
            raise ValueError("No polar data available")
        
        # If requesting coefficients at a specific alpha
        if alpha is not None:
            return self._get_interpolated_coefficients(r_R_target, alpha)
        
        # For full polar interpolation, create an alpha range and get coefficients for each alpha
        # Find the global min/max alpha range across all airfoils
        alpha_min = float('inf')
        alpha_max = float('-inf')
        
        for polar in self.airfoil_polars.values():
            if 'Alpha' in polar.columns:
                alpha_values = polar['Alpha'].values
                alpha_min = min(alpha_min, np.nanmin(alpha_values))
                alpha_max = max(alpha_max, np.nanmax(alpha_values))
        
        # Create a range of alpha values
        alpha_range = np.linspace(alpha_min, alpha_max, 100)
        
        # Initialize result DataFrame
        result = pd.DataFrame({'Alpha': alpha_range})
        
        # Get coefficients for each alpha
        for a in alpha_range:
            coeffs = self._get_interpolated_coefficients(r_R_target, a)
            idx = result['Alpha'] == a
            result.loc[idx, 'Cl'] = coeffs['cl'] if coeffs['cl'] is not None else np.nan
            result.loc[idx, 'Cd'] = coeffs['cd'] if coeffs['cd'] is not None else np.nan
            result.loc[idx, 'Cm'] = coeffs['cm'] if coeffs['cm'] is not None else np.nan
                
        return result
    
    def _get_interpolated_coefficients(self, r_R, alpha):
        """Get interpolated coefficient values at a specific r/R and alpha."""
        point = np.array([[float(r_R), float(alpha)]])
        
        # Initialize default results
        cl, cd, cm = None, None, None
        
        # Get interpolated values, handling all potential errors
        try:
            if self.cl_interp is not None:
                cl_val = self.cl_interp(point)
                if not np.isnan(cl_val).all():
                    cl = float(cl_val[0]) if isinstance(cl_val, np.ndarray) else float(cl_val)
        except Exception:
            cl = None
            
        try:
            if self.cd_interp is not None:
                cd_val = self.cd_interp(point)
                if not np.isnan(cd_val).all():
                    cd = float(cd_val[0]) if isinstance(cd_val, np.ndarray) else float(cd_val)
        except Exception:
            cd = None
            
        try:
            if self.cm_interp is not None:
                cm_val = self.cm_interp(point)
                if not np.isnan(cm_val).all():
                    cm = float(cm_val[0]) if isinstance(cm_val, np.ndarray) else float(cm_val)
        except Exception:
            cm = None
            
        # Check for NaN values
        if cl is not None and np.isnan(cl):
            cl = None
        if cd is not None and np.isnan(cd):
            cd = None
        if cm is not None and np.isnan(cm):
            cm = None
            
        return {'cl': cl, 'cd': cd, 'cm': cm}
        
    def _get_coefficients_at_alpha(self, polar_data, alpha):
        """
        Get interpolated coefficients at a specific angle of attack.
        
        Args:
            polar_data (pandas.DataFrame): Polar data with Alpha, Cl, Cd, Cm columns
            alpha (float): Angle of attack in degrees
            
        Returns:
            dict: Dictionary with interpolated Cl, Cd, Cm values
        """
        result = {}
        alpha_values = polar_data['Alpha'].values
        
        # Check if the requested alpha is within the range of available data
        if alpha < alpha_values.min() or alpha > alpha_values.max():
            # If outside range, we'll set coefficients to None
            for coef in ['cl', 'cd', 'cm']:
                result[coef] = None
            return result
            
        # Get interpolated coefficients for each parameter
        for coef_key, coef_name in zip(['cl', 'cd', 'cm'], ['Cl', 'Cd', 'Cm']):
            if coef_name in polar_data.columns:
                # Check for NaN values in the coefficient data
                coef_values = polar_data[coef_name].values
                if np.isnan(coef_values).any():
                    # Remove NaN values before interpolation
                    valid_indices = ~np.isnan(coef_values)
                    if np.sum(valid_indices) > 1:  # Need at least 2 points for interpolation
                        result[coef_key] = np.interp(
                            alpha, 
                            alpha_values[valid_indices], 
                            coef_values[valid_indices]
                        )
                    else:
                        result[coef_key] = None
                else:
                    # Normal interpolation if no NaN values
                    result[coef_key] = np.interp(alpha, alpha_values, coef_values)
            else:
                result[coef_key] = None
                
        return result

    def rotation_matrix(self, theta_x=0, theta_y=0, theta_z=0):
        """Generate a 3D rotation matrix for rotation around X, Y, and Z axes."""
        # X-axis rotation
        R_x = np.array([
            [1, 0, 0],
            [0, np.cos(theta_x), -np.sin(theta_x)],
            [0, np.sin(theta_x), np.cos(theta_x)]
        ])
        
        # Y-axis rotation
        R_y = np.array([
            [np.cos(theta_y), 0, np.sin(theta_y)],
            [0, 1, 0],
            [-np.sin(theta_y), 0, np.cos(theta_y)]
        ])
        
        # Z-axis rotation
        R_z = np.array([
            [np.cos(theta_z), -np.sin(theta_z), 0],
            [np.sin(theta_z), np.cos(theta_z), 0],
            [0, 0, 1]
        ])
        
        # Combined rotation matrix
        return R_z @ R_y @ R_x

    def apply_rotation(self, X, Y, Z, rotation_matrix):
        """Apply a rotation matrix to 3D coordinates."""
        points = np.stack((X, Y, Z), axis=-1)  # Combine into (N, 3)
        points_rotated = np.dot(points, rotation_matrix.T)  # Apply rotation
        return points_rotated[..., 0], points_rotated[..., 1], points_rotated[..., 2]

    def compute_pairwise_midpoints_unsorted(self, r_R_target):
        """
        Compute the pairwise midpoints of the y/c values for camber line calculation.
        """
        airfoil = self._interpolate_airfoil(r_R_target)
        
        x_c = airfoil['x/c'].values
        y_c = airfoil['y/c'].values
        
        n = len(x_c)
        x_c_pair1 = []
        y_mid = []
        
        # Pair the points symmetrically
        for i in range((n + 1) // 2):
            x_c_pair1.append(x_c[i])
            y_mid.append((y_c[i] + y_c[n - 1 - i]) / 2)
        
        return pd.DataFrame({
            'x/c': x_c_pair1,
            'y_c': y_mid
        })

    def generate_blade_geometry(self):
        """Generate 3D blade geometry based on distributions."""
        # Create normalized span positions
        r_R_interpolated = np.linspace(
            self.R_hub / self.R_tip,
            1.0,
            100
        )
        
        X_list, Y_list, Z_list = [], [], []
        
        # Generate geometry section by section
        for r_R in r_R_interpolated:
            # Calculate actual radius and chord length
            r_actual = r_R * self.R_tip
            chord_length = self.chord_spline(r_R) * self.R_tip
            twist_angle = -np.radians(self.pitch_spline(r_R))
            
            # Get optional geometry adjustments
            height = self.height_spline(r_R) * self.R_tip if self.height_spline else 0
            sweep = self.sweep_spline(r_R) * self.R_tip if self.sweep_spline else 0
            
            # Get airfoil shape and scale it
            airfoil = self._interpolate_airfoil(r_R)
            x_scaled = airfoil['x/c'].values * chord_length
            z_scaled = airfoil['y/c'].values * chord_length
            
            # Apply twist rotation
            x_rotated = x_scaled * np.cos(twist_angle) - z_scaled * np.sin(twist_angle)
            z_rotated = x_scaled * np.sin(twist_angle) + z_scaled * np.cos(twist_angle)
            
            # Apply sweep and height offsets
            x_final = x_rotated - sweep
            y_final = np.full_like(x_final, r_actual)
            z_final = z_rotated + height
            
            X_list.append(x_final)
            Y_list.append(y_final)
            Z_list.append(z_final)
        
        # Convert to numpy arrays
        X = np.array(X_list)
        Y = np.array(Y_list)
        Z = np.array(Z_list)
        
        # Create the second blade by rotating 180 degrees
        rotation_mat = self.rotation_matrix(theta_z=np.radians(180))
        X2, Y2, Z2 = self.apply_rotation(X, Y, Z, rotation_mat)
        
        return X, Y, Z, X2, Y2, Z2
    
    def create_blade_surface(self, X, Y, Z):
        """Creates a smooth surface for the blade."""
        # Ensure we're working with numpy arrays
        X = np.array(X)
        Y = np.array(Y)
        Z = np.array(Z)
        
        # Surface is already created by the generate_blade_geometry
        # Just return the arrays directly
        return X, Y, Z

    def generate_flat_blade_surface(self, span_resolution, chord_resolution):
        """
        Generate a flat blade surface following the mean camber line.
        """
        r_R_interpolated = np.linspace(
            self.R_hub / self.R_tip,
            1.0,
            span_resolution
        )
        
        X_flat, Y_flat, Z_flat = [], [], []
        
        for r_R in r_R_interpolated:
            # Get section properties
            r_actual = r_R * self.R_tip
            chord_length = self.chord_spline(r_R) * self.R_tip
            twist_angle = -np.radians(self.pitch_spline(r_R))
            height = self.height_spline(r_R) * self.R_tip if self.height_spline else 0
            sweep = self.sweep_spline(r_R) * self.R_tip if self.sweep_spline else 0
            
            # Compute mean camber line
            midpoints_df = self.compute_pairwise_midpoints_unsorted(r_R)
            
            # Resample the camber line to desired resolution
            x_c_original = midpoints_df['x/c'].values
            y_mid_original = midpoints_df['y_c'].values
            x_c_resampled = np.linspace(x_c_original.min(), x_c_original.max(), chord_resolution)
            y_mid_resampled = np.interp(x_c_resampled, x_c_original, y_mid_original)
            
            # Scale and transform
            x_scaled = x_c_resampled * chord_length
            z_scaled = y_mid_resampled * chord_length
            
            # Apply twist
            x_rotated = x_scaled * np.cos(twist_angle) - z_scaled * np.sin(twist_angle)
            z_rotated = x_scaled * np.sin(twist_angle) + z_scaled * np.cos(twist_angle)
            
            # Apply sweep and height
            x_final = x_rotated - sweep
            y_final = np.full_like(x_final, r_actual)
            z_final = z_rotated + height
            
            X_flat.append(x_final)
            Y_flat.append(y_final)
            Z_flat.append(z_final)
        
        return np.array(X_flat), np.array(Y_flat), np.array(Z_flat)

    def generate_flat_blade_surfaces(self, span_resolution=20, chord_resolution=15):
        """
        Generate flat blade surfaces for all blades.
        """
        # Generate the flat surface for the first blade
        X1_flat, Y1_flat, Z1_flat = self.generate_flat_blade_surface(span_resolution, chord_resolution)
        
        # Initialize lists to store all blade surfaces
        X_flats = [X1_flat]
        Y_flats = [Y1_flat]
        Z_flats = [Z1_flat]
        
        # Calculate angular spacing between blades and generate remaining blades
        angle_spacing = 2 * np.pi / self.num_blades
        
        for i in range(1, self.num_blades):
            # Calculate rotation for this blade
            rotation_angle = i * angle_spacing
            rotation_matrix = np.array([
                [np.cos(rotation_angle), -np.sin(rotation_angle), 0],
                [np.sin(rotation_angle),  np.cos(rotation_angle), 0],
                [0, 0, 1]
            ])
            
            # Apply rotation to first blade to create additional blades
            points = np.stack([X1_flat.ravel(), Y1_flat.ravel(), Z1_flat.ravel()], axis=1)
            points_rotated = points @ rotation_matrix.T
            
            # Store the new blade
            X_flats.append(points_rotated[:, 0].reshape(X1_flat.shape))
            Y_flats.append(points_rotated[:, 1].reshape(Y1_flat.shape))
            Z_flats.append(points_rotated[:, 2].reshape(Z1_flat.shape))
        
        return X_flats, Y_flats, Z_flats

    def save_blade_surfaces_to_mat(self, file_name, X1, Y1, Z1, X2, Y2, Z2):
        """Save the blade surface coordinates to a .mat file for MATLAB visualization."""
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

    def plot_flat_blades_surface(self, X_flats, Y_flats, Z_flats):
        """Plot the flat blade surfaces for all blades with uniform scaling."""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Colors for different blades
        colors = plt.cm.rainbow(np.linspace(0, 1, self.num_blades))
        
        # Plot each blade surface
        for i in range(self.num_blades):
            ax.plot_surface(
                X_flats[i], Y_flats[i], Z_flats[i],
                color=colors[i], alpha=0.8, edgecolor='none',
                label=f'Blade {i+1}'
            )
        
        # Get global min/max for uniform scaling
        all_mins = [
            min(X.min() for X in X_flats),
            min(Y.min() for Y in Y_flats),
            min(Z.min() for Z in Z_flats)
        ]
        all_maxs = [
            max(X.max() for X in X_flats),
            max(Y.max() for Y in Y_flats),
            max(Z.max() for Z in Z_flats)
        ]
        
        # Calculate the largest range for uniform scaling
        max_range = max(maxi - mini for mini, maxi in zip(all_mins, all_maxs)) / 2.0
        mid_points = [(mini + maxi) / 2.0 for mini, maxi in zip(all_mins, all_maxs)]
        
        # Set limits for uniform aspect ratio
        ax.set_xlim(mid_points[0] - max_range, mid_points[0] + max_range)
        ax.set_ylim(mid_points[1] - max_range, mid_points[1] + max_range)
        ax.set_zlim(mid_points[2] - max_range, mid_points[2] + max_range)
        
        # Set labels and force uniform scaling
        ax.set_title(f'{self.num_blades}-Blade Propeller Surfaces', fontsize=14)
        ax.set_xlabel('Chordwise Direction (X)', fontsize=12)
        ax.set_ylabel('Spanwise Direction (Y)', fontsize=12)
        ax.set_zlabel('Vertical Direction (Z)', fontsize=12)
        ax.set_box_aspect([1, 1, 1])
        ax.view_init(elev=90, azim=90)
        
        plt.show()

# # Example usage:
# if __name__ == "__main__":
#     propeller = PropellerGeometry(
#         airfoil_distribution_file='DJI9443_airfoils.csv',
#         chorddist_file='DJI9443_chorddist.csv',
#         pitchdist_file='DJI9443_pitchdist.csv',
#         sweepdist_file='DJI9443_sweepdist.csv',
#         heightdist_file='DJI9443_heightdist.csv',
#         R_tip=0.12,  # Tip radius in meters
#         R_hub=0.0064,  # Hub radius in meters
#         num_blades=2  # Number of blades
#     )

#     # Generate blade geometry
#     X, Y, Z, X2, Y2, Z2 = propeller.generate_blade_geometry()

#     # Create blade surfaces
#     X1_surf, Y1_surf, Z1_surf = propeller.create_blade_surface(X, Y, Z)
#     X2_surf, Y2_surf, Z2_surf = propeller.create_blade_surface(X2, Y2, Z2)

#     # Save surfaces to MATLAB file
#     propeller.save_blade_surfaces_to_mat('blade_surfaces.mat', X1_surf, Y1_surf, Z1_surf, X2_surf, Y2_surf, Z2_surf)

#     # Generate and plot flat blade surfaces
#     X_flats, Y_flats, Z_flats = propeller.generate_flat_blade_surfaces(span_resolution=15, chord_resolution=8)
#     propeller.plot_flat_blades_surface(X_flats, Y_flats, Z_flats)


#     # Get specific coefficients at r/R = 0.7 and alpha = 5 degrees
#     coefficients = propeller.get_polar_data(0.92, alpha=19)
#     print(f"At r/R = 0.7, alpha = 5°:")
    
#     # Fixed formatting for coefficient display
#     cl_str = f"{coefficients['cl']:.4f}" if coefficients['cl'] is not None else "N/A"
#     cd_str = f"{coefficients['cd']:.4f}" if coefficients['cd'] is not None else "N/A"
#     cm_str = f"{coefficients['cm']:.4f}" if coefficients['cm'] is not None else "N/A"
    
#     print(f"Cl = {cl_str}")
#     print(f"Cd = {cd_str}")
#     print(f"Cm = {cm_str}")

#     # Plot polar data for different r/R stations
#     r_R_stations = [0.2, 0.5, 0.8]
#     plt.figure(figsize=(12, 8))

#     # Create subplots for Cl, Cd, and Cl/Cd
#     fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
#     for r_R in r_R_stations:
#         try:
#             polar = propeller.get_polar_data(r_R)
#             if 'Cl' in polar.columns and 'Alpha' in polar.columns:
#                 ax1.plot(polar['Alpha'], polar['Cl'], label=f'r/R = {r_R:.2f}')
#             if 'Cd' in polar.columns and 'Alpha' in polar.columns:
#                 ax2.plot(polar['Alpha'], polar['Cd'], label=f'r/R = {r_R:.2f}')
#             if 'Cd' in polar.columns and 'Cl' in polar.columns:
#                 ax3.plot(polar['Cd'], polar['Cl'], label=f'r/R = {r_R:.2f}')
#         except Exception as e:
#             print(f"Could not plot polar data for r/R = {r_R}: {e}")

#     ax1.set_xlabel('Alpha (deg)')
#     ax1.set_ylabel('Lift Coefficient (Cl)')
#     ax1.set_title('Lift Curve')
#     ax1.grid(True)
#     ax1.legend()

#     ax2.set_xlabel('Alpha (deg)')
#     ax2.set_ylabel('Drag Coefficient (Cd)')
#     ax2.set_title('Drag Curve')
#     ax2.grid(True)
#     ax2.legend()

#     ax3.set_xlabel('Drag Coefficient (Cd)')
#     ax3.set_ylabel('Lift Coefficient (Cl)')
#     ax3.set_title('Drag Polar')
#     ax3.grid(True)
#     ax3.legend()

#     plt.tight_layout()
#     plt.show()