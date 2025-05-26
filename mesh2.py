import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from propeller import PropellerGeometry


class PropellerMesh:
    def __init__(self, propeller_geometry, arm_length, com=(0, 0, 0)):
        """
        Initialize the PropellerMesh system with propeller geometry and hub configuration.

        Args:
            propeller_geometry (PropellerGeometry): Instance of the PropellerGeometry class.
            arm_length (float): Length of the quadcopter's arms.
            com (tuple): Center of mass (x, y, z).
        """
        self.propeller_geometry = propeller_geometry
        self.import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.io import savemat
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PropellerMesh:
    def __init__(self, propeller_geometry, arm_length, com=(0, 0, 0), fuselage=None):

        self.propeller_geometry = propeller_geometry
        self.arm_length = arm_length
        self.com = np.array(com)
        self.fuselage = fuselage
        
        # Define alpha range for polar data
        self.alpha_range = np.linspace(-12, 20, 40)  # From -10 to 25 degrees in 1-degree steps

    def _generate_hub_points(self):
        """Generate positions for the propeller hubs relative to the COM."""
        return np.array([
            [self.arm_length, self.arm_length, 0],   # Front-right
            # [self.arm_length, -self.arm_length, 0],  # Front-left
            # [-self.arm_length, self.arm_length, 0],  # Back-right
            # [-self.arm_length, -self.arm_length, 0]  # Back-left
        ]) + self.com
    
        # R = 2.809 
        # return np.array([
        #     [self.arm_length * R, self.arm_length * R, 0.25 * R],   # Front-right
        #     # [self.arm_length * R, -self.arm_length * R,  0.25 * R],  # Front-left
        #     # [-self.arm_length * R, self.arm_length * R,  0.6 * R],  # Back-right
        #     # [-self.arm_length * R, -self.arm_length * R,  0.6 * R]  # Back-left
        # ]) + self.com

    def _compute_blade_mesh(self, X_flat, Y_flat, Z_flat):
        panels = {}
        control_points = {}
        vortex_rings = {}
        normals = {}
        gamma = {}
        tangential_vectors = {}
        twist = {}
        polar_data = {}  # New dictionary to store polar data for each panel

        # Shape info for looping
        spanwise_points, chordwise_points = X_flat.shape

        # Loop through spanwise and chordwise sections with interchanged roles
        for i in range(chordwise_points - 1):  # Now 'i' represents chordwise direction
            for j in range(spanwise_points - 1):  # Now 'j' represents spanwise direction
                # Corrected indexing logic
                leading_j = j  # Leading edge along span (j-direction)
                trailing_j = j + 1  # Trailing edge along span (j-direction)
                top_i = i  # Top edge along chord (i-direction)
                bottom_i = i + 1  # Bottom edge along chord (i-direction)

                # Panel corners
                panel_corners = [
                    [X_flat[leading_j, top_i], Y_flat[leading_j, top_i], Z_flat[leading_j, top_i]],  # Top-left
                    [X_flat[leading_j, bottom_i], Y_flat[leading_j, bottom_i], Z_flat[leading_j, bottom_i]],  # Top-right
                    [X_flat[trailing_j, bottom_i], Y_flat[trailing_j, bottom_i], Z_flat[trailing_j, bottom_i]],  # Bottom-right
                    [X_flat[trailing_j, top_i], Y_flat[trailing_j, top_i], Z_flat[trailing_j, top_i]]  # Bottom-left
                ]

                # Panel key
                panel_index = (i, j)  # Interchanged indexing
                panels[panel_index] = panel_corners

                # Control point
                span_mid = (np.array(panel_corners[0]) + np.array(panel_corners[3])) / 2  # Midpoint along span
                chord_mid = (np.array(panel_corners[1]) + np.array(panel_corners[2])) / 2  # Midpoint along chord
                control_point = span_mid + 0.75 * (chord_mid - span_mid)
                control_points[panel_index] = control_point

                if self.propeller_geometry.R_tip > 0:
                    r_R = np.linalg.norm(control_point[1]) / self.propeller_geometry.R_tip
                    
                    twist_angle = self.propeller_geometry.pitch_spline(r_R)
                    twist[panel_index] = twist_angle

                    panel_polar_data = {
                        'r/R': r_R,
                        'alpha': [],
                        'cl': [],
                        'cd': [],
                        'cm': []
                    }
             
                    for alpha in self.alpha_range:
                        try:
                            coeffs = self.propeller_geometry._get_interpolated_coefficients(r_R, alpha)
                            panel_polar_data['alpha'].append(alpha)
                            panel_polar_data['cl'].append(coeffs['cl'] if coeffs['cl'] is not None else np.nan)
                            panel_polar_data['cd'].append(coeffs['cd'] if coeffs['cd'] is not None else np.nan)
                            panel_polar_data['cm'].append(coeffs['cm'] if coeffs['cm'] is not None else np.nan)
                        except Exception as e:
                            # If interpolation fails, add NaN values
                            panel_polar_data['alpha'].append(alpha)
                            panel_polar_data['cl'].append(np.nan)
                            panel_polar_data['cd'].append(np.nan)
                            panel_polar_data['cm'].append(np.nan)
                    
                    # Store the polar data for this panel
                    polar_data[panel_index] = panel_polar_data
                else:
                    twist[panel_index] = 0.0  # Default if R_tip is not properly set
                    polar_data[panel_index] = None  # No polar data available

                # Vortex ring
                leading_edge_start = np.array(panel_corners[0]) + 0.25 * (np.array(panel_corners[1]) - np.array(panel_corners[0]))
                leading_edge_end = np.array(panel_corners[3]) + 0.25 * (np.array(panel_corners[2]) - np.array(panel_corners[3]))
                trailing_edge_start = np.array(panel_corners[1]) + 0.25 * (np.array(panel_corners[1]) - np.array(panel_corners[0]))
                trailing_edge_end = np.array(panel_corners[2]) + 0.25 * (np.array(panel_corners[2]) - np.array(panel_corners[3]))

                vortex_ring = {
                    'Vertices': [
                        leading_edge_start,  # Start 1/4 behind the leading edge, top
                        trailing_edge_start,  # Start 1/4 behind the trailing edge, top
                        trailing_edge_end,  # End 1/4 behind the trailing edge, bottom
                        leading_edge_end  # End 1/4 behind the leading edge, bottom
                    ],
                    'Edge Vectors': [
                        trailing_edge_start - leading_edge_start,  # Top edge
                        trailing_edge_end - trailing_edge_start,  # Trailing edge
                        leading_edge_end - trailing_edge_end,  # Bottom edge
                        leading_edge_start - leading_edge_end   # Leading edge
                    ]
                }
                vortex_rings[panel_index] = vortex_ring  # Use (i, j) as the key

                # Normal vector
                span_vector = np.array(panel_corners[2]) - np.array(panel_corners[0])  # Leading edge span vector
                chord_vector = np.array(panel_corners[3]) - np.array(panel_corners[1])  # Leading edge chord vector
                normal = np.cross(span_vector, chord_vector)
                normals[panel_index] = normal / np.linalg.norm(normal)  # Normalize

                # Initialize gamma
                gamma[panel_index] = 1.0

                # Tangential vectors
                # Compute average spanwise tangent (tangential_i)
                span_vector_1 = np.array(panel_corners[3]) - np.array(panel_corners[0])  # Spanwise edge 1
                span_vector_2 = np.array(panel_corners[2]) - np.array(panel_corners[1])  # Spanwise edge 2
                span_vector = 0.5 * (span_vector_1 + span_vector_2)  # Average
                tangential_i = span_vector # Normalize

                # Compute average chordwise tangent (tangential_j)
                chord_vector_1 = np.array(panel_corners[1]) - np.array(panel_corners[0])  # Chordwise edge 1
                chord_vector_2 = np.array(panel_corners[2]) - np.array(panel_corners[3]) # Chordwise edge 2
                chord_vector = 0.5 * (chord_vector_1 + chord_vector_2)  # Average
                tangential_j = chord_vector # Normalize

                tangential_vectors[panel_index] = {
                    'Tangential i': tangential_i,
                    'Tangential j': tangential_j
                }

        # Return the dictionary for the blade mesh with added polar data
        return {
            'Panels': panels,
            'Control Points': control_points,
            'Vortex Rings': vortex_rings,   
            'Normals': normals,
            'Gamma': gamma,
            'Tangential Vectors': tangential_vectors,
            'Twist': twist,
            'Polar Data': polar_data  # Add the polar data information to the returned dictionary
        }

    def generate_mesh(self):
        """
        Generate the mesh for a single propeller with all blades, including polar data.

        Returns:
            dict: The dictionary containing the mesh for the single propeller.
        """
        # Generate blade surfaces
        X_flats, Y_flats, Z_flats = self.propeller_geometry.generate_flat_blade_surfaces()

        # Create dictionaries for all blades
        blades = {}
        for i in range(self.propeller_geometry.num_blades):
            blade_key = f'Blade_{i+1}'
            blades[blade_key] = self._compute_blade_mesh(X_flats[i], Y_flats[i], Z_flats[i])

        # Return the combined dictionary
        return {
            'Blades': blades,
        }         
    
    def generate_quad_propeller_mesh(self):
        """
        Generate the quadcopter mesh by creating individual propeller meshes and translating them to the
        respective hub positions. Use different methods for calculating normals for Propellers 1 & 4 and Propellers 2 & 3.
        Also include twist information and polar data at each control point.

        Returns:
            dict: A dictionary containing the mesh data for all four propellers.
        """
        # Get hub positions
        hub_positions = self._generate_hub_points()

        # Initialize quadcopter mesh
        quad_mesh = {}

        # Generate the mesh for a single propeller
        single_propeller_mesh = self.generate_mesh()

        # Loop over hub positions and create translated propellers
        for idx, hub_position in enumerate(hub_positions):
            propeller_key = f'Propeller_{idx + 1}'  # Unique key for each propeller

            # Translate blades
            translated_blades = {}
            for blade_key, blade_data in single_propeller_mesh['Blades'].items():
                # Translate Panels
                translated_panels = {}
                translated_normals = {}
                translated_tangential_vectors = {}
                translated_twist = {}
                translated_polar_data = {}  # Store translated polar data

                for (i, j), panel in blade_data['Panels'].items():
                    # Transform panels based on the propeller index
                    if idx in [0, 3]:  # Propellers 1 and 4 (CW)
                        transformed_panel = [np.array(vertex) + hub_position for vertex in panel]
                    else:  # Propellers 2 and 3 (CCW)
                        transformed_panel = [
                            np.array([-vertex[0], vertex[1], vertex[2]]) + hub_position for vertex in panel
                        ]
                    translated_panels[(i, j)] = transformed_panel

                    # Calculate normals based on the propeller index
                    if idx in [0, 3]:  # Propellers 1 and 4: Standard normal calculation
                        span_vector = transformed_panel[2] - transformed_panel[0]  # Span vector
                        chord_vector = transformed_panel[3] - transformed_panel[1]  # Chord vector
                        normal = np.cross(span_vector, chord_vector)
                    else:  # Propellers 2 and 3: Custom normal calculation
                        span_vector = transformed_panel[2] - transformed_panel[0]  # Span vector
                        chord_vector = transformed_panel[3] - transformed_panel[1]  # Chord vector
                        normal = np.cross(span_vector, chord_vector)

                    # Normalize and ensure direction consistency
                    normal = normal / np.linalg.norm(normal)
                    translated_normals[(i, j)] = normal

                    # Transfer the twist information directly
                    if 'Twist' in blade_data and (i, j) in blade_data['Twist']:
                        translated_twist[(i, j)] = blade_data['Twist'][(i, j)]
                    else:
                        # If twist data is missing, try to calculate it based on r/R
                        if self.propeller_geometry.R_tip > 0:
                            # Use the translated control point's Y value to determine r/R
                            cp = blade_data['Control Points'][(i, j)]
                            if idx in [0, 3]:
                                translated_cp = np.array(cp) + hub_position
                            else:
                                translated_cp = np.array([-cp[0], cp[1], cp[2]]) + hub_position
                            
                            r_R = np.linalg.norm(translated_cp[1] - hub_position[1]) / self.propeller_geometry.R_tip
                            twist_angle = self.propeller_geometry.pitch_spline(r_R)
                            translated_twist[(i, j)] = twist_angle
                        else:
                            translated_twist[(i, j)] = 0.0  # Default if R_tip is not properly set
                            
                    # Transfer polar data
                    if 'Polar Data' in blade_data and (i, j) in blade_data['Polar Data']:
                        translated_polar_data[(i, j)] = blade_data['Polar Data'][(i, j)]
                    else:
                        # If polar data is missing, set to None
                        translated_polar_data[(i, j)] = None

                    # Calculate tangential vectors based on the propeller index
                    span_vector_1 = transformed_panel[3] - transformed_panel[0]  # Chordwise edge 1
                    span_vector_2 = transformed_panel[2] - transformed_panel[1]  # Chordwise edge 2
                    avg_span_vector = 0.5 * (span_vector_1 + span_vector_2)

                    chord_vector_1 = transformed_panel[1] - transformed_panel[0]  # Chordwise edge 1
                    chord_vector_2 = transformed_panel[2] - transformed_panel[3]  # Chordwise edge 2
                    avg_chord_vector = 0.5 * (chord_vector_1 + chord_vector_2)

                    tangential_i = avg_span_vector
                    tangential_j = avg_chord_vector 
                
                    translated_tangential_vectors[(i, j)] = {
                        'Tangential i': tangential_i,
                        'Tangential j': tangential_j
                    }

                # Translate Control Points
                translated_control_points = {
                    (i, j): (
                        np.array(cp) + hub_position if idx in [0, 3]  # Propellers 1 and 4
                        else np.array([-cp[0], cp[1], cp[2]]) + hub_position  # Propellers 2 and 3 mirrored
                            
                    )   
                    for (i, j), cp in blade_data['Control Points'].items()       
                }
        
                # Translate Vortex Rings (only vertices; edge vectors remain the same)
                translated_vortex_rings = {
                    (i, j): {
                        'Vertices': [
                            np.array(vertex) + hub_position if idx in [0, 3]  # Propellers 1 and 4
                            else np.array([-vertex[0], vertex[1], vertex[2]]) + hub_position  # Propellers 2 and 3 mirrored
                            for vertex in vortex_data['Vertices']
                        ]
                
                    }
                    for (i, j), vortex_data in blade_data['Vortex Rings'].items()
                }

                # Add all translated and recalculated components to the blade
                translated_blades[blade_key] = {
                    'Panels': translated_panels,
                    'Normals': translated_normals,
                    'Tangential Vectors': translated_tangential_vectors,
                    'Control Points': translated_control_points,
                    'Vortex Rings': translated_vortex_rings,
                    'Gamma': blade_data['Gamma'],
                    'Twist': translated_twist,
                    'Polar Data': translated_polar_data  # Add polar data to the translated blade
                }
                
            # Add the translated blades to the quadcopter mesh
            quad_mesh[propeller_key] = {
                'Blades': translated_blades,
                'Hub Position': hub_position
            } 
        return quad_mesh

    def plot_propeller_mesh_with_highlight(self, quad_propeller_mesh, propeller_key='Propeller_1', 
                                       blade_key=None, highlight_panel=None, show_twist=True, show_polar=True):
        """
        Plot the entire propeller mesh with all blades and option to highlight a specific panel.
        Also plots the fuselage if provided, and displays twist information and polar data for highlighted panels.
        
        Args:
            quad_propeller_mesh: The full mesh
            propeller_key: Which propeller to visualize
            blade_key: Which blade to highlight (optional)
            highlight_panel: Tuple of (i,j) for panel to highlight, or None
            show_twist: Whether to display twist information (default: True)
            show_polar: Whether to display polar data (default: True)
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Plot all blades in the propeller
        propeller_data = quad_propeller_mesh[propeller_key]
        
        # Generate colors for each blade
        num_blades = len(propeller_data['Blades'])
        blade_colors = plt.cm.rainbow(np.linspace(0, 1, num_blades))
        
        # Plot each blade
        for idx, (current_blade_key, blade_data) in enumerate(propeller_data['Blades'].items()):
            # Set alpha based on whether this is the highlighted blade
            alpha = 1.0 if current_blade_key == blade_key else 0.3
            color = blade_colors[idx]
            
            # Plot all panels
            for panel_idx, panel in blade_data['Panels'].items():
                panel_array = np.array(panel)
                
                # If this is the highlighted panel
                if current_blade_key == blade_key and panel_idx == highlight_panel:
                    # Plot highlighted panel features
                    for i in range(4):
                        start = panel[i]
                        end = panel[(i+1)%4]
                        ax.plot([start[0], end[0]], 
                            [start[1], end[1]], 
                           [start[2], end[2]], 
                           'b-', linewidth=2)
                
                    # Plot panel vertices
                    for i, vertex in enumerate(panel):
                        ax.scatter(vertex[0], vertex[1], vertex[2], color='blue', s=100)
                        ax.text(vertex[0], vertex[1], vertex[2], f'P{i}', fontsize=12)
                
                    # Plot vortex ring features
                    vortex_ring = blade_data['Vortex Rings'][panel_idx]
                    vortex_vertices = vortex_ring['Vertices']
                    for i, vertex in enumerate(vortex_vertices):
                        ax.scatter(vertex[0], vertex[1], vertex[2], color='red', s=100)
                        ax.text(vertex[0], vertex[1], vertex[2], f'V{i}', fontsize=12)
                
                    # Connect vortex ring points
                    for i in range(4):
                        start = vortex_vertices[i]
                        end = vortex_vertices[(i+1)%4]
                        ax.plot([start[0], end[0]], 
                               [start[1], end[1]], 
                               [start[2], end[2]], 
                               'r--', linewidth=2)
                
                    # Plot control point and normal vector
                    control_point = blade_data['Control Points'][panel_idx]
                    normal = blade_data['Normals'][panel_idx]
                
                    ax.scatter(control_point[0], control_point[1], control_point[2], 
                              color='green', s=100)
                    ax.text(control_point[0], control_point[1], control_point[2], 
                           'CP', fontsize=12)
                
                    # Plot normal vector
                    scale = 0.05
                    ax.quiver(control_point[0], control_point[1], control_point[2],
                             normal[0], normal[1], normal[2],
                             length=scale, color='k', arrow_length_ratio=0.2)
                
                    # Add twist information if available and requested
                    if show_twist and 'Twist' in blade_data and panel_idx in blade_data['Twist']:
                        twist_angle = blade_data['Twist'][panel_idx]
                        # Position the twist text a bit above the control point
                        text_pos = control_point + np.array([0, 0, 0.02])
                        ax.text(text_pos[0], text_pos[1], text_pos[2], 
                               f'Twist: {twist_angle:.2f}°', 
                               fontsize=12, color='purple',
                               bbox=dict(facecolor='white', alpha=0.7, edgecolor='purple'))
                        
                    # Display polar data summary if requested
                    if show_polar and 'Polar Data' in blade_data and panel_idx in blade_data['Polar Data']:
                        polar = blade_data['Polar Data'][panel_idx]
                        if polar is not None:
                            # Get r/R value
                            r_R = polar['r/R']
                            
                            # Find valid indices (non-NaN values)
                            valid_cl = ~np.isnan(polar['cl'])
                            valid_cd = ~np.isnan(polar['cd'])
                            
                            # Calculate mean values for valid data
                            mean_cl = np.nanmean(polar['cl']) if np.any(valid_cl) else np.nan
                            mean_cd = np.nanmean(polar['cd']) if np.any(valid_cd) else np.nan
                            
                            # Get the lift coefficient at zero angle of attack (if available)
                            cl0 = np.nan
                            zero_alpha_idx = np.argmin(np.abs(np.array(polar['alpha'])))
                            if zero_alpha_idx < len(polar['cl']) and not np.isnan(polar['cl'][zero_alpha_idx]):
                                cl0 = polar['cl'][zero_alpha_idx]
                            
                            # Position the polar data text below the twist information
                            text_pos = control_point + np.array([0, 0, -0.02])
                            ax.text(text_pos[0], text_pos[1], text_pos[2], 
                                  f'r/R: {r_R:.2f}\nMean Cl: {mean_cl:.3f}\nMean Cd: {mean_cd:.3f}\nCl(α=0): {cl0:.3f}', 
                                  fontsize=10, color='darkgreen',
                                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='darkgreen'))
                else:
                    # Plot regular panel edges
                    for i in range(4):
                        start = panel[i]
                        end = panel[(i+1)%4]
                        ax.plot([start[0], end[0]], 
                               [start[1], end[1]], 
                               [start[2], end[2]], 
                               color=color, alpha=alpha)
                
                    # Plot control point
                    cp = blade_data['Control Points'][panel_idx]
                    ax.scatter(cp[0], cp[1], cp[2], color=color, alpha=alpha, s=30)
    
        # Add legend
        if self.fuselage:
            ax.scatter([], [], [], c='gray', alpha=0.5, s=100, label='Fuselage')
        for i in range(num_blades):
            ax.scatter([], [], [], c=[blade_colors[i]], label=f'Blade {i+1}')
        
        if highlight_panel:
            ax.scatter([], [], [], c='blue', s=100, label='Panel Vertices (P0-P3)')
            ax.scatter([], [], [], c='red', s=100, label='Vortex Ring Points (V0-V3)')
            ax.scatter([], [], [], c='green', s=100, label='Control Point (CP)')
            ax.quiver([], [], [], [], [], [], color='k', label='Normal Vector')
            if show_twist:
                # Add an empty entry for the twist in the legend
                ax.plot([], [], color='purple', label='Twist Angle')
            if show_polar:
                # Add an empty entry for the polar data in the legend
                ax.plot([], [], color='darkgreen', label='Polar Data')
        
        ax.legend()
        
        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        title = f'Propeller Mesh ({propeller_key})'
        if blade_key and highlight_panel:
            title += f'\nHighlighted Panel: {highlight_panel} on {blade_key}'
        ax.set_title(title)
        
        # Set equal aspect ratio
        max_range = np.array([
            ax.get_xlim3d()[1] - ax.get_xlim3d()[0],
            ax.get_ylim3d()[1] - ax.get_ylim3d()[0],
            ax.get_zlim3d()[1] - ax.get_zlim3d()[0]
        ]).max() / 2.0
        
        mid_x = (ax.get_xlim3d()[1] + ax.get_xlim3d()[0]) / 2.0
        mid_y = (ax.get_ylim3d()[1] + ax.get_ylim3d()[0]) / 2.0
        mid_z = (ax.get_zlim3d()[1] + ax.get_zlim3d()[0]) / 2.0
        ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
        ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
        ax.set_zlim3d([mid_z - max_range, mid_z + max_range])
        
        plt.show()
        
    def plot_polar_data(self, quad_propeller_mesh, propeller_key, blade_key, panel_idx):
        """
        Plot the polar (Cl, Cd, Cm) data for a specific panel.
        
        Args:
            quad_propeller_mesh: The full quadcopter mesh
            propeller_key: Which propeller to use
            blade_key: Which blade to use
            panel_idx: Tuple of (i,j) for panel to visualize
        """
        # Get the polar data
        try:
            blade_data = quad_propeller_mesh[propeller_key]['Blades'][blade_key]
            polar = blade_data['Polar Data'][panel_idx]
            
            if polar is None:
                print(f"No polar data available for {propeller_key}, {blade_key}, panel {panel_idx}")
                return
                
            # Extract data
            alpha = polar['alpha']
            cl = polar['cl']
            cd = polar['cd']
            cm = polar['cm']
            r_R = polar['r/R']
            
            # Create the plot
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
            
            # Cl vs alpha
            ax1.plot(alpha, cl, 'b-o', label='Cl')
            ax1.set_xlabel('Angle of Attack (deg)')
            ax1.set_ylabel('Lift Coefficient (Cl)')
            ax1.set_title(f'Lift Curve (r/R = {r_R:.2f})')
            ax1.grid(True)
            ax1.legend()
            
            # Cd vs alpha
            ax2.plot(alpha, cd, 'r-o', label='Cd')
            ax2.set_xlabel('Angle of Attack (deg)')
            ax2.set_ylabel('Drag Coefficient (Cd)')
            ax2.set_title(f'Drag Curve (r/R = {r_R:.2f})')
            ax2.grid(True)
            ax2.legend()
            
            # Cl vs Cd (drag polar)
            ax3.plot(cd, cl, 'g-o', label='Cl vs Cd')
            ax3.set_xlabel('Drag Coefficient (Cd)')
            ax3.set_ylabel('Lift Coefficient (Cl)')
            ax3.set_title(f'Drag Polar (r/R = {r_R:.2f})')
            ax3.grid(True)
            ax3.legend()
            
            # Add panel information to the figure
            plt.suptitle(f'Aerodynamic Coefficients for {propeller_key}, {blade_key}, Panel {panel_idx}', fontsize=16)
            
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting polar data: {e}")
            
    def get_panel_aerodynamic_coefficients(self, quad_propeller_mesh, propeller_key, blade_key, panel_idx, alpha):
        """
        Get aerodynamic coefficients (Cl, Cd, Cm) for a specific panel at a given angle of attack.
        
        Args:
            quad_propeller_mesh: The full quadcopter mesh
            propeller_key: Which propeller to use
            blade_key: Which blade to use
            panel_idx: Tuple of (i,j) for panel
            alpha: Angle of attack in degrees
            
        Returns:
            dict: Dictionary containing Cl, Cd, Cm values (or None if data unavailable)
        """
        try:
            blade_data = quad_propeller_mesh[propeller_key]['Blades'][blade_key]
            polar = blade_data['Polar Data'][panel_idx]
            
            if polar is None:
                return {'cl': None, 'cd': None, 'cm': None, 'r/R': None}
                
            # Get the closest alpha value in our data
            alpha_idx = np.argmin(np.abs(np.array(polar['alpha']) - alpha))
            
            # Extract coefficients
            cl = polar['cl'][alpha_idx] if alpha_idx < len(polar['cl']) else None
            cd = polar['cd'][alpha_idx] if alpha_idx < len(polar['cd']) else None
            cm = polar['cm'][alpha_idx] if alpha_idx < len(polar['cm']) else None
            
            return {
                'cl': cl,
                'cd': cd,
                'cm': cm,
                'r/R': polar['r/R'],
                'actual_alpha': polar['alpha'][alpha_idx]
            }
        except Exception as e:
            print(f"Error getting coefficients: {e}")
            return {'cl': None, 'cd': None, 'cm': None, 'r/R': None}
            
    def visualize_coefficients_distribution(self, quad_propeller_mesh, propeller_key, blade_key, alpha, coefficient='cl'):
        """
        Visualize the distribution of a specific aerodynamic coefficient across the blade.
        
        Args:
            quad_propeller_mesh: The full quadcopter mesh
            propeller_key: Which propeller to use
            blade_key: Which blade to use
            alpha: Angle of attack in degrees
            coefficient: Which coefficient to visualize ('cl', 'cd', or 'cm')
        """
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        blade_data = quad_propeller_mesh[propeller_key]['Blades'][blade_key]
        
        # Collect all control points and coefficient values
        points = []
        values = []
        r_Rs = []
        
        for panel_idx, control_point in blade_data['Control Points'].items():
            coeff_data = self.get_panel_aerodynamic_coefficients(
                quad_propeller_mesh, propeller_key, blade_key, panel_idx, alpha)
            
            if coeff_data[coefficient] is not None and not np.isnan(coeff_data[coefficient]):
                points.append(control_point)
                values.append(coeff_data[coefficient])
                r_Rs.append(coeff_data['r/R'])
        
        if not points:
            print(f"No valid {coefficient} data for α={alpha}° on {blade_key} of {propeller_key}")
            return
            
        # Convert to numpy arrays
        points = np.array(points)
        values = np.array(values)
        
        # Create color map
        norm = plt.Normalize(np.min(values), np.max(values))
        cmap = plt.cm.plasma
        colors = cmap(norm(values))
        
        # Plot the control points colored by coefficient value
        sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                        c=values, cmap=cmap, s=100, alpha=0.8)
        
        # Add colorbar
        cbar = plt.colorbar(sc)
        cbar.set_label(f"{coefficient.upper()} at α={alpha}°")
        
        # Plot the blade outline
        for panel_idx, panel in blade_data['Panels'].items():
            panel_array = np.array(panel)
            for i in range(4):
                start = panel[i]
                end = panel[(i+1)%4]
                ax.plot([start[0], end[0]], 
                       [start[1], end[1]], 
                       [start[2], end[2]], 
                       color='gray', alpha=0.3)
        
        # Plot text labels showing r/R values at a few points
        num_labels = min(5, len(r_Rs))
        label_indices = np.linspace(0, len(r_Rs)-1, num_labels, dtype=int)
        
        for idx in label_indices:
            ax.text(points[idx, 0], points[idx, 1], points[idx, 2],
                   f"r/R={r_Rs[idx]:.2f}", fontsize=10)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        title = f'{coefficient.upper()} Distribution on {blade_key} of {propeller_key} at α={alpha}°'
        ax.set_title(title)
        
        # Set equal aspect ratio
        max_range = np.array([
            ax.get_xlim3d()[1] - ax.get_xlim3d()[0],
            ax.get_ylim3d()[1] - ax.get_ylim3d()[0],
            ax.get_zlim3d()[1] - ax.get_zlim3d()[0]
        ]).max() / 2.0
        
        mid_x = (ax.get_xlim3d()[1] + ax.get_xlim3d()[0]) / 2.0
        mid_y = (ax.get_ylim3d()[1] + ax.get_ylim3d()[0]) / 2.0
        mid_z = (ax.get_zlim3d()[1] + ax.get_zlim3d()[0]) / 2.0
        ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
        ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
        ax.set_zlim3d([mid_z - max_range, mid_z + max_range])
        
        plt.tight_layout()
        plt.show()
        
    def plot_coefficient_vs_r_R(self, quad_propeller_mesh, propeller_key, blade_key, alpha, coefficient='cl'):
        """
        Plot how a specific aerodynamic coefficient varies with radial position (r/R).
        
        Args:
            quad_propeller_mesh: The full quadcopter mesh
            propeller_key: Which propeller to use
            blade_key: Which blade to use
            alpha: Angle of attack in degrees
            coefficient: Which coefficient to visualize ('cl', 'cd', or 'cm')
        """
        blade_data = quad_propeller_mesh[propeller_key]['Blades'][blade_key]
        
        # Collect r/R values and coefficient values
        r_Rs = []
        values = []
        
        for panel_idx, _ in blade_data['Control Points'].items():
            coeff_data = self.get_panel_aerodynamic_coefficients(
                quad_propeller_mesh, propeller_key, blade_key, panel_idx, alpha)
            
            if coeff_data[coefficient] is not None and not np.isnan(coeff_data[coefficient]):
                r_Rs.append(coeff_data['r/R'])
                values.append(coeff_data[coefficient])
        
        if not r_Rs:
            print(f"No valid {coefficient} data for α={alpha}° on {blade_key} of {propeller_key}")
            return
            
        # Create sorted arrays based on r/R
        sorted_indices = np.argsort(r_Rs)
        r_Rs = np.array(r_Rs)[sorted_indices]
        values = np.array(values)[sorted_indices]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(r_Rs, values, 'o-', linewidth=2)
        plt.xlabel('Radial Position (r/R)')
        plt.ylabel(f'{coefficient.upper()}')
        plt.title(f'{coefficient.upper()} vs Radial Position at α={alpha}° for {blade_key} of {propeller_key}')
        plt.grid(True)
        
        # Add a trendline
        if len(r_Rs) > 2:
            z = np.polyfit(r_Rs, values, 3)
            p = np.poly1d(z)
            r_Rs_fine = np.linspace(min(r_Rs), max(r_Rs), 100)
            plt.plot(r_Rs_fine, p(r_Rs_fine), 'r--', linewidth=1, label='Trend')
            
        plt.legend()
        plt.tight_layout()
        plt.show()
        

# # Create the propeller geometry
#     propeller_geometry = PropellerGeometry(
#         airfoil_distribution_file='DJI9443_airfoils.csv',
#         chorddist_file='DJI9443_chorddist.csv',
#         pitchdist_file='DJI9443_pitchdist.csv',
#         sweepdist_file='DJI9443_sweepdist.csv',
#         heightdist_file='DJI9443_heightdist.csv',
#         R_tip=0.12,  # Tip radius in meters
#         R_hub=0.0064,  # Hub radius in meters
#         num_blades=3  # Number of blades
#     )
    
#     # Create the propeller mesh system
#     propeller_mesh_system = PropellerMesh(
#         propeller_geometry, 
#         arm_length=0.5,
#         com=(0, 0, 0)
#     )
    
#     # Generate the quadcopter mesh with polar data
#     quad_mesh = propeller_mesh_system.generate_quad_propeller_mesh()
    
#     # Example 1: Print available panel indices for Blade_1 of Propeller_1
#     propeller_key = 'Propeller_1'
#     blade_key = 'Blade_2'
#     blade_data = quad_mesh[propeller_key]['Blades'][blade_key]
    
#     print("Available panel indices in Blade_1:")
#     panel_indices = list(blade_data['Panels'].keys())
#     print(panel_indices[:50])  # Print first 5 panels
    
#     # Example 2: Access twist and polar data for a specific panel
#     panel_idx = panel_indices[0]  # Pick the first panel
    
#     # Get twist angle
#     if panel_idx in blade_data['Twist']:
#         twist_angle = blade_data['Twist'][panel_idx]
#         print(f"\nTwist at panel {panel_idx}: {twist_angle:.2f} degrees")
    
#     # Get polar data
#     if panel_idx in blade_data['Polar Data']:
#         polar = blade_data['Polar Data'][panel_idx]
#         if polar is not None:
#             r_R = polar['r/R']
#             print(f"Panel at r/R = {r_R:.2f}")
            
#             # Check if we have valid data at α=0°
#             alpha_idx = np.argmin(np.abs(np.array(polar['alpha'])))
#             if alpha_idx < len(polar['cl']):
#                 print(f"Coefficients at α≈{polar['alpha'][alpha_idx]:.1f}°:")
#                 print(f"  Cl = {polar['cl'][alpha_idx]:.4f}")
#                 print(f"  Cd = {polar['cd'][alpha_idx]:.4f}")
#                 print(f"  Cm = {polar['cm'][alpha_idx]:.4f}")
    
#     # Example 3: Plot a propeller mesh with highlighted panel
#     print("\nGenerating mesh visualization...")
#     propeller_mesh_system.plot_propeller_mesh_with_highlight(
#         quad_mesh,
#         propeller_key='Propeller_1',
#         blade_key='Blade_1',
#         highlight_panel=panel_idx,
#         show_twist=True,
#         show_polar=True
#     )
    
#     # Example 4: Plot polar data for a specific panel
#     print("\nGenerating polar data plots...")
#     propeller_mesh_system.plot_polar_data(
#         quad_mesh,
#         propeller_key='Propeller_1',
#         blade_key='Blade_1',
#         panel_idx=panel_idx
#     )
    
#     # Example 5: Get aerodynamic coefficients at a specific angle of attack
#     alpha = 5.0  # Angle of attack in degrees
#     coeffs = propeller_mesh_system.get_panel_aerodynamic_coefficients(
#         quad_mesh,
#         propeller_key='Propeller_1',
#         blade_key='Blade_1',
#         panel_idx=panel_idx,
#         alpha=alpha
#     )
    
    
#     # Example 6: Visualize coefficient distribution across the blade
#     print("\nGenerating coefficient distribution visualization...")
#     propeller_mesh_system.visualize_coefficients_distribution(
#         quad_mesh,
#         propeller_key='Propeller_1',
#         blade_key='Blade_1',
#         alpha=5.0,
#         coefficient='cl'  # Visualize Cl distribution
#     )
    
#     # Example 7: Plot coefficient vs radial position
#     print("\nGenerating coefficient vs r/R plot...")
#     propeller_mesh_system.plot_coefficient_vs_r_R(
#         quad_mesh,
#         propeller_key='Propeller_1',
#         blade_key='Blade_1',
#         alpha=5.0,
#         coefficient='cl'  # Plot Cl vs r/R
#     )
    
#     # Example 8: Export panel data to CSV
#     output_file = 'propeller_panel_data.csv'
#     print(f"\nExporting panel data to {output_file}...")
#     propeller_mesh_system.export_panel_data_to_csv(
#         quad_mesh,
#         propeller_key='Propeller_1',
#         blade_key='Blade_1',
#         filename=output_file
#     )
    
#     print("\nAll examples completed!")arm_length = arm_length
        self.com = np.array(com)

    def _generate_hub_points(self):
        """Generate positions for the propeller hubs relative to the COM."""
        return np.array([
            [self.arm_length, self.arm_length, 0],   # Front-right
            [self.arm_length, -self.arm_length, 0],  # Front-left
            [-self.arm_length, self.arm_length, 0],  # Back-right
            [-self.arm_length, -self.arm_length, 0]  # Back-left
        ]) + self.com

    def _compute_blade_mesh(self, X_flat, Y_flat, Z_flat):
        """
        Compute the mesh for a single blade with interchanged spanwise and chordwise indexing.

        Args:
            X_flat, Y_flat, Z_flat: Arrays representing the flat blade surface.

        Returns:
            dict: A dictionary containing the mesh data for the blade.
        """
        panels = {}
        control_points = {}
        vortex_rings = {}
        normals = {}
        gamma = {}
        tangential_vectors = {}

        # Shape info for looping
        spanwise_points, chordwise_points = X_flat.shape

        # Loop through spanwise and chordwise sections with interchanged roles
        for i in range(chordwise_points - 1):  # Now 'i' represents chordwise direction
            for j in range(spanwise_points - 1):  # Now 'j' represents spanwise direction
                # Corrected indexing logic
                leading_j = j  # Leading edge along span (j-direction)
                trailing_j = j + 1  # Trailing edge along span (j-direction)
                top_i = i  # Top edge along chord (i-direction)
                bottom_i = i + 1  # Bottom edge along chord (i-direction)

                # Panel corners
                panel_corners = [
                    [X_flat[leading_j, top_i], Y_flat[leading_j, top_i], Z_flat[leading_j, top_i]],  # Top-left
                    [X_flat[leading_j, bottom_i], Y_flat[leading_j, bottom_i], Z_flat[leading_j, bottom_i]],  # Top-right
                    [X_flat[trailing_j, bottom_i], Y_flat[trailing_j, bottom_i], Z_flat[trailing_j, bottom_i]],  # Bottom-right
                    [X_flat[trailing_j, top_i], Y_flat[trailing_j, top_i], Z_flat[trailing_j, top_i]]  # Bottom-left
                ]

                # Panel key
                panel_index = (i, j)  # Interchanged indexing
                panels[panel_index] = panel_corners

                # Control point
                span_mid = (np.array(panel_corners[0]) + np.array(panel_corners[3])) / 2  # Midpoint along span
                chord_mid = (np.array(panel_corners[1]) + np.array(panel_corners[2])) / 2  # Midpoint along chord
                control_point = span_mid + 0.75 * (chord_mid - span_mid)
                control_points[panel_index] = control_point

                # Vortex ring
                leading_edge_start = np.array(panel_corners[0]) + 0.25 * (np.array(panel_corners[1]) - np.array(panel_corners[0]))
                leading_edge_end = np.array(panel_corners[3]) + 0.25 * (np.array(panel_corners[2]) - np.array(panel_corners[3]))
                trailing_edge_start = np.array(panel_corners[1]) + 0.25 * (np.array(panel_corners[1]) - np.array(panel_corners[0]))
                trailing_edge_end = np.array(panel_corners[2]) + 0.25 * (np.array(panel_corners[2]) - np.array(panel_corners[3]))

                vortex_ring = {
                    'Vertices': [
                        leading_edge_start,  # Start 1/4 behind the leading edge, top
                        trailing_edge_start,  # Start 1/4 behind the trailing edge, top
                        trailing_edge_end,  # End 1/4 behind the trailing edge, bottom
                        leading_edge_end  # End 1/4 behind the leading edge, bottom
                    ],
                    'Edge Vectors': [
                        trailing_edge_start - leading_edge_start,  # Top edge
                        trailing_edge_end - trailing_edge_start,  # Trailing edge
                        leading_edge_end - trailing_edge_end,  # Bottom edge
                        leading_edge_start - leading_edge_end   # Leading edge
                    ]
                }
                vortex_rings[panel_index] = vortex_ring  # Use (j, i) as the key

                # Normal vector
                span_vector = np.array(panel_corners[2]) - np.array(panel_corners[0])  # Leading edge span vector
                chord_vector = np.array(panel_corners[3]) - np.array(panel_corners[1])  # Leading edge chord vector
                normal = np.cross(span_vector, chord_vector)
                normals[panel_index] = normal / np.linalg.norm(normal)  # Normalize

                # Initialize gamma
                gamma[panel_index] = 1.0

                # Tangential vectors
                # Compute average spanwise tangent (tangential_i)
                span_vector_1 = np.array(panel_corners[3]) - np.array(panel_corners[0])  # Spanwise edge 1
                span_vector_2 = np.array(panel_corners[2]) - np.array(panel_corners[1])  # Spanwise edge 2
                span_vector = 0.5 * (span_vector_1 + span_vector_2)  # Average
                tangential_i = span_vector # Normalize

                # Compute average chordwise tangent (tangential_j)
                chord_vector_1 = np.array(panel_corners[1]) - np.array(panel_corners[0])  # Chordwise edge 1
                chord_vector_2 = np.array(panel_corners[2]) - np.array(panel_corners[3]) # Chordwise edge 2
                chord_vector = 0.5 * (chord_vector_1 + chord_vector_2)  # Average
                tangential_j = chord_vector # Normalize

                # span_vector = np.array(panel_corners[3]) - np.array(panel_corners[0])  # Spanwise direction (leading edge to trailing edge)
                # chord_vector = np.array(panel_corners[1]) - np.array(panel_corners[0])  # Chordwise direction (leading edge to trailing edge)

                # tangential_i = span_vector / np.linalg.norm(span_vector)  # Normalize to get unit tangential vector
                # tangential_j = chord_vector / np.linalg.norm(chord_vector)  # Normalize to get unit tangential vector

                tangential_vectors[panel_index] = {
                    'Tangential i': tangential_i,
                    'Tangential j': tangential_j
                }

        # Return the dictionary for the blade mesh
        return {
            'Panels': panels,
            'Control Points': control_points,
            'Vortex Rings': vortex_rings,   
            'Normals': normals,
            'Gamma': gamma,
            'Tangential Vectors': tangential_vectors
        }

    def generate_mesh(self):
        """
        Generate the mesh for a single propeller with two blades, including distance data.

        Returns:
            dict: The dictionary containing the mesh for the single propeller.
        """
        # Generate blade surfaces
        X1_flat, Y1_flat, Z1_flat, X2_flat, Y2_flat, Z2_flat = self.propeller_geometry.generate_flat_blade_surfaces()

        # Create dictionaries for Blade 1 and Blade 2
        blades = {
            'Blade_1': self._compute_blade_mesh(X1_flat, Y1_flat, Z1_flat),
            'Blade_2': self._compute_blade_mesh(X2_flat, Y2_flat, Z2_flat)
        }

        # Initialize Distance Data
        distance_data = {'Control Point': []}

        # Helper function to calculate distances
        def calculate_distances(cp_index, cp_coords, target_vortex_rings):
            distances_to_vertices = []
            for (ring_index, vortex_data) in target_vortex_rings.items():
                distances = [np.array(cp_coords) - np.array(vertex) for vertex in vortex_data['Vertices']]
                distances_to_vertices.append({
                    'Ring Index': ring_index,
                    'Distances': distances
                })
            return distances_to_vertices

        # Process control points for both blades
        for blade_key, other_blade_key in [('Blade_1', 'Blade_2'), ('Blade_2', 'Blade_1')]:
            for cp_index, cp_coords in blades[blade_key]['Control Points'].items():
                cp_distance_data = {
                    'Index': cp_index,
                    'Distances to Vertices Blade 1': calculate_distances(cp_index, cp_coords, blades['Blade_1']['Vortex Rings']),
                    'Distances to Vertices Blade 2': calculate_distances(cp_index, cp_coords, blades['Blade_2']['Vortex Rings']),
                }
                distance_data['Control Point'].append(cp_distance_data)

        # Return the combined dictionary
        return {
            'Blades': blades,
            'Distance Data': distance_data
        }
    
    def generate_quad_propeller_mesh(self):
        """
        Generate the quadcopter mesh by creating individual propeller meshes and translating them to the
        respective hub positions. Use different methods for calculating normals for Propellers 1 & 4 and Propellers 2 & 3.

        Returns:
            dict: A dictionary containing the mesh data for all four propellers.
        """

        # Get hub positions
        hub_positions = self._generate_hub_points()

        # Initialize quadcopter mesh
        quad_mesh = {}

        # Generate the mesh for a single propeller
        single_propeller_mesh = self.generate_mesh()

        # Loop over hub positions and create translated propellers
        for idx, hub_position in enumerate(hub_positions):
            propeller_key = f'Propeller_{idx + 1}'  # Unique key for each propeller

            # Translate blades
            translated_blades = {}
            for blade_key, blade_data in single_propeller_mesh['Blades'].items():
                # Translate Panels
                translated_panels = {}
                translated_normals = {}
                translated_tangential_vectors = {}

                for (i, j), panel in blade_data['Panels'].items():
                    # Transform panels based on the propeller index
                    if idx in [0, 3]:  # Propellers 1 and 4 (CW)
                        transformed_panel = [np.array(vertex) + hub_position for vertex in panel]
                    else:  # Propellers 2 and 3 (CCW)
                        transformed_panel = [
                            np.array([-vertex[0], vertex[1], vertex[2]]) + hub_position for vertex in panel
                        ]
                    translated_panels[(i, j)] = transformed_panel

                    # Calculate normals based on the propeller index
                    if idx in [0, 3]:  # Propellers 1 and 4: Standard normal calculation
                        span_vector = transformed_panel[2] - transformed_panel[0]  # Span vector
                        chord_vector = transformed_panel[3] - transformed_panel[1]  # Chord vector
                        normal = np.cross(span_vector, chord_vector)
                    else:  # Propellers 2 and 3: Custom normal calculation
                        span_vector = transformed_panel[2] - transformed_panel[0]  # Span vector
                        chord_vector = transformed_panel[3] - transformed_panel[1]  # Chord vector
                        normal = np.cross(span_vector, chord_vector)

                    # Normalize and ensure direction consistency
                    normal = normal / np.linalg.norm(normal)
                    translated_normals[(i, j)] = normal

                    # Calculate tangential vectors based on the propeller index
                    span_vector_1 = transformed_panel[3] - transformed_panel[0]  # Chordwise edge 1
                    span_vector_2 = transformed_panel[2] - transformed_panel[1]  # Chordwise edge 2
                    avg_span_vector = 0.5 * (span_vector_1 + span_vector_2)

                    chord_vector_1 = transformed_panel[1] - transformed_panel[0]  # Chordwise edge 1
                    chord_vector_2 = transformed_panel[2] - transformed_panel[3]  # Chordwise edge 2
                    avg_chord_vector = 0.5 * (chord_vector_1 + chord_vector_2)

                    tangential_i = avg_span_vector
                    tangential_j = avg_chord_vector 
                
                    translated_tangential_vectors[(i, j)] = {
                        'Tangential i': tangential_i,
                        'Tangential j': tangential_j
                    }

                # Translate Control Points
                translated_control_points = {
                    (i, j): (
                        np.array(cp) + hub_position if idx in [0, 3]  # Propellers 1 and 4
                        else np.array([-cp[0], cp[1], cp[2]]) + hub_position  # Propellers 2 and 3 mirrored
                            
                    )   
                    for (i, j), cp in blade_data['Control Points'].items()       
                }
         
                # Translate Vortex Rings (only vertices; edge vectors remain the same)
                translated_vortex_rings = {
                    (i, j): {
                        'Vertices': [
                            np.array(vertex) + hub_position if idx in [0, 3]  # Propellers 1 and 4
                            else np.array([-vertex[0], vertex[1], vertex[2]]) + hub_position  # Propellers 2 and 3 mirrored
                            for vertex in vortex_data['Vertices']
                        ],
                        'Edge Vectors': vortex_data['Edge Vectors']  # Edge vectors remain unchanged
                    }
                    for (i, j), vortex_data in blade_data['Vortex Rings'].items()
                }

                # Add all translated and recalculated components to the blade
                translated_blades[blade_key] = {
                    'Panels': translated_panels,
                    'Normals': translated_normals,
                    'Tangential Vectors': translated_tangential_vectors,
                    'Control Points': translated_control_points,
                    'Vortex Rings': translated_vortex_rings,
                    'Gamma': blade_data['Gamma']
                }
                
            # Add the translated blades to the quadcopter mesh
            quad_mesh[propeller_key] = {
                'Blades': translated_blades,
                'Distance Data': single_propeller_mesh['Distance Data'],  # Use original, unmodified Distance Data
                'Hub Position': hub_position
            }
            
        return quad_mesh

    def plot_propeller_mesh_with_highlight(self, quad_propeller_mesh, propeller_key='Propeller_1', 
                                    blade_key='Blade_1', highlight_panel=None):
        """
        Plot the entire propeller mesh with option to highlight a specific panel.
        
        Args:
            quad_propeller_mesh: The full mesh
            propeller_key: Which propeller to visualize
            blade_key: Which blade to visualize
            highlight_panel: Tuple of (i,j) for panel to highlight, or None
        """
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        blade_data = quad_propeller_mesh[propeller_key]['Blades'][blade_key]
        
        # Plot all panels in light gray
        for panel_idx, panel in blade_data['Panels'].items():
            panel_array = np.array(panel)
            
            # Regular panels in light gray
            if panel_idx != highlight_panel:
                # Plot panel edges
                for i in range(4):
                    start = panel[i]
                    end = panel[(i+1)%4]
                    ax.plot([start[0], end[0]], 
                        [start[1], end[1]], 
                        [start[2], end[2]], 
                        'gray', alpha=0.3)
                    
                # Plot control point
                cp = blade_data['Control Points'][panel_idx]
                ax.scatter(cp[0], cp[1], cp[2], color='gray', alpha=0.3, s=30)
        
        # If there's a panel to highlight
        if highlight_panel and highlight_panel in blade_data['Panels']:
            panel = blade_data['Panels'][highlight_panel]
            vortex_ring = blade_data['Vortex Rings'][highlight_panel]
            control_point = blade_data['Control Points'][highlight_panel]
            normal = blade_data['Normals'][highlight_panel]
            
            # Plot highlighted panel vertices
            for i, vertex in enumerate(panel):
                ax.scatter(vertex[0], vertex[1], vertex[2], color='blue', s=100)
                ax.text(vertex[0], vertex[1], vertex[2], f'P{i}', fontsize=12)
            
            # Connect highlighted panel vertices
            for i in range(4):
                start = panel[i]
                end = panel[(i+1)%4]
                ax.plot([start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]], 
                    'b-', linewidth=2)
            
            # Plot vortex ring points
            vortex_vertices = vortex_ring['Vertices']
            for i, vertex in enumerate(vortex_vertices):
                ax.scatter(vertex[0], vertex[1], vertex[2], color='red', s=100)
                ax.text(vertex[0], vertex[1], vertex[2], f'V{i}', fontsize=12)
            
            # Connect vortex ring points
            for i in range(4):
                start = vortex_vertices[i]
                end = vortex_vertices[(i+1)%4]
                ax.plot([start[0], end[0]], 
                    [start[1], end[1]], 
                    [start[2], end[2]], 
                    'r--', linewidth=2)
            
            # Plot control point
            ax.scatter(control_point[0], control_point[1], control_point[2], 
                    color='green', s=100)
            ax.text(control_point[0], control_point[1], control_point[2], 
                    'CP', fontsize=12)
            
            # Plot normal vector
            scale = 0.05
            normal_end = control_point + normal * scale
            ax.quiver(control_point[0], control_point[1], control_point[2],
                    normal[0], normal[1], normal[2],
                    length=scale, color='k', arrow_length_ratio=0.2)
        
        # Add legend
        if highlight_panel:
            ax.scatter([], [], [], c='blue', s=100, label='Panel Vertices (P0-P3)')
            ax.scatter([], [], [], c='red', s=100, label='Vortex Ring Points (V0-V3)')
            ax.scatter([], [], [], c='green', s=100, label='Control Point (CP)')
            ax.quiver([], [], [], [], [], [], color='k', label='Normal Vector')
            ax.legend()
        
        # Labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        title = f'Propeller Mesh ({propeller_key}, {blade_key})'
        if highlight_panel:
            title += f'\nHighlighted Panel: {highlight_panel}'
        ax.set_title(title)
        
        # Set equal aspect ratio
        max_range = np.array([
            ax.get_xlim3d()[1] - ax.get_xlim3d()[0],
            ax.get_ylim3d()[1] - ax.get_ylim3d()[0],
            ax.get_zlim3d()[1] - ax.get_zlim3d()[0]
        ]).max() / 2.0
        
        mid_x = (ax.get_xlim3d()[1] + ax.get_xlim3d()[0]) / 2.0
        mid_y = (ax.get_ylim3d()[1] + ax.get_ylim3d()[0]) / 2.0
        mid_z = (ax.get_zlim3d()[1] + ax.get_zlim3d()[0]) / 2.0
        ax.set_xlim3d([mid_x - max_range, mid_x + max_range])
        ax.set_ylim3d([mid_y - max_range, mid_y + max_range])
        ax.set_zlim3d([mid_z - max_range, mid_z + max_range])
        
        plt.show()

    def print_panel_indices(self, quad_propeller_mesh, propeller_key='Propeller_1', blade_key='Blade_1'):
            """
            Print available panel indices for selection.
            """
            blade_data = quad_propeller_mesh[propeller_key]['Blades'][blade_key]
            print(f"\nAvailable panel indices for {propeller_key}, {blade_key}:")
            indices = sorted(blade_data['Panels'].keys())
            for idx in indices:
                print(f"Panel index: {idx}")

# # Initialize and generate mesh as before
# propeller_geometry = PropellerGeometry(
#     airfoil_distribution_file='DJI9443_airfoils.csv',
#     chorddist_file='DJI9443_chorddist.csv',
#     pitchdist_file='DJI9443_pitchdist.csv',
#     sweepdist_file='DJI9443_sweepdist.csv',
#     heightdist_file='DJI9443_heightdist.csv',
#     R_tip=0.12,
#     R_hub=0.0064
# )
# propeller_mesh_system = PropellerMesh(propeller_geometry, arm_length=0.5, com=(0, 0, 0))
# quad_propeller_mesh = propeller_mesh_system.generate_quad_propeller_mesh()

# # First print available panel indices
# propeller_mesh_system.print_panel_indices(quad_propeller_mesh)

# # Then plot with a highlighted panel
# propeller_mesh_system.plot_propeller_mesh_with_highlight(
#     quad_propeller_mesh,
#     propeller_key='Propeller_1',
#     blade_key='Blade_1',
#     highlight_panel=(3,0)  # Change this to highlight different panels
# )
