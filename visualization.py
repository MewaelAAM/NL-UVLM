import numpy as np
from propeller import PropellerGeometry
from mesh import PropellerMesh
from scipy.interpolate import UnivariateSpline
from scipy.io import savemat
from mpl_toolkits.mplot3d import Axes3D
from propeller import PropellerGeometry
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numba as nb



class UVLM:
    def __init__(self, propeller_mesh):
        """
        Initialize the UVLM solver with the given propeller mesh.

        Args:
            propeller_mesh (dict): Dictionary containing the mesh information for all propellers.
        """
        self.propeller_mesh = propeller_mesh
        self.induced_velocities = {}  # To store induced velocities at each timestep

    def biot_savart(self, control_point, r1, r2, r0, gamma):
        """
        Calculate the induced velocity at a control point due to a vortex filament using Biot-Savart law.

        Args:
            control_point (np.ndarray): Coordinates of the control point (shape: (3,)).
            r1 (np.ndarray): Vector from the control point to the first vertex of the vortex filament.
            r2 (np.ndarray): Vector from the control point to the second vertex of the vortex filament.
            r0 (np.ndarray): Vortex filament vector (from vertex 1 to vertex 2).
            gamma (float): Circulation strength of the vortex filament.

        Returns:
            np.ndarray: Induced velocity at the control point (shape: (3,)).
        """
  
        cross_r1_r2 = np.cross(r1, r2)
        norm_cross_r1_r2 = np.linalg.norm(cross_r1_r2)

        if norm_cross_r1_r2 == 0:
            norm_cross_r1_r2 = 1e-8
        
        segment_induced_velocity =    (
        (gamma / (4 * np.pi * norm_cross_r1_r2 ** 2)) *
        (np.dot(r0, r1) / np.linalg.norm(r1) - np.dot(r0, r2) / np.linalg.norm(r2))
    ) * cross_r1_r2




        return segment_induced_velocity

    def calculate_global_induced_velocity_matrix(self, quad_propeller_mesh):
        """
        Calculate the global induced velocity matrix for all control points and vortex rings
        in a propeller with gamma set to 1.

        Args:
            quad_propeller_mesh (dict): Mesh data for all propellers.

        Returns:
            dict: A dictionary where each propeller contains its global induced velocity matrix.
                Each element is a 3D vector [u, v, w].
        """
        global_matrices = {}  # To store the global induced velocity matrix for each propeller

        for propeller_key, propeller_data in quad_propeller_mesh.items():
            # print(f"Processing {propeller_key}...")

            # Flatten all control points and vortex rings with their indices
            control_points = []
            vortex_rings = []

            for blade_key, blade_data in propeller_data['Blades'].items():
         
                # Collect control points
                control_points.extend(
                    [(blade_key, cp_index, control_point) for cp_index, control_point in blade_data['Control Points'].items()]
                )
                # Collect vortex rings
                vortex_rings.extend(
                    [(blade_key, vr_index, vortex_ring_data) for vr_index, vortex_ring_data in blade_data['Vortex Rings'].items()]
                )

            # for vortex_ring in control_points:
            #     print(vortex_ring)
            # Number of control points and vortex rings
            num_control_points = len(control_points)
            num_vortex_rings = len(vortex_rings)

            # Initialize the global matrix
            global_matrix = np.zeros((num_control_points, num_vortex_rings, 3))  # Each element is a 3D vector [u, v, w]
            
            # Compute induced velocity for each control point and vortex ring
            for row, (cp_blade_key, cp_index, control_point) in enumerate(control_points):
                for col, (vr_blade_key, vr_index, vortex_ring) in enumerate(vortex_rings):
                    total_induced_velocity = np.zeros(3)  # Initialize induced velocity for this control point-vortex ring pair
                    # Extract vertices and edge vectors for the current vortex ring
                    vertices = vortex_ring['Vertices']
                    edge_vectors = vortex_ring['Edge Vectors']

                    for filament_index in range(4):
                        vertex_start = np.array(vertices[filament_index])
                        vertex_end = np.array(vertices[(filament_index + 1) % 4])  # Wrap around to close the ring
                        edge_vector = edge_vectors[filament_index]
                        # print((vertex_start) - (vertex_end))

                        # print("mirrored:", vr_blade_key) 

                        r1 = control_point - vertex_start
                        r2 = control_point - vertex_end
                        r0 = vertex_end - vertex_start
               
                        # Biot-Savart contribution for this filament
                        induced_velocity = self.biot_savart(control_point, r1, r2, r0, gamma=1)  # Gamma = 1
                     
                        total_induced_velocity += induced_velocity
     
                    global_matrix[row, col] = total_induced_velocity
            # print(global_matrix)
            # Store the global matrix for the current propeller
            global_matrices[propeller_key] = global_matrix
        
        
        return global_matrices
    
    def calculate_gamma(self, quad_propeller_mesh, global_induced_matrix, omega_dict, body_velocity, wind_function, roll, pitch, yaw):
        """
        Calculate gamma (circulation strength) for each propeller using the Neumann boundary condition.

        Args:
            quad_propeller_mesh (dict): Mesh data for all propellers.
            global_induced_matrix (dict): Induced velocity matrix for each propeller.
            omega_dict (dict): Dictionary of angular velocity vectors for each propeller.
                            E.g., {'Propeller 1': omega1, 'Propeller 2': omega2, ...}.
            body_velocity (np.ndarray): Body velocity vector.
            wind_function (function): Function to calculate wind velocity at a control point.
            roll (float): Roll angle in radians.
            pitch (float): Pitch angle in radians.
            yaw (float): Yaw angle in radians.

        Returns:
            dict: Gamma values for each propeller.
        """
        def rotation_matrix(roll, pitch, yaw):
            """Compute rotation matrix for roll, pitch, and yaw."""
            R_yaw = np.array([
                [np.cos(yaw), -np.sin(yaw), 0],
                [np.sin(yaw), np.cos(yaw), 0],
                [0, 0, 1]
            ])
            R_pitch = np.array([
                [np.cos(pitch), 0, np.sin(pitch)],
                [0, 1, 0],
                [-np.sin(pitch), 0, np.cos(pitch)]
            ])
            R_roll = np.array([
                [1, 0, 0],
                [0, np.cos(roll), -np.sin(roll)],
                [0, np.sin(roll), np.cos(roll)]
            ])

            
            
            return R_yaw @ R_pitch @ R_roll

        R = rotation_matrix(roll, pitch, yaw)
        gamma_matrices = {}
    
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            print(f"Processing {propeller_key}...")
            
            # Retrieve the angular velocity vector for this propeller
            effective_omega = omega_dict[propeller_key]

            # Collect data for control points
            control_points = []
            normals = []
            rhs = []
            hub_position = np.array(propeller_data['Hub Position'])

            for blade_key, blade_data in propeller_data['Blades'].items():
                for cp_index, control_point in blade_data['Control Points'].items():
    
                    # Rotated normal vector
                    normal = blade_data['Normals'][cp_index]
                    rotated_normal = R @ normal
                    normals.append(rotated_normal)
                    # Radius vector and wind velocity
                    radius_vector = control_point - hub_position
                    # print(control_point / /np.linalg.norm(control_point))
                    # print(cp_index)
                    # Calculate omega x r
                    omega_cross_r = np.cross(effective_omega, radius_vector)
                    # print(omega_cross_r)
                    # Store omega x r in the control point's dictionary
                    if "Omega_Cross_R" not in blade_data:
                        blade_data['Omega_Cross_R'] = {}
                    blade_data['Omega_Cross_R'][cp_index] = omega_cross_r
                    
                    # Wind velocity
                    wind_velocity = wind_function(control_point)

                    # Calculate the RHS term
                    velocity_term =  omega_cross_r 
                
                    rhs_value = -np.dot(velocity_term, rotated_normal)
                    rhs.append(rhs_value)
                
                    # Store control point
                    control_points.append(control_point)
            
            # Convert to NumPy arrays
            normals = np.array(normals)  # Shape: (N, 3)
            rhs = np.array(rhs).reshape(-1, 1)  # Shape: (N, 1)
            
            # Retrieve the global induced velocity matrix
            induced_matrix = global_induced_matrix[propeller_key]  # Shape: (N, N, 3)

            # Compute A matrix using dot product of induced velocities and normals
            A = np.einsum('ijk,ik->ij', induced_matrix, normals)  # Shape: (N, N)

        
            # Solve for gamma
            gamma = np.linalg.solve(A, rhs)  # Shape: (N, 1)
            # gamma_matrices[propeller_key] = gamma.copy()  # Make a copy to prevent reference issues
            # Compute the residual
            # Inside the loop where gamma is being assigned
            # Store gamma in the matrix for the current propeller
            gamma_matrices[propeller_key] = gamma.flatten()

            # Update gamma values in the propeller mesh for each blade
             # Update gamma values
            gamma_index = 0
            for blade_key, blade_data in propeller_data['Blades'].items():
                for cp_index in blade_data['Control Points'].keys():
                    new_gamma = float(gamma[gamma_index, 0])
                    
                    # Create a deep copy of the Gamma dictionary
                    new_gamma_dict = blade_data['Gamma'].copy()
                    new_gamma_dict[cp_index] = new_gamma
                    blade_data['Gamma'] = new_gamma_dict
                    
                    gamma_index += 1
            # Immediately update the mesh for this propeller
            # gamma_idx = 0
            # for blade_key, blade_data in propeller_data['Blades'].items():
            #     for cp_index in blade_data['Control Points'].keys():
            #         gamma_val = float(gamma[gamma_idx])  # Get the value for this control point
            #         # Make a new dictionary entry for each gamma value
            #         blade_data['Gamma'] = {cp_index: gamma_val}
            #         gamma_idx += 1
            # print("Gamma", quad_propeller_mesh['Propeller_4']['Blades']['Blade_1']['Gamma'])
            print('finihsed')

        print("Gamma", quad_propeller_mesh['Propeller_1']['Blades']['Blade_1']['Gamma'])
        print("Gamma", quad_propeller_mesh['Propeller_1']['Blades']['Blade_2']['Gamma'])
        # print("-------------------------------------------------------------------------")
        print("Gamma", quad_propeller_mesh['Propeller_2']['Blades']['Blade_1']['Gamma'])
        print("Gamma", quad_propeller_mesh['Propeller_2']['Blades']['Blade_2']['Gamma'])
        
        # print("-------------------------------------------------------------------------")
        # print("Gamma", quad_propeller_mesh['Propeller_3']['Blades']['Blade_1']['Gamma'])
        # print("Gamma", quad_propeller_mesh['Propeller_3']['Blades']['Blade_2']['Gamma'])
        # print("-------------------------------------------------------------------------")
        # print("Gamma", quad_propeller_mesh['Propeller_4']['Blades']['Blade_1']['Gamma'])
        # print("Gamma", quad_propeller_mesh['Propeller_4']['Blades']['Blade_2']['Gamma'])
        return gamma_matrices
            
            
    def pressure_difference(self, quad_propeller_mesh, global_matrices, body_velocity, omega, wind_function, dt, rho):
        """
        Calculate the pressure difference for each panel and store it in the mesh dictionary.

        Args:
            quad_propeller_mesh (dict): Mesh data for all propellers.
            global_matrices (dict): Induced velocity matrices for all propellers.
            body_velocity (np.ndarray): Velocity of the propeller body (3D vector).
            wind_function (function): Function to compute wind velocity at a given point.
            dt (float): Time step for calculating the time derivative of gamma.
            rho (float): Air density.
        """
        total_forces = {}
        # Iterate over all propellers
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            # print(f"Processing {propeller_key}...")

            # Get the induced velocity matrix for the current propeller
            induced_velocity_matrix = global_matrices[propeller_key]
            # print(induced_velocity_matrix)
            for blade_key, blade_data in propeller_data['Blades'].items():
                control_points = blade_data['Control Points']
                normals = blade_data['Normals']
                tangential_vectors = blade_data['Tangential Vectors']
                gamma_old = blade_data.get('Gamma Old', {})  # Previous gamma values (default to 0 if not available)
                total_force = np.zeros(3)
                pressure_difference = {}  # Store pressure difference for each panel
                
                for panel_index, control_point in control_points.items():
                    # Get the normal and tangential vectors for the current panel
                    normal = normals[panel_index]
                    tangent_span = tangential_vectors[panel_index]['Tangential i'] 
                    tangent_chord = tangential_vectors[panel_index]['Tangential j'] 

                    omega_cross_r = blade_data['Omega_Cross_R'][panel_index]
                    induced_velocity = induced_velocity_matrix[panel_index]
                    wind_velocity = wind_function(control_point)

                    total_velocity = omega_cross_r 

         
                    # Circulation differences (γ_i,j - γ_i-1,j and γ_i,j - γ_i,j-1)
                    gamma_current = blade_data['Gamma'][panel_index]
                    # print(gamma_current)


                    # Spanwise circulation (handle edge cases where index might be invalid)
                    if panel_index[0] - 1 < 0:
                        gamma_previous_span = 0  # No neighbor in the spanwise direction
                    else:
                        gamma_previous_span = blade_data['Gamma'].get((panel_index[0] - 1, panel_index[1]), 0)

                    # Chordwise circulation (handle edge cases where index might be invalid)z
                    if panel_index[1] - 1 < 0:
                        gamma_previous_chord = 0  # No neighbor in the chordwise direction
                    else:
                        gamma_previous_chord = blade_data['Gamma'].get((panel_index[0], panel_index[1] - 1), 0)


                    gamma_diff_span = (gamma_current - gamma_previous_span) / np.linalg.norm(tangent_span)
                    gamma_diff_chord = (gamma_current - gamma_previous_chord) / np.linalg.norm(tangent_chord)


                    # Time derivative of gamma (dγ/dt)
                    gamma_dot = (gamma_current - gamma_old.get(panel_index, 0)) / dt
                    
                    pressure = rho * (
                        np.dot(total_velocity, tangent_span / np.linalg.norm(tangent_span)  * gamma_diff_span) +
                        np.dot(total_velocity, tangent_chord / np.linalg.norm(tangent_chord) * gamma_diff_chord)
              
                    )
                    # fsteady = rho * gamma_current * np.cross(total_velocity, (tangent_span)) 

                    # funsteady = rho * gamma_dot * np.linalg.norm(tangent_span) * np.cross(total_velocity/np.linalg.norm(total_velocity), tangent_span)
                    # pressure = + fsteady + funsteady

                    pressure_difference[panel_index] = pressure
                    
                # Update the mesh dictionary with pressure difference and gamma_old
                blade_data['Pressure Difference'] = pressure_difference
                blade_data['Gamma Old'] = blade_data['Gamma'].copy()
                

        print("Presuusre Difference", quad_propeller_mesh['Propeller_1']['Blades']['Blade_1']['Pressure Difference'])
        print("Presuusre Difference", quad_propeller_mesh['Propeller_1']['Blades']['Blade_2']['Pressure Difference'])
        print("-"*100)
        print("Presuusre Difference", quad_propeller_mesh['Propeller_2']['Blades']['Blade_1']['Pressure Difference'])
        print("Presuusre Difference", quad_propeller_mesh['Propeller_2']['Blades']['Blade_2']['Pressure Difference'])
        print("-"*100)
        # print("Presuusre Difference", quad_propeller_mesh['Propeller_3']['Blades']['Blade_1']['Pressure Difference'])
        # print("Presuusre Difference", quad_propeller_mesh['Propeller_3']['Blades']['Blade_2']['Pressure Difference'])
        # print("-"*100)
        # print("Presuusre Difference", quad_propeller_mesh['Propeller_4']['Blades']['Blade_1']['Pressure Difference'])
        # print("Presuusre Difference", quad_propeller_mesh['Propeller_4']['Blades']['Blade_2']['Pressure Difference'])
        # print("-"*100)

    def calculate_total_forces_and_moments(self, propeller_mesh, body_velocity, omega, wind_function, dt, rho):
        """
        Calculate aerodynamic forces and moments for each panel of each propeller
        using UVLM. This includes updating the pressure difference for each panel.

        Args:
            propeller_mesh (dict): Mesh data for all propellers.
            global_matrices (dict): Induced velocity matrices for all propellers.
            body_velocity (np.ndarray): Velocity of the propeller body (3D vector).
            omega (np.ndarray): Angular velocity of the propeller.
            wind_function (function): Function to compute wind velocity at a given point.

        Returns:
            dict: Total forces and moments for each propeller, in the form:
                { propeller_key: {'force': total_force, 'moment': total_moment} }
        Updates:
            - Calls the UVLM solver functions to compute induced velocities, gamma, and pressure differences.
            - Adds 'Panel Forces' and 'Panel Moments' entries to each blade in the propeller_mesh dictionary.
            Each panel's force and moment contributions are stored individually.
        """
        # Step 1: Compute the global induced velocity matrix
        global_matrices = self.calculate_global_induced_velocity_matrix(propeller_mesh)

        # Step 2: Calculate gamma for each panel
        self.calculate_gamma(
            propeller_mesh,
            global_matrices,
            omega,
            body_velocity,
            wind_function,
            roll=0.0,
            pitch=0.0,
            yaw=0.0
        )

        # Step 3: Update the pressure differences for all panels
        self.pressure_difference(
            propeller_mesh,
            global_matrices,
            body_velocity,
            omega,
            wind_function,
            dt,
            rho
        )

        # Initialize dictionary to store total forces and moments for each propeller
        total_forces_and_moments = {}

        # Step 4: Calculate forces and moments for each panel
        for propeller_key, propeller_data in propeller_mesh.items():
            hub_position = np.array(propeller_data['Hub Position'])

            # Initialize total force and moment for this propeller
            total_force = np.zeros(3)
            total_moment = np.zeros(3)

            for blade_key, blade_data in propeller_data['Blades'].items():
                # Initialize storage for panel forces and moments
                blade_data['Panel Forces'] = {}
                blade_data['Panel Moments'] = {}
            
                for panel_index, pressure in blade_data['Pressure Difference'].items():
                    # Panel geometry
                    panel = blade_data['Panels'][panel_index]
                    panel_array = np.array(panel)
                    panel_center = panel_array.mean(axis=0)
            
                    
                    # Calculate the normal vector and panel area
                    normal = blade_data['Normals'][panel_index]
                    
                    area_triangle_1 = 0.5 * np.linalg.norm(np.cross(panel_array[1] - panel_array[0], panel_array[3] - panel_array[0]))
                    area_triangle_2 = 0.5 * np.linalg.norm(np.cross(panel_array[2] - panel_array[1], panel_array[3] - panel_array[1]))

                    # panel_area =  0.5 * np.linalg.norm(
                    #     np.cross(panel_array[2] - panel_array[0], panel_array[3] - panel_array[1])
                    # )
                    panel_area = area_triangle_1 + area_triangle_2

                    # Calculate vectors for Triangle 1
                    
                    # Force contribution for this panel
                    force = -np.dot(pressure * panel_area, normal)
                    
                    # force = pressure
                    # Moment contribution for this panel
                    moment_arm = panel_center - hub_position
                    # moment = np.cross(moment_arm, force)
                    moment = force

                    # Store the force and moment for this panel
                    blade_data['Panel Forces'][panel_index] = force
                    blade_data['Panel Moments'][panel_index] = moment

                    # Accumulate to total force and moment
                    total_force += force
                    total_moment += moment

            # Store total force and moment for this propeller
            total_forces_and_moments[propeller_key] = {'force': total_force, 'moment': total_moment}
            
        # print("Panel Forces", quad_propeller_mesh['Propeller_1']['Blades']['Blade_1']['Panel Forces'])
        # print("Panel Forces", quad_propeller_mesh['Propeller_1']['Blades']['Blade_2']['Panel Forces'])
        # print('--'*50)
        # print("Panel Forces", quad_propeller_mesh['Propeller_2']['Blades']['Blade_1']['Panel Forces'])
        # print("Panel Forces", quad_propeller_mesh['Propeller_2']['Blades']['Blade_2']['Panel Forces'])
        # print('--'*50)
        # print("Panel Forces", quad_propeller_mesh['Propeller_3']['Blades']['Blade_1']['Panel Forces'])
        # print("Panel Forces", quad_propeller_mesh['Propeller_3']['Blades']['Blade_2']['Panel Forces'])
        # print('--'*50)
        # print("Panel Forces", quad_propeller_mesh['Propeller_4']['Blades']['Blade_1']['Panel Forces'])
        # print("Panel Forces", quad_propeller_mesh['Propeller_4']['Blades']['Blade_2']['Panel Forces'])
        # print('--'*50)
        # print('--'*50)
        print("Forces", total_forces_and_moments['Propeller_1']['force'])
        print("Forces", total_forces_and_moments['Propeller_2']['force'])
        print("Forces", total_forces_and_moments['Propeller_3']['force'])
        print("Forces", total_forces_and_moments['Propeller_4']['force'])
        # # return total_forces_and_moments

    def plot_entire_propeller_with_normals_and_tangentials(
        self, quad_propeller_mesh, propeller_key, highlight_blade_key=None, highlight_panel_index=None,
        normal_scale=0.01, tangential_scale=0.01, additional_normal=None, additional_normal_color="cyan",
        plot_control_points=True, control_point_color="magenta", control_point_size=50,
        highlight_control_points=None, highlight_control_point_color="yellow", highlight_control_point_size=100
    ):
        """
        Extended to include plotting of vortex rings and labeling vertices.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Extract propeller data
        propeller_data = quad_propeller_mesh[propeller_key]

        # Start plotting
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection="3d")

        # Loop through all blades and panels
        for blade_key, blade_data in propeller_data["Blades"].items():
            for panel_index, panel in blade_data["Panels"].items():
                panel_array = np.array(panel)

                # Highlight the selected panel if specified
                if blade_key == highlight_blade_key and panel_index == highlight_panel_index:
                    color = "yellow"
                    alpha = 0.8
                else:
                    color = "blue"
                    alpha = 0.3

                # Plot the panel
                poly = Poly3DCollection([panel_array], alpha=alpha, edgecolor="black", facecolor=color)
                ax.add_collection3d(poly)

                # Retrieve the control point for this panel
                control_point = blade_data["Control Points"][panel_index]

                # Plot control points if enabled
                if plot_control_points:
                    # Check if the control point should be highlighted
                    if highlight_control_points and (blade_key, panel_index) in highlight_control_points:
                        ax.scatter(
                            control_point[0], control_point[1], control_point[2],
                            color=highlight_control_point_color, s=highlight_control_point_size,
                            label="Highlighted Control Point" if panel_index == (0, 0) else None
                        )
                    else:
                        ax.scatter(
                            control_point[0], control_point[1], control_point[2],
                            color=control_point_color, s=control_point_size,
                            label="Control Point" if panel_index == (0, 0) else None
                        )

                # Retrieve and plot the stored normal vector
                normal = blade_data["Normals"][panel_index]
                ax.quiver(
                    control_point[0], control_point[1], control_point[2],  # Origin of the vector (control point position)
                    normal[0] * normal_scale, normal[1] * normal_scale, normal[2] * normal_scale,  # Components of the vector
                    color="red",
                    label="Normal Vector" if panel_index == (0, 0) else None  # Add label only once
                )

                # Retrieve tangential vectors for the panel
                tangential_i = blade_data["Tangential Vectors"][panel_index]["Tangential i"]
                tangential_j = blade_data["Tangential Vectors"][panel_index]["Tangential j"]

                # Plot the tangential vectors
                ax.quiver(
                    control_point[0], control_point[1], control_point[2],  # Origin of Tangential i
                    tangential_i[0] * tangential_scale, tangential_i[1] * tangential_scale, tangential_i[2] * tangential_scale,
                    color="green",
                    label="Tangential i" if panel_index == (0, 0) else None  # Add label only once
                )
                ax.quiver(
                    control_point[0], control_point[1], control_point[2],  # Origin of Tangential j
                    tangential_j[0] * tangential_scale, tangential_j[1] * tangential_scale, tangential_j[2] * tangential_scale,
                    color="blue",
                    label="Tangential j" if panel_index == (0, 0) else None  # Add label only once
                )

                # Plot vortex rings and label vertices
                vortex_ring = blade_data["Vortex Rings"][panel_index]
                vertices = vortex_ring["Vertices"]

                # Plot the edges of the vortex ring
                for i in range(len(vertices)):
                    start_vertex = vertices[i]
                    end_vertex = vertices[(i + 1) % len(vertices)]  # Wrap around to form a closed loop
                    ax.plot(
                        [start_vertex[0], end_vertex[0]],
                        [start_vertex[1], end_vertex[1]],
                        [start_vertex[2], end_vertex[2]],
                        color="cyan",
                        linestyle="--"
                    )

                # Label each vertex with its index
                for i, vertex in enumerate(vertices):
                    ax.text(
                        vertex[0], vertex[1], vertex[2], str(i),
                        fontsize=10, color="purple", weight="bold"
                    )

            # Label each blade at its approximate center
            blade_center = np.mean(
                [np.mean(np.array(panel), axis=0) for panel in blade_data["Panels"].values()], axis=0
            )
            ax.text(*blade_center, blade_key, fontsize=12, color="red", weight="bold")

        # Set uniform scaling
        all_points = []
        for blade_data in propeller_data["Blades"].values():
            for panel in blade_data["Panels"].values():
                all_points.append(panel)
            for control_point in blade_data["Control Points"].values():
                all_points.append([control_point])
        all_points = np.vstack(all_points)
        max_range = np.ptp(all_points, axis=0).max()
        mid_x, mid_y, mid_z = all_points.mean(axis=0)
        ax.set_xlim(mid_x - max_range / 2, mid_x + max_range / 2)
        ax.set_ylim(mid_y - max_range / 2, mid_y + max_range / 2)
        ax.set_zlim(mid_z - max_range / 2, mid_z + max_range / 2)

        # Add labels and legend
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_title(f"{propeller_key} - Entire Propeller with Normals, Tangentials, and Vortex Rings")
        ax.legend(loc="best")

        plt.show()
    def plot_detailed_gamma_distribution(self, quad_propeller_mesh):
        """
        Plot detailed gamma distribution across all propellers and their blades
        including radial position information.
        """
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gamma Distribution Across Propellers (with Radial Position)')

        colors = ['b', 'r']
        axs = axs.flatten()

        for prop_idx, (propeller_key, propeller_data) in enumerate(quad_propeller_mesh.items()):
            ax = axs[prop_idx]
            
            for blade_idx, (blade_key, blade_data) in enumerate(propeller_data['Blades'].items()):
                # Extract gamma values and control point positions
                gamma_values = []
                radial_positions = []
                
                for cp_index, control_point in blade_data['Control Points'].items():
                    gamma = blade_data['Gamma'][cp_index]
                    # Calculate radial position (r/R)
                    r = np.sqrt(control_point[0]**2 + control_point[1]**2)
                    r_R = r/0.12  # Normalize by tip radius (R_tip)
                    
                    gamma_values.append(gamma)
                    radial_positions.append(r_R)
                
                # Plot for this blade
                ax.plot(radial_positions, gamma_values, f'{colors[blade_idx]}o-', 
                    label=f'{blade_key}', linewidth=2, markersize=8)

            ax.set_title(f'{propeller_key}')
            ax.set_xlabel('Radial Position (r/R)')
            ax.set_ylabel('Gamma')
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()
    def plot_gamma_distribution(self, quad_propeller_mesh):
        """
        Plot gamma distribution across all propellers and their blades.
        """
        import matplotlib.pyplot as plt

        # Create a figure with multiple subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Gamma Distribution Across Propellers')

        # Colors for blades
        colors = ['b', 'r']

        # Flatten the axes for easier iteration
        axs = axs.flatten()

        for prop_idx, (propeller_key, propeller_data) in enumerate(quad_propeller_mesh.items()):
            ax = axs[prop_idx]
            
            for blade_idx, (blade_key, blade_data) in enumerate(propeller_data['Blades'].items()):
                # Extract gamma values and their positions
                gamma_values = []
                positions = []
                
                for cp_index, gamma in blade_data['Gamma'].items():
                    gamma_values.append(gamma)
                    positions.append(cp_index[0])  # Using first index as position
                
                # Plot for this blade
                ax.plot(positions, gamma_values, f'{colors[blade_idx]}o-', 
                    label=f'{blade_key}', linewidth=2, markersize=8)

            ax.set_title(f'{propeller_key}')
            ax.set_xlabel('Control Point Index')
            ax.set_ylabel('Gamma')
            ax.grid(True)
            ax.legend()

        plt.tight_layout()
        plt.show()
    
# Initialize geometry and mesh
propeller_geometry = PropellerGeometry(
    airfoil_distribution_file='DJI9443_airfoils.csv',
    chorddist_file='DJI9443_chorddist.csv',
    pitchdist_file='DJI9443_pitchdist.csv',
    sweepdist_file='DJI9443_sweepdist.csv',
    heightdist_file='DJI9443_heightdist.csv',
    R_tip=0.12,
    R_hub=0.0064
)
propeller_mesh_system = PropellerMesh(propeller_geometry, arm_length=0.5, com=(0, 0, 0))

# propeller_mesh_system.single_propeller_mesh = single_propeller_mesh  # Assign to the system
quad_propeller_mesh = propeller_mesh_system.generate_quad_propeller_mesh()

# Initialize the UVLM solver
uvlm_solver = UVLM(quad_propeller_mesh)

# Example wind function: No wind for simplicity
def wind_function(control_point):
    return np.array([0.0, 0.0, 0.0])  # No wind

# Example body velocity
body_velocity = np.array([0, 0.0, 0.0])  # Uniform body velocity in x-direction


# uvlm_solver.plot_pressure_distribution(quad_propeller_mesh, propeller_key="Propeller_1")

omega_dict = {
    'Propeller_1': np.array([0, 0, 550]),
    'Propeller_2': np.array([0, 0, -550]),
    'Propeller_3': np.array([0, 0, -550]),
    'Propeller_4': np.array([0, 0, 550])
}
dt =0.05
rho = 1.225

# global_matrices = uvlm_solver.caLeftlculate_global_induced_velocity_matrix(quad_propeller_mesh)

uvlm_solver.calculate_total_forces_and_moments(
            propeller_mesh=quad_propeller_mesh,
            body_velocity=body_velocity,
            omega=omega_dict,
            wind_function=lambda _: np.array([0.0, 0.0, 0.0]),  # Example: no wind
            dt=dt,
            rho=rho
        )

# # Assuming `quad_propeller_mesh` is already generated
uvlm_solver.plot_entire_propeller_with_normals_and_tangentials(
    quad_propeller_mesh=quad_propeller_mesh,
    propeller_key="Propeller_1",
    highlight_blade_key="Blade_1",
    highlight_panel_index=(1, 1),  # Replace with desired panel index
    normal_scale=0.05,
    additional_normal = [0.5, 0.5, 0.5] ,
    plot_control_points=True,
    control_point_color="magenta",
    control_point_size=30,
    highlight_control_points=[("Blade_2", (0, 1))],
    highlight_control_point_color="black",
    highlight_control_point_size=100

)
# uvlm_solver.validate_normals(quad_propeller_mesh)


def plot_detailed_gamma_distribution(self, quad_propeller_mesh):
    """
    Plot detailed gamma distribution across all propellers and their blades
    including radial position information.
    """
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Gamma Distribution Across Propellers (with Radial Position)')

    colors = ['b', 'r']
    axs = axs.flatten()

    for prop_idx, (propeller_key, propeller_data) in enumerate(quad_propeller_mesh.items()):
        ax = axs[prop_idx]
        
        for blade_idx, (blade_key, blade_data) in enumerate(propeller_data['Blades'].items()):
            # Extract gamma values and control point positions
            gamma_values = []
            radial_positions = []
            
            for cp_index, control_point in blade_data['Control Points'].items():
                gamma = blade_data['Gamma'][cp_index]
                # Calculate radial position (r/R)
                r = np.sqrt(control_point[0]**2 + control_point[1]**2)
                r_R = r/0.12  # Normalize by tip radius (R_tip)
                
                gamma_values.append(gamma)
                radial_positions.append(r_R)
            
            # Plot for this blade
            ax.plot(radial_positions, gamma_values, f'{colors[blade_idx]}o-', 
                   label=f'{blade_key}', linewidth=2, markersize=8)

        ax.set_title(f'{propeller_key}')
        ax.set_xlabel('Radial Position (r/R)')
        ax.set_ylabel('Gamma')
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


# or
uvlm_solver.plot_detailed_gamma_distribution(quad_propeller_mesh)