from wind import WindField
from scipy.interpolate import UnivariateSpline
from scipy.io import savemat
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import copy
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import griddata
from propeller import PropellerGeometry

class VLM:
    def __init__(self, propeller_mesh):
        """Initialize the UVLM solver with the given propeller mesh."""
        self.propeller_mesh = propeller_mesh
        self.wake_system = {}
        # self.bound_to_bound_global_matrices = self.calculate_bound_to_bound_induced_velocity_matrix(propeller_mesh)
        self.initialize_wake_system(propeller_mesh)         

    def initialize_wake_system(self, propeller_mesh):
        """Initialize wake system with specific control point indexing pattern."""
        for prop_key, propeller_data in propeller_mesh.items():
            self.wake_system[prop_key] = {}
            for blade_key, blade_data in propeller_data['Blades'].items():
                self.wake_system[prop_key][blade_key] = {}
                
                # Get unique indices from existing control points
                max_chordwise_index = max(idx[1] for idx in blade_data['Vortex Rings'].keys())
               
                # Initialize panels for first two wake rows
                for row in range(1):  # 0 and 1
                    for chordwise_idx in range(max_chordwise_index + 1):
                        cp_index = (row, chordwise_idx)
                        self.wake_system[prop_key][blade_key][cp_index] = {
                            'Gamma': None,
                            'Control Points': None,
                            'Vortex Rings': {
                                'Vertices': np.zeros((4, 3))
                            }
                        }
        
    @staticmethod
    @nb.jit(nopython=True)
    def biot_savart(r1, r2, r0, gamma):
        """
        Calculate induced velocity using Biot-Savart law with a viscous core model
        
        Args:
            cp: Control point
            r1: Vector from start point to control point
            r2: Vector from end point to control point
            r0: Vector from start to end point (r2 - r1)
            gamma: Circulation strength
        """
        cross_r1_r2 = np.cross(r1, r2)
        norm_cross_r1_r2 = np.linalg.norm(cross_r1_r2)
        
        # Perpendicular distance for core model
        h = np.linalg.norm(np.cross(r1, r2)) / np.linalg.norm(r0) 
        test = r0 
        # Core radius and parameters
        rc = 2.92e-4  # Core radius
        n = 2         # Core model exponent
        Kv = h**2 / np.sqrt((h**4 + rc**4))
        # Denominator with viscous core correction
        # denominator = ((norm_cross_r1_r2**(2*n)) + ((np.linalg.norm(rc*r0))**(2*n)))**(1/n) 
        denominator = (norm_cross_r1_r2**(2))

        # Compute velocity
        induced_velocity = Kv * (gamma / (4 * np.pi)) * (cross_r1_r2 / denominator) * \
                        np.dot(r0, (r1/np.linalg.norm(r1) - r2/np.linalg.norm(r2)))
        
        return induced_velocity
    
    def calculate_bound_to_bound_induced_velocity_matrix(self, quad_propeller_mesh, omega_dict):
        """
        Calculate the global induced velocity matrix for all control points and vortex rings.
        Optimized for better performance while maintaining exact calculations.
        """

        global_matrices = {}

        for propeller_key, propeller_data in quad_propeller_mesh.items():

            # hub_position = np.array(propeller_data['Hub Position'])
            hub_position = np.array(propeller_data['Hub Position'])
            control_points = []
            vortex_rings = []
            blade_indices = []
            effective_omega = omega_dict[propeller_key]
            hub_position = np.array(propeller_data['Hub Position'])
            for blade_key, blade_data in propeller_data['Blades'].items():
                
                # Collect control points
                for cp_index, control_point in blade_data['Control Points'].items():
                    control_points.append((blade_key, cp_index, control_point))
                    # Collect vortex rings
                    
                    vortex_rings.append((blade_key, cp_index, blade_data['Vortex Rings'][cp_index]))
                    blade_indices.append((blade_key, cp_index))
            
            # Initialize matrix dimensions
            num_points = len(control_points)
            global_matrix = np.zeros((num_points, num_points, 3))
            max_spanwise = max(cp_index[0] if isinstance(cp_index, tuple) else int(cp_index.split(',')[0])
                          for blade_data in propeller_data['Blades'].values() 
                          for cp_index in blade_data['Control Points'].keys())
            
            # Calculate induced velocities
            for i, (cp_blade_key, cp_index, control_point) in enumerate(control_points):
                for j, (vr_blade_key, vr_index, vortex_ring) in enumerate(vortex_rings):
                    
                    # Initialize induced velocity for this pair
                    total_induced_velocity = np.zeros(3)
                    # Extract vertices and process each vortex filament
                    vertices = vortex_ring['Vertices']

                    # Handle tuple format (i, j)
                    spanwise_idx = vr_index[0]
                    chordwise_idx = vr_index[1]
                    is_trailing_edge = (spanwise_idx == max_spanwise)

                    # Process each filament of the vortex ring
                    for k in range(4):
                        vertex_start = np.array(vertices[k])
                        vertex_end = np.array(vertices[(k + 1) % 4])
                        
                        r1 = control_point - vertex_end
                        r2 = control_point - vertex_start
                        r0 =  r1 - r2

                        # Calculate induced velocity contribution
                        induced_velocity = self.biot_savart(
                            r1, r2, r0, gamma=1.0
                        )
                        total_induced_velocity += induced_velocity

                    if is_trailing_edge:
                        # Define infinity factor
                        radius_vector = control_point - hub_position
                        effective_omega = omega_dict[propeller_key]
                        omega_cross_r = -np.cross(effective_omega, radius_vector)


                        tangential_vectors = propeller_data['Blades'][vr_blade_key]['Tangential Vectors']
                        tangent_span = tangential_vectors[vr_index]['Tangential j']
                        tangent_norm = tangent_span / np.linalg.norm(tangent_span)
                        rot_vel_norm = omega_cross_r / np.linalg.norm(omega_cross_r)

                        # Combine them to get helical direction
                        # You can adjust these weights to change helix pitch
                        axial_weight = 1.0  
                        tangential_weight = 1.0
                        rotational_weight = 10.0

                        infinity_factor = 1000  # Adjust based on domain size

                        
                        trailing_direction = tangent_span / np.linalg.norm(tangent_span)
                        # trailing_direction = np.array([trailing_direction[0], trailing_direction[1], 0.0])
                        # trailing_direction = np.array([1,0,0])
                        # trailing_direction = (axial_weight * tangent_norm + 
                        # tangential_weight * tangent_norm +
                        # rotational_weight * rot_vel_norm)
                        # trailing_direction = tangent_span / np.linalg.norm(tangent_span)
                        # trailing_direction = trailing_direction / np.linalg.norm(trailing_direction)
        
                        new_vertices = [
                            np.array(vertices[1]),                                      # Point 0 (original v1)
                            np.array(vertices[1]) + infinity_factor * trailing_direction,  # Point 1 (v1 at infinity)
                            np.array(vertices[2]) + infinity_factor * trailing_direction,  # Point 2 (v2 at infinity)
                            np.array(vertices[2])                                # Point 3 (original v2)
                        ]

                        
                        for k in range(4):
                            vertex_start = new_vertices[k]
                            vertex_end = new_vertices[(k + 1) % 4]
                            
                            r1 = control_point - vertex_end
                            r2 = control_point - vertex_start
                            r0 =  vertex_start - vertex_end
                            
                            # Calculate induced velocity contribution
                            induced_velocity = self.biot_savart(
                                r1, r2, r0, gamma=1.0
                            )
                            total_induced_velocity += (induced_velocity)
                        
        

                    # Store in global matrix
                    global_matrix[i, j] = total_induced_velocity
                    # print(total_induced_velocity)
            global_matrices[propeller_key] = global_matrix

        return global_matrices
    
    def calculate_gamma(self, quad_propeller_mesh, bound_to_bound_global_matrices, 
                   wake_to_bound_induced_velocity_matrices, omega_dict, 
                   body_velocity, wind_field, com_position, time_step, roll, pitch, yaw):
        """ Calculate gamma (circulation strength) for each propeller using the Neumann boundary condition. """
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
        induced_velocities = {}
 
        # Get wind function
        # wind_func = WindField.update_wind_function(wind_field, com_position)
        # Define simplified wake effect parameters
        wind_velocity = wind_field
        wake_effect_strength = 0.3  # Adjust this value based on desired wake effect strength
        wake_direction = np.array([1, 0, 0])  
        decay_length = 0.1
        
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            # Get angular velocity for this propeller
            effective_omega = omega_dict[propeller_key]
            # hub_position = np.array(propeller_data['Hub Position'])
            hub_position = np.array(propeller_data['Hub Position'])
            # Collect data for control points
            control_points = []
            normals = []
            rhs = []

            for blade_key, blade_data in propeller_data['Blades'].items():
                for cp_index, control_point in blade_data['Control Points'].items():
                    control_point = np.array(control_point)
                    
                    # wind_velocity = -wind_func(control_point)
                    # print(wind_velocity)
                    # Rotate normal vector
                    normal = blade_data['Normals'][cp_index]
                    rotated_normal =  normal
                    normals.append(rotated_normal)
                    
                    if "RHS" in blade_data and cp_index in blade_data['RHS']:
                        rhs_value = blade_data['RHS'][cp_index]
                    else:
   
                        radius_vector = control_point - hub_position
                        omega_cross_r = np.cross(effective_omega, radius_vector)
            
                        if "Wind Velocity" not in blade_data:
                            blade_data['Wind Velocity'] = {}
                        blade_data['Wind Velocity'][cp_index] = wind_velocity

                        # Store omega x r
                        if "Omega_Cross_R" not in blade_data:
                            blade_data['Omega_Cross_R'] = {}
                        blade_data['Omega_Cross_R'][cp_index] = omega_cross_r 

                        # Calculate velocity 
                        velocity_term = -omega_cross_r + wind_velocity
                
                        rhs_value = -np.dot(velocity_term, normal)

                        if "RHS" not in blade_data:
                            blade_data['RHS'] = {}
                        blade_data['RHS'][cp_index] = rhs_value

                    rhs.append(rhs_value)
                    control_points.append(control_point)
            
            # Convert to numpy arrays
            normals = np.array(normals)
            rhs = np.array(rhs).reshape(-1, 1)

            # Get influence matrix for this propeller
            bound_to_bound_induced_matrix = bound_to_bound_global_matrices[propeller_key]
         
            Ab = np.einsum('ijk,ik->ij', bound_to_bound_induced_matrix, normals)
            # print("bound_to_bound_induced_matrix shape:", bound_to_bound_induced_matrix.shape)
            # print("normals shape:", normals.shape) 
            
            # if time_step>2:
            #     wake_to_bound_induced_velocity_matrix = wake_to_bound_induced_velocity_matrices[propeller_key]
            #     wake_influence = wake_to_bound_induced_velocity_matrix
            #     Aw = np.einsum('ij,ij->i', wake_to_bound_induced_velocity_matrix, normals).reshape(-1, 1)
            #     # print('bound_to_bound_induced_matrix Velocities:', bound_to_bound_induced_matrix)
            #     # print('wake_to_bound_induced_velocity_matrix Velocities:', wake_to_bound_induced_velocity_matrix)
            # else:    
            #     Aw = np.zeros_like(rhs)
            
            gamma = np.linalg.solve(Ab, (rhs))
            
            # print("gamma shape:", gamma.shape)
            gamma_matrices[propeller_key] = gamma.flatten()

            induced_vel = np.einsum('ijk,j->ik', bound_to_bound_induced_matrix, gamma.flatten())
            induced_velocities[propeller_key] = induced_vel
            
            # Update gamma values in the mesh
            gamma_index = 0
            for blade_key, blade_data in propeller_data['Blades'].items():
                blade_data['Gamma'] = {}
                blade_data['Induced_Velocities'] = {}  # Add new field for induced velocities
                blade_data['Wake_Induced_Velocities'] = {}
                for cp_index in blade_data['Control Points'].keys():
                    blade_data['Gamma'][cp_index] = float(gamma[gamma_index])
                    blade_data['Induced_Velocities'][cp_index] = induced_vel[gamma_index]
                    
                    # if time_step > 2:
                    #     # Get wake-induced velocities for this control point
                    #     wake_induced_vel = wake_to_bound_induced_velocity_matrices[propeller_key][gamma_index]
                    #     blade_data['Wake_Induced_Velocities'][cp_index] = wake_induced_vel

                    gamma_index += 1

            # return gamma_matrices, induced_velocities
 
    def pressure_difference(self, quad_propeller_mesh, bound_to_bound_global_matrices, wake_to_bound_induced_velocity_matrices, body_velocity, omega, time_step, dt, rho):
        """
        Calculate the pressure difference for each panel.
        More efficient implementation while maintaining exact calculations.
        
        """
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            # Get induced velocity matrix for current propeller
            induced_velocity_matrix = bound_to_bound_global_matrices[propeller_key]
            # wake_to_bound_induced_velocity_matrix = wake_to_bound_induced_velocity_matrices[propeller_key]

            for blade_key, blade_data in propeller_data['Blades'].items():
                control_points = blade_data['Control Points']
                normals = blade_data['Normals']
                tangential_vectors = blade_data['Tangential Vectors']
                gamma_old = blade_data.get('Gamma Old', {})
                pressure_difference = {}

                for panel_index, control_point in control_points.items():

                    # Get panel vectors and properties
                    normal = normals[panel_index]
                    tangent_span = tangential_vectors[panel_index]['Tangential i']
                    tangent_chord = tangential_vectors[panel_index]['Tangential j']
                    
                    # Get rotational velocity term
                    omega_cross_r = blade_data['Omega_Cross_R'][panel_index]
                    induced_velocity = blade_data['Induced_Velocities'][panel_index]
                    wind_velocity = blade_data['Wind Velocity'][panel_index]

                    # if time_step>2:
                    #     wake_to_bound_induced_velocity = blade_data['Wake_Induced_Velocities'][panel_index]
                    # else:    
                    #     wake_to_bound_induced_velocity = np.zeros_like(omega_cross_r)

                    # tangent_span = panel_array[3] - panel_array[0] 
                    # tangent_chord = panel_array[1] - panel_array[0]

                    # Get current and previous circulation values
                    gamma_current = blade_data['Gamma'][panel_index]
                    gamma_previous_span = blade_data['Gamma'].get((panel_index[0] - 1, panel_index[1]), 0) if panel_index[0] > 0 else 0
                    gamma_previous_chord = blade_data['Gamma'].get((panel_index[0], panel_index[1] - 1), 0) if panel_index[1] > 0 else 0

                    # Calculate gamma differences (normalized)
                    gamma_diff_span = (gamma_current - gamma_previous_span) / np.linalg.norm(tangent_chord)
                    gamma_diff_chord = (gamma_current - gamma_previous_chord) / np.linalg.norm(tangent_span)
                                  
                    # Calculate the total velocity (freestream + induced + rotational)
                    total_velocity =  -omega_cross_r + wind_velocity
                    panel = blade_data['Panels'][panel_index]
                    panel_array = np.array(panel)

                    panel_center = panel_array.mean(axis=0)
                    area_triangle_1 = 0.5 * np.linalg.norm(np.cross(panel_array[1] - panel_array[0], panel_array[3] - panel_array[0]))
                    area_triangle_2 = 0.5 * np.linalg.norm(np.cross(panel_array[2] - panel_array[1], panel_array[3] - panel_array[1]))

                    panel_area = area_triangle_1 + area_triangle_2

                    if gamma_old == {}:
                        gamma_previous = 0
                    else:
                        gamma_previous = gamma_old[panel_index]       
                    
                    gamma_dot = (gamma_current - gamma_previous) / dt

                    # force_1 = rho * (gamma_current-gamma_previous_span) * np.linalg.cross(total_velocity, tangent_span)
                    # force_2 = rho * (gamma_current-gamma_previous_chord) * np.linalg.cross(total_velocity, tangent_chord)
                    # force_steady = force_1 + force_2
                    # force_unsteady = rho * gamma_dot *  panel_area

                    # spanwise_index = max(idx[0] for idx in blade_data['Vortex Rings'].keys())
                    # if time_step > 2:
                    #     if spanwise_index == panel_index[0]:
                    #         chordwise_index = panel_index[1]
                    #         bound_gamma = blade_data['Gamma'][panel_index]
                    #         wake_gamma = self.wake_system[propeller_key][blade_key][(0, chordwise_index)]['Gamma']
                    #         net_gamma = bound_gamma - wake_gamma
                    #         span = panel_array[2] - panel_array[1] 
                    #         force = rho * net_gamma * np.linalg.cross(total_velocity, span)
                    #         force_steady = force_steady + force

                    pressure = rho * (
                        np.dot(total_velocity, tangent_chord / np.linalg.norm(tangent_chord)  * gamma_diff_span) +
                        np.dot(total_velocity, tangent_span / np.linalg.norm(tangent_span) * gamma_diff_chord)     
                    )
                    # pressure = (np.cross(rho *  total_velocity * (gamma_current - gamma_previous_span), (tangent_span)))
                    # pressure2 = (np.cross(rho *  total_velocity * (gamma_current - gamma_previous_chord), (tangent_chord)))
                    # pressure = (rho * np.linalg.norm(total_velocity) * (gamma_current - gamma_previous_span) * np.linalg.norm(tangent_span))
                    # pressure = -2 * ((total_velocity + induced_velocity) / total_velocity)**2
                    # pressure = np.linalg.norm(total_velocity) * rho * (gamma_current) * np.linalg.norm(tangent_span)
                    # pressure = rho * (gamma_current-gamma_previous_span) * np.cross((total_velocity + induced_velocity), tangent_span)
                    # pressure2 = rho * (gamma_current-gamma_previous_chord) * np.cross((total_velocity + induced_velocity), tangent_chord)
            
                    
                    pressure_difference[panel_index] = pressure 
                
                # Update blade data with pressure difference and store old gamma values
                blade_data['Pressure Difference'] = pressure_difference
                blade_data['Gamma Old'] = blade_data['Gamma'].copy()

        # For debugging and verification
        # print("Pressure Difference", quad_propeller_mesh['Propeller_1']['Blades']['Blade_1']['Gamma'])
        # print("Pressure Difference", quad_propeller_mesh['Propeller_1']['Blades']['Blade_2']['Pressure Difference'])
        # print("-"*100)
        # print("Pressure Difference", quad_propeller_mesh['Propeller_2']['Blades']['Blade_1']['Gamma'])
        # # print("Pressure Difference", quad_propeller_mesh['Propeller_2']['Blades']['Blade_2']['Pressure Difference'])
        # print("-"*100)

    def calculate_total_forces_and_moments(self, propeller_mesh, dt, time_step, rho, body_velocity, omega, wind_field, com_position, roll, pitch, yaw):
        """
        Calculate aerodynamic forces and moments for each panel of each propeller
        using UVLM. This includes updating the pressure difference for each panel.
        """
        # Step 1: Compute the global induced velocity matrix
        bound_to_bound_global_matrices = self.calculate_bound_to_bound_induced_velocity_matrix(propeller_mesh, omega)

        wake_to_bound_induced_velocity_matrices = None
        
        # # Only calculate wake influences after time step 2
        # if time_step > 2:
        #     wake_to_bound_induced_velocity_matrices = self.calculate_wake_to_bound_induced_velocity_matrix(propeller_mesh)

        # Step 2: Calculate gamma for each panel
        # self.calculate_gamma(
        #     propeller_mesh,
        #     bound_to_bound_global_matrices,
        #     wake_to_bound_induced_velocity_matrices,
        #     omega,
        #     body_velocity,
        #     wind_field, 
        #     com_position, 
        #     time_step,
        #     roll=roll,
        #     pitch=pitch,
        #     yaw=yaw
        # )

        # # Step 3: Update the pressure differences for all panels
        # self.pressure_difference(
        #     propeller_mesh,
        #     bound_to_bound_global_matrices,
        #     wake_to_bound_induced_velocity_matrices,
        #     body_velocity,
        #     omega,
        #     time_step,
        #     dt,
        #     rho
        # )

        # Initialize dictionary to store total forces and moments for each propeller
        total_forces_and_moments = {}
        total_forces_and_moments_all = {}
        total_force_all = np.zeros(3)
        total_moment_all = np.zeros(3)
        # Step 4: Calculate forces and moments for each panel
        for propeller_key, propeller_data in propeller_mesh.items():
            hub_position = np.array(propeller_data['Hub Position'])
            # Initialize total force and moment for this propeller
            total_force = np.zeros(3)
            total_moment = np.zeros(3)
            total_body_moment = np.zeros(3)
  
            for blade_key, blade_data in propeller_data['Blades'].items():
                # Initialize storage for panel forces and moments
                blade_data['Panel Forces'] = {}
                blade_data['Panel Moments'] = {}
                blade_data['Body Moments'] = {}

                total_area=0.0
                for panel_index, pressure in blade_data['Pressure Difference'].items():
                    # Panel geometry
                    panel = blade_data['Panels'][panel_index]
                    panel_array = np.array(panel)
                    panel_center = panel_array.mean(axis=0)
                    control_point = blade_data['Control Points'][panel_index]
                    
                    # Calculate the normal vector and panel area
                    normal = blade_data['Normals'][panel_index]
                    
                    area_triangle_1 = 0.5 * np.linalg.norm(np.cross(panel_array[1] - panel_array[0], panel_array[3] - panel_array[0]))
                    area_triangle_2 = 0.5 * np.linalg.norm(np.cross(panel_array[2] - panel_array[1], panel_array[3] - panel_array[1]))

                    # panel_area =  0.5 * np.linalg.norm(
                    #     np.cross(panel_array[2] - panel_array[0], panel_array[3] - panel_array[1])
                    # )
                    panel_area = area_triangle_1 + area_triangle_2
                    total_area += panel_area
                    # Calculate vectors for Triangle 1
                    
                    # Force contribution for this pane
                    # panel_force = (pressure)
                    panel_force = np.dot(pressure * panel_area, normal)
                    
                    moment_arm_local = panel_center - hub_position
                    # moment_local = np.cross(moment_arm_local, force)
                   
                    # # Moment contribution for this panel
                    moment_arm_global = control_point 
                    # moment_global = np.cross(moment_arm_global, force)
                    # moment = np.cross(moment_arm, force)

                    # moment_arm_total = control_point - hub_position  # Vector from origin to force application point
                    panel_moment = np.cross(moment_arm_local, panel_force)
                    body_moment =np.cross(moment_arm_global, panel_force)
                    
                    # Store the force and moment for this panel
                    blade_data['Panel Forces'][panel_index] = panel_force
                    blade_data['Panel Moments'][panel_index] = panel_moment
                    blade_data['Body Moments'][panel_index] = body_moment


                    # Accumulate to total force and moment
                    total_force += panel_force
                    total_moment += panel_moment
                    total_body_moment+= body_moment
                    # print('panel_index', panel_index)
                    # print(panel_force)
            # total_force_all += total_force
            # total_moment_all += total_moment
            # Store total force and moment for this propeller
            print(total_force)
            total_forces_and_moments[propeller_key] = {'force': total_force, 'moment': total_moment, 'total_body_moment': total_body_moment}
        # total_forces_and_moments_all = {'total_force': total_force_all, 'total_moment': total_moment_all}

        return  total_forces_and_moments
           
    def viscous_coupling(self, quad_propeller_mesh, omega_dict, 
                        body_velocity, wind_field, com_position, time_step, dt, roll, pitch, yaw,
                        rho=1.07178, relaxation_factor=0.05, max_iterations=1, convergence_tolerance=1e-4):
            """
            Implement viscous coupling using the Kutta-Joukowski theorem and iterative coupling
            between the UVLM solution and airfoil polar data.
            """
            # Store convergence history for diagnostics
            convergence_history = []
            
            # Step 1: Initial calculation of circulation strength
            # Calculate bound-to-bound influence matrix
            bound_to_bound_global_matrices = self.calculate_bound_to_bound_induced_velocity_matrix(quad_propeller_mesh, omega_dict)
            wake_to_bound_induced_velocity_matrices = None

            if time_step > 2:
                wake_to_bound_induced_velocity_matrices = self.calculate_wake_to_bound_induced_velocity_matrix(quad_propeller_mesh, dt, time_step)
            
            # Initial gamma calculation without corrections
            self.calculate_gamma(
                quad_propeller_mesh,
                bound_to_bound_global_matrices,
                wake_to_bound_induced_velocity_matrices,
                omega_dict,
                body_velocity,
                wind_field,
                com_position,
                time_step,
                roll,
                pitch,
                yaw
            )

            # Initialize storage for viscous coupling
            tip_radius = 0.11938  # Propeller tip radius (adjust as needed)
            
            # Process each propeller
            for propeller_key, propeller_data in quad_propeller_mesh.items():
                hub_position = np.array(propeller_data['Hub Position'])
                omega_vector = np.array(omega_dict[propeller_key])
                propeller_axis = np.array([0, 0, 1])  # Assuming Z-axis is along propeller shaft
                
                # Process each blade
                for blade_key, blade_data in propeller_data['Blades'].items():
                    print("----Processing Blade 1----")
                    spanwise_sections = {}
                    for cp_index, gamma_value in blade_data['Gamma'].items():
                        spanwise_idx = cp_index[1]  
                        if spanwise_idx not in spanwise_sections:
                            spanwise_sections[spanwise_idx] = []
                        spanwise_sections[spanwise_idx].append(cp_index)
        
                    if 'Section_Inflow' not in blade_data:
                        blade_data['Section_Inflow'] = {}

                    previous_rhs = None
                    previous_residual = None
                    alpha_aitken = relaxation_factor
                    vis_for =0.0
                    for spanwise_idx, cp_indices in spanwise_sections.items():
                        print("-----Spanwise section-----", spanwise_idx)
                        
                        # Choose representative control point (middle of section)
                        rep_idx = cp_indices[len(cp_indices)//2]
                        print('rep_idx', rep_idx)
                        rep_control_point = blade_data['Control Points'][rep_idx]
                    
                        # Calculate basic geometric parameters
                        cp_position = np.array(rep_control_point)
                        radius_vector = cp_position - hub_position
                        print('radius vector', radius_vector)
                        radial_proj = radius_vector - np.dot(radius_vector, propeller_axis) * propeller_axis
                        r = np.linalg.norm(radius_vector)
                        r_R = r / tip_radius
                        
                        # Get twist angle and chord
                        twist_rad = np.deg2rad(blade_data['Twist'][rep_idx])
                        
                        # chord = PropellerGeometry.chord_spline(r_R)
                        panel = np.array(blade_data['Panels'][rep_idx])
                        section_length = panel[3] - panel[0]

                        # Calculate direction vectors
                        radial_dir = radial_proj / np.linalg.norm(radial_proj)
                        tangential_dir = np.cross(propeller_axis, radial_dir)
                        tangential_dir = tangential_dir / np.linalg.norm(tangential_dir)

                        # Calculate section properties
                        section_area = 0.0
                        chord = 0.0 
                        for cp_index in cp_indices:
                            panel = np.array(blade_data['Panels'][cp_index])
                            area_triangle_1 = 0.5 * np.linalg.norm(np.cross(panel[1] - panel[0], panel[3] - panel[0]))
                            area_triangle_2 = 0.5 * np.linalg.norm(np.cross(panel[2] - panel[1], panel[3] - panel[1]))
                            panel_area = area_triangle_1 + area_triangle_2
                            section_area += panel_area
                            
                            # Calculate chord length (average of leading and trailing edges)
                            section_chord = np.linalg.norm(panel[1] - panel[0])
                            chord += section_chord

                        total_gamma = sum(blade_data['Gamma'][cp_idx] for cp_idx in cp_indices)
                        avg_gamma = total_gamma / len(cp_indices)
                        gamma_table_previous = avg_gamma
                        gamma_table = avg_gamma
                        alpa_local = 0.0
                        # EXTRACT ZERO-LIFT ANGLE FROM POLAR DATA
                        print('r_R', r_R)
                        polar = blade_data['Polar Data'][rep_idx]
                        if polar is not None and 'alpha' in polar and 'cl' in polar:
                            alpha_arr = np.array(polar['alpha'])
                            cl_arr = np.array(polar['cl'])
                            
                            alpha_0 = 0.0  # Default zero-lift angle
                            
                            # Method 1: Find where cl is closest to zero (simplest)
                            zero_lift_idx = np.argmin(np.abs(cl_arr))
                            alpha_0_nearest = alpha_arr[zero_lift_idx]
                            
                            # Method 2: Linear interpolation to find exact zero-lift angle (more accurate)
                            alpha_0_interpolated = None
                            
                            # Find two points that bracket cl = 0
                            for i in range(len(cl_arr) - 1):
                                cl_1 = cl_arr[i]
                                cl_2 = cl_arr[i + 1]
                                
                                # Check if zero is bracketed between these points
                                if (cl_1 <= 0 <= cl_2) or (cl_2 <= 0 <= cl_1):
                                    alpha_1 = alpha_arr[i]
                                    alpha_2 = alpha_arr[i + 1]
                                    
                                    # Linear interpolation: α_0 = α_1 + (α_2 - α_1) * (0 - cl_1) / (cl_2 - cl_1)
                                    if abs(cl_2 - cl_1) > 1e-10:  # Avoid division by zero
                                        alpha_0_interpolated = alpha_1 + (alpha_2 - alpha_1) * (0 - cl_1) / (cl_2 - cl_1)
                                        break
                            
                            # Method 3: Linear regression through the linear portion (most robust)
                            # Find linear region around cl = 0
                            linear_region_mask = np.abs(cl_arr) < 0.8  # Points with |cl| < 0.8
                            if np.sum(linear_region_mask) >= 2:
                                alpha_linear = alpha_arr[linear_region_mask]
                                cl_linear = cl_arr[linear_region_mask]
                                
                                # Linear fit: cl = slope * alpha + intercept
                                # Zero-lift angle: alpha_0 = -intercept / slope
                                coeffs = np.polyfit(alpha_linear, cl_linear, 1)  # [slope, intercept]
                                slope = coeffs[0]
                                intercept = coeffs[1]
                                
                                if abs(slope) > 1e-10:
                                    alpha_0_regression = -intercept / slope
                                else:
                                    alpha_0_regression = None
                            
                            # Choose the best method (priority: interpolated > regression > nearest)
                            if alpha_0_interpolated is not None:
                                alpha_0 = alpha_0_interpolated
                                print(f"Zero-lift angle (interpolated): {alpha_0:.3f}°")
                            elif 'alpha_0_regression' in locals() and alpha_0_regression is not None:
                                alpha_0 = alpha_0_regression  
                                print(f"Zero-lift angle (regression): {alpha_0:.3f}°")
                            else:
                                alpha_0 = alpha_0_nearest
                                print(f"Zero-lift angle (nearest): {alpha_0:.3f}°")
                        else:
                            alpha_0 = 0.0  # Default for symmetric airfoil
                            print("No polar data available, assuming symmetric airfoil (α₀ = 0°)")

                        # Start iterative process for this section
                        converged = False
                        current_alpha = 0.0  # Initial angle of attack      
                        
                        for iteration in range(max_iterations):     

                            # Calculate inflow velocity components
                            rotational_velocity = np.cross(omega_vector, radius_vector)
                            induced_velocity = blade_data['Induced_Velocities'][rep_idx]
                            
                            wake_induced_velocity = np.zeros(3)
                            if time_step > 2:
                                wake_induced_velocity = blade_data['Wake_Induced_Velocities'][rep_idx]

                            # Total inflow velocity
                            total_inflow = -rotational_velocity + induced_velocity + wake_induced_velocity 
                            total_velocity = -rotational_velocity

                            # Project onto blade plane
                            axial_component = np.dot(total_inflow, propeller_axis)
                            tangential_component = np.dot(total_inflow, tangential_dir)
   
                             # Calculate inflow angle
                            V_mag = np.linalg.norm(total_inflow)
                            if V_mag > 1e-6:
                                alpha_inflow = np.arctan2(abs(axial_component), abs(tangential_component))
                            else:
                                alpha_inflow = 0.0

                            if iteration==0:
                                alpha_3d = alpha_inflow
                                alpha_local = alpha_3d
                
                            # Get current total gamma for this section
                            total_gamma = sum(blade_data['Gamma'][cp_idx] for cp_idx in cp_indices)

                            total_force=0.0
                            section_area=0.0
                            cl_inv = 0.0
                            for cp_index in cp_indices:
                                print('span', spanwise_idx, 'and', cp_index)
                                normal = blade_data['Normals'][cp_index]
                                tangential_vectors = blade_data['Tangential Vectors']
                                tangent_span = tangential_vectors[cp_index]['Tangential i']
                                tangent_chord = tangential_vectors[cp_index]['Tangential j']
                                induced = blade_data['Induced_Velocities'][cp_index]
                                rotational = blade_data['Omega_Cross_R'][cp_index]
                                total_velocity_in = -rotational
                                total_inflow_in = -rotational + induced
                                cp_position_in = blade_data['Control Points'][cp_index]
                                
                                twist_rad = np.deg2rad(blade_data['Twist'][cp_index])
                                
                                radius_vector = cp_position_in - hub_position
                                print('radius vector', radius_vector)
                                radial_proj = radius_vector - np.dot(radius_vector, propeller_axis) * propeller_axis
                                r = np.linalg.norm(radius_vector)
                                r_R = r / tip_radius

                                # Calculate direction vectors
                                radial_dir = radial_proj / np.linalg.norm(radial_proj)
                                tangential_dir = np.cross(propeller_axis, radial_dir)
                                tangential_dir = tangential_dir / np.linalg.norm(tangential_dir)

                                # Project onto blade plane
                                axial_component = np.dot(total_inflow_in, propeller_axis)
                                tangential_component = np.dot(total_inflow_in, tangential_dir)
                                print('axia component', axial_component)
                                print('tangential component', tangential_component)

                                # Calculate inflow angle
                                V_mag = np.linalg.norm(total_inflow_in)
                                if V_mag > 1e-6:
                                    alpha_inflow = np.arctan2(abs(axial_component), abs(tangential_component))
                                else:
                                    alpha_inflow = 0.
                                
                                gamma_current = blade_data['Gamma'][cp_index]
                                gamma_previous_span = blade_data['Gamma'].get((cp_index[0] - 1, cp_index[1]), 0) if cp_index[0] > 0 else 0
                                gamma_previous_chord = blade_data['Gamma'].get((cp_index[0], cp_index[1] - 1), 0) if cp_index[1] > 0 else 0

                                gamma_diff_span = (gamma_current - gamma_previous_span) / np.linalg.norm(tangent_chord)
                                gamma_diff_chord = (gamma_current - gamma_previous_chord) / np.linalg.norm(tangent_span)

                                pressure = rho * (
                                    np.dot(total_velocity_in, tangent_chord / np.linalg.norm(tangent_chord)  * gamma_diff_span) +
                                    np.dot(total_velocity_in, tangent_span / np.linalg.norm(tangent_span) * gamma_diff_chord)     
                                )
                                
                                # total_gamma += blade_data['Gamma'][cp_index]
                                panel = np.array(blade_data['Panels'][cp_index])
                                area_triangle_1 = 0.5 * np.linalg.norm(np.cross(panel[1] - panel[0], panel[3] - panel[0]))
                                area_triangle_2 = 0.5 * np.linalg.norm(np.cross(panel[2] - panel[1], panel[3] - panel[1]))
                                panel_area = area_triangle_1 + area_triangle_2
                                section_area += panel_area  

                                panel_force = (pressure * panel_area * normal)
                                # panel_force = total_velocity_in * rho * np.linalg.norm(tangent_span) * (gamma_current - gamma_previous_span)
                                # total_inflow = -rotational + induced 
                                # inflow_velocity_dir = (total_inflow) / np.linalg.norm(total_inflow)
                                # lift_vector = panel_force - np.dot(panel_force, inflow_velocity_dir) * inflow_velocity_dir
                                # total_lift = np.linalg.norm(lift_vector)
                                # cl_inv += 2 * np.linalg.norm(panel_force[2]) / (rho * (np.linalg.norm(total_velocity)**2) *  panel_area)
                                # panel_force = rho * (gamma_current - gamma_previous_span) * np.cross((-rotational + induced), tangent_span)
                                # print('panel force', panel_force)
                                inflow_velocity_dir = (total_inflow_in) / np.linalg.norm(total_inflow_in)
                                total_force += panel_force
                                panel_lift = panel_force - np.dot(panel_force, inflow_velocity_dir) * inflow_velocity_dir
                                total_lift = np.linalg.norm(panel_lift)

                                cl_inv = 2 * np.linalg.norm(total_lift) / (rho * (np.linalg.norm(total_inflow_in)**2) *  panel_area)
                                alpha = np.rad2deg(cl_inv / (2 * np.pi) ) 
                                polar = blade_data['Polar Data'][cp_index]
                                if polar is not None:
                                    alpha_arr = np.array(polar['alpha'])
                                    cl_arr = np.array(polar['cl'])
                                    cd_arr = np.array(polar['cd'])
                                    
                                    # Find closest angle in polar data
                                    alpha_idx = np.argmin(np.abs(alpha_arr - alpha))
                                    cl_vis = cl_arr[alpha_idx]
                                    cd_vis = cd_arr[alpha_idx]
                                print('control points', cp_position_in)
                                print('total_velocity_in', total_velocity_in)
                                print('total_inflow_in', total_inflow_in)
                                print('induced', induced)
                                print('radius vector', radius_vector)
                                print('alpha_inflow', np.rad2deg(alpha_inflow))
                                print('alpha', alpha)
                                print('twist', np.rad2deg(twist_rad))
                                print('panel_force', panel_force)
                                print('cl_inv', cl_inv)
                                print('cl_vis', cl_vis)

                            

                                print('=================')

                            inflow_velocity_dir = (total_inflow) / np.linalg.norm(total_inflow)
                            # panel_force = np.dot(pressure * panel_area, normal)
                            # print('panle force', panel_force)
                            # print('panle area', pane/l_area)
                            # print('inflow_vleocity', total_inflow)
                            # total_force1 = rho * (gamma - gamma_previous_span) * (np.cross(inflow_velocity, section_length))
                            # print('total_force', total_force)
                            
                            lift_vector = total_force - np.dot(total_force, inflow_velocity_dir) * inflow_velocity_dir
                            total_lift = np.linalg.norm(lift_vector)

                            cl_inv = 2 * np.linalg.norm(total_lift) / (rho * (np.linalg.norm(total_inflow)**2) *  section_area)
                            
                            # cl_inv = 2 * total / ((np.linalg.norm(total_velocity)) *  np.linalg.norm(section_length))
                            # Calculate lift coefficient from circulation (Kutta-Joukowski)
                            # if V_mag > 1e-6 and section_chord > 1e-6:
                            #     cl_inv = 2 * gamma_table / (V_mag * np.linalg.norm(section_length))
                            # else:
                            #     cl_inv = 0.0
                            
                            # Calculate effective angle of attack
                            alpha_eff = (cl_inv / (2 * np.pi) ) 
                            alpha_eff_2 = - alpha_inflow + twist_rad
                            alpha_eff_deg = np.rad2deg(alpha_eff)
                            alpha_eff_deg_2 = np.rad2deg(alpha_eff_2)
                            twist_deg = np.rad2deg(twist_rad)
                            polar = blade_data['Polar Data'][rep_idx]
                            if polar is not None:
                                alpha_arr = np.array(polar['alpha'])
                                cl_arr = np.array(polar['cl'])
                                cd_arr = np.array(polar['cd'])
                                
                                # Find closest angle in polar data
                                alpha_idx = np.argmin(np.abs(alpha_arr - alpha_eff_deg))
                                cl_vis = cl_arr[alpha_idx]
                                cd_vis = cd_arr[alpha_idx]
                            cl_flow = cl_vis * np.cos(alpha_inflow) - cd_vis * np.sin(alpha_inflow)
                            cd_flow = cl_vis * np.sin(alpha_inflow) + cd_vis * np.cos(alpha_inflow)

                            

                            numerator = 0.5 * (axial_component**2 + tangential_component**2) * (cl_vis * section_area)
                            denominator = np.sqrt( (np.dot(np.cross(total_inflow, section_length), tangential_dir))**2 +
                                                  (np.dot(np.cross(total_inflow, section_length), propeller_axis))**2
                                                  )
                            gamma_table = numerator / denominator
                            viscous =  0.5  * rho * (np.linalg.norm(total_inflow)**2) *  section_area * cl_vis
                            print('viscous', viscous)
                            # Calculate residual
                            residual = np.abs(gamma_table_previous - gamma_table)
                            alpha_local = alpha_local + relaxation_factor * ( (cl_inv - cl_vis) / (2 * np.pi) )
                            
                            print('total_lift', total_lift)
                            print('total_inflow', total_inflow) 
                            print('axial_component', axial_component) 
                            print('tangential_component', tangential_component) 
                            print('cl_vis', cl_vis) 
                            print('cl_flow', cl_flow) 
                            print('gamma_table', gamma_table) 
                            print('avg_gamma', avg_gamma) 
                            print('alpha_eff_deg_2', alpha_eff_deg_2) 
                            print('alpha_eff_deg', alpha_eff_deg) 
                            print('alpha_inflow', np.rad2deg(alpha_inflow) )
                            print('twist', np.rad2deg(twist_rad) )
                            print('cl_inv', cl_inv)
                            print('induced_velocity', induced_velocity ) 
                            print('total_velocity', total_velocity ) 

                            print('residual', residual)
                            print('thrust', total_force[2])
              
                            # Check convergence
                            if abs(residual) < convergence_tolerance:
                                converged = True
                                break

                            # # Update angle of attack using relaxation
                            # if iteration == 0 or previous_residual is None:
                            #     gamma_table = avg_gamma + relaxation_factor * (gamma_table - avg_gamma)
                            # else:
                            #     gamma_place = gamma_table
                            #     gamma_table = gamma_table + relaxation_factor * (gamma_table - gamma_table_previous)
                            #     gamma_table_previous = gamma_place

                            alpha_local = alpha_local + relaxation_factor * ((cl_inv  - cl_vis) / (2*np.pi))

                            # total_induced_velocity = np.zeros(3)
                            # vortex_ring = blade_data['Vortex Rings'][rep_idx]
                            # vertices = vortex_ring['Vertices']
                            # # Process each filament of the vortex ring
                            # for k in range(4):
                            #     vertex_start = np.array(vertices[k])
                            #     vertex_end = np.array(vertices[(k + 1) % 4])
                                
                            #     r1 = rep_control_point - vertex_end
                            #     r2 = rep_control_point - vertex_start
                            #     r0 =  r1 - r2

                            #     # Calculate induced velocity contribution
                            #     self_induced_velocity = self.biot_savart(
                            #         r1, r2, r0, gamma=1.0
                            #     )
                            #     total_induced_velocity += self_induced_velocity

                            # blade_data['Gamma'][rep_idx] = gamma_table
                            # blade_data['Induced_Velocities'][rep_idx] = induced_velocity + total_induced_velocity * (gamma_table - gamma_table_previous)
                            # print('induced, velcotiy', induced_velocity)
                            print('==============================')
                            
                            # self.calculate_gamma(
                            # quad_propeller_mesh,
                            # bound_to_bound_global_matrices,
                            # wake_to_bound_induced_velocity_matrices,
                            # omega_dict,
                            # body_velocity,
                            # wind_field,
                            # com_position,
                            # time_step,
                            # roll,
                            # pitch,
                            # yaw
                            # )


                            # current_alpha += alpha_correction
                            # previous_residual = residual

                            # rotational_velocity = np.cross(omega_vector, radius_vector)
                            # total_inflow = -rotational_velocity + wind_field  # Base inflow
                            
                            # # Calculate lift direction (perpendicular to inflow, in blade plane)
                            # inflow_direction = total_inflow / np.linalg.norm(total_inflow) if np.linalg.norm(total_inflow) > 1e-6 else tangential_dir
                            # lift_direction = np.cross(inflow_direction, np.cross(propeller_axis, inflow_direction))
                            # lift_direction = lift_direction / np.linalg.norm(lift_direction) if np.linalg.norm(lift_direction) > 1e-6 else propeller_axis
                                   
                            # V_mag = np.linalg.norm(total_inflow) if np.linalg.norm(total_inflow) > 1e-6 else 1.0
                            # velocity_correction_magnitude = relaxation_factor * residual * V_mag / (2 * np.pi)
                            
                            # # Apply correction in the direction that affects angle of attack
                            # # Typically this is in the axial direction (changes inflow angle)
                            # velocity_correction_vector = velocity_correction_magnitude * propeller_axis
                            
                            # # Update RHS for all control points in this section
                            # for cp_index in cp_indices:

                            #     # Get the current RHS value
                            #     current_rhs = blade_data['RHS'][cp_index]
            
                            #     # Get normal for this control point
                            #     normal = blade_data['Normals'][cp_index]
                                
                            #     # Calculate RHS correction: Δ(RHS) = -n·(Δu)
                            #     rhs_correction = -np.dot(velocity_correction_vector, normal)
                                
                            #     # Apply incremental correction: RHS_i = RHS_i-1 + α · r
                            #     new_rhs_value = current_rhs + rhs_correction
                                
                            #     # Update the stored RHS value
                            #     blade_data['RHS'][cp_index] = new_rhs_value
              
                    if not converged:
                        print(f"Viscous coupling did not converge after {max_iterations} iterations")
                    else:
                        print("Viscous coupling completed successfully.")
                 
             
                # Update pressure differences with new circulation values
            self.pressure_difference(
                quad_propeller_mesh,
                bound_to_bound_global_matrices,
                wake_to_bound_induced_velocity_matrices,
                body_velocity,
                omega_dict,
                time_step,
                dt,
                rho
            )

            # Calculate final forces and moments after convergence or max iterations
            forces_and_moments = self.calculate_total_forces_and_moments(
                quad_propeller_mesh,
                dt,
                time_step,
                rho,
                body_velocity,
                omega_dict,
                wind_field,
                com_position,
                roll,
                pitch,
                yaw
            )
                     
            # Update wake with new circulation values
            # # This passes the updated circulation values to the wake system for time-marching
            # if time_step > 0:
            #     self.update_wake(
            #         quad_propeller_mesh, 
            #         time_step, 
            #         dt, 
            #         body_velocity, 
            #         omega_dict, 
            #         wind_field, 
            #         com_position
            #     )
                
            return forces_and_moments

    def plot_span_gamma_distribution(self, quad_propeller_mesh, fixed_radial_index):
        """
        Plot gamma distribution across all propellers and their blades
        for a fixed radial position while varying cp_index[1].
        """

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Gamma Distribution Across Propellers (Fixed Radial Index: {fixed_radial_index})')
        colors = ['b', 'r']
        axs = axs.flatten()

        for prop_idx, (propeller_key, propeller_data) in enumerate(quad_propeller_mesh.items()):
            ax = axs[prop_idx]
            
            for blade_idx, (blade_key, blade_data) in enumerate(propeller_data['Blades'].items()):
                # Extract gamma values and chordwise positions
                gamma_values = []
                chordwise_positions = []
                
                # Get values for fixed radial position
                for cp_index, control_point in blade_data['Control Points'].items():
                    if cp_index[0] == fixed_radial_index:  # Only take points at the fixed radial position
                        gamma = blade_data['Pressure Difference'][cp_index]
                        
                        gamma_values.append(gamma)
                        chordwise_positions.append(cp_index[1])
                        # print(cp_index[0])
                # Plot for this blade
                ax.plot(chordwise_positions, gamma_values, f'{colors[blade_idx]}o-',
                    label=f'{blade_key}', linewidth=2, markersize=8)
                
                ax.set_title(f'{propeller_key}')
                ax.set_xlabel('Chordwise Position (cp_index[1])')
                ax.set_ylabel('Gamma')
                ax.grid(True)
                ax.legend()

        plt.tight_layout()
        plt.show()

    def plot_chord_gamma_distribution(self, quad_propeller_mesh, fixed_radial_index):
        """
        Plot gamma distribution across all propellers and their blades
        for a fixed radial position while varying cp_index[1].
        """

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Gamma Distribution Across Propellers (Fixed Radial Index: {fixed_radial_index})')
        colors = ['b', 'r']
        axs = axs.flatten()

        for prop_idx, (propeller_key, propeller_data) in enumerate(quad_propeller_mesh.items()):
            ax = axs[prop_idx]
            
            for blade_idx, (blade_key, blade_data) in enumerate(propeller_data['Blades'].items()):
                # Extract gamma values and chordwise positions
                gamma_values = []
                chordwise_positions = []
                
                # Get values for fixed radial position
                for cp_index, control_point in blade_data['Control Points'].items():
                    if cp_index[1] == fixed_radial_index:  # Only take points at the fixed radial position
                        gamma = blade_data['Pressure Difference'][cp_index]
                        
                        gamma_values.append(gamma)
                        chordwise_positions.append(cp_index[0])
                        # print(cp_index[0])
                # Plot for this blade
                ax.plot(chordwise_positions, gamma_values, f'{colors[blade_idx]}o-',
                    label=f'{blade_key}', linewidth=2, markersize=8)
                
                ax.set_title(f'{propeller_key}')
                ax.set_xlabel('Chordwise Position (cp_index[1])')
                ax.set_ylabel('Gamma')
                ax.grid(True)
                ax.legend()

        plt.tight_layout()
        plt.show()

    def store_thrust(self, forces_and_moments, time_step):
        """
        Store thrust (z-component of force) for each time step.
        
        Args:
            forces_and_moments: Dictionary containing forces for each propeller
            time_step: Current simulation time step
        """
        # Initialize thrust history if it doesn't exist
        if not hasattr(self, 'thrust_history'):
            self.thrust_history = []
        
        # Extract z-component (thrust) from Propeller_1 force
        thrust = forces_and_moments['Propeller_1']['force'][2]
        
        # Store time step and thrust value
        self.thrust_history.append({
            'time_step': time_step,
            'thrust': thrust
        })
    
    def plot_thrust_history(self):
        """
        Plot the thrust history over time steps.
        """
        if not hasattr(self, 'thrust_history') or not self.thrust_history:
            print("No thrust history available. Run simulation first.")
            return
        
        # Extract time steps and thrust values
        time_steps = [data['time_step'] for data in self.thrust_history]
        thrust_values = [data['thrust'] for data in self.thrust_history]
        
        # Create the plot
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, thrust_values, 'b-', linewidth=2, marker='o')
        
        # Add labels and title
        plt.xlabel('Time Step')
        plt.ylabel('Thrust (N)')
        plt.title('Propeller Thrust vs Time Step')
        
        # Add grid
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Customize axis
        plt.tight_layout()
        
        # Show plot
        plt.show()
        
    def plot_detailed_gamma_distribution(self, quad_propeller_mesh):
        """
        Plot detailed gamma distribution across all propellers and their blades
        including radial position information.
        """

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
                    r_R = r/2.809  # Normalize by tip radius (R_tip)
                    
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

    def plot_blades_pressure_top_view(self, quad_propeller_mesh, propeller_key, omega, body_velocity=np.array([0., 0., 0.]), rho=1.225):
        """
        Plot the pressure coefficient distribution with smooth interpolation and blue-red colormap
        """
        
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        
        propeller_data = quad_propeller_mesh[propeller_key]
        patches = []
        pressure_values = []
        
        hub_position = np.array(propeller_data['Hub Position'])
        
        # Collect all pressure values
        all_pressures = []
        blade_keys = list(propeller_data['Blades'].keys())
        for blade_key in blade_keys:
            blade_data = propeller_data['Blades'][blade_key]
            all_pressures.extend(list(blade_data['Pressure Difference'].values()))
        
        max_abs_pressure = max(abs(min(all_pressures)), abs(max(all_pressures)))
        
        # Process blades with improved interpolation
        for blade_key in blade_keys:
            blade_data = propeller_data['Blades'][blade_key]
            
            for panel_index, panel in blade_data['Panels'].items():
                panel_2d = [(-p[1], p[0]) for p in panel]
                patch = Polygon(panel_2d, closed=True)
                patches.append(patch)
                
                pressure = blade_data['Pressure Difference'][panel_index]
                normalized_pressure = pressure / max_abs_pressure if max_abs_pressure != 0 else 0
                pressure_values.append(normalized_pressure)
        
        # Use coolwarm colormap and adjust shading for smoother appearance
        collection = PatchCollection(patches, cmap='inferno', alpha=1.0, edgecolors='face', 
                                linewidths=0.1, antialiased=True)
        collection.set_array(np.array(pressure_values))
        
        # Center the colormap around zero for better visualization
        collection.set_clim(-1, 1)
        ax.add_collection(collection)
        
        # Adjust colorbar
        cax = fig.add_axes([0.1, 0.3, 0.8, 0.03])
        cbar = plt.colorbar(collection, cax=cax, orientation='horizontal')
        cbar.set_label('Pressure Difference', size=11)
        cbar.ax.tick_params(labelsize=11)
        
        # Set symmetric limits around zero
        vmin = -max(abs(min(pressure_values)), abs(max(pressure_values)))
        vmax = -vmin
        collection.set_clim(vmin, vmax)
        
        ax.set_aspect('equal')
        margin = 0.05
        all_points = np.array([(-point[1], point[0]) for blade_key in blade_keys
                            for panel in propeller_data['Blades'][blade_key]['Panels'].values() 
                            for point in panel])
        
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        ax.set_xlim(x_min - margin * width, x_max + margin * width)
        ax.set_ylim(y_min - margin * height, y_max + margin * height)
        
        ax.plot(-hub_position[1], hub_position[0], 'ko', markersize=6)
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        plt.show()

    def plot_pressure_contour(self, quad_propeller_mesh, propeller_key):
        """
        Plot the pressure coefficient distribution as a contour plot,
        interpolating between panel centers.
        """
        fig = plt.figure(figsize=(12, 4))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        
        propeller_data = quad_propeller_mesh[propeller_key]
        hub_position = np.array(propeller_data['Hub Position'])
        
        # Collect all points and pressures for interpolation
        points = []
        pressures = []
        all_points = []
        
        # First collect all pressure values to find the maximum
        all_pressures = []
        for blade_key in ['Blade_1', 'Blade_2']:
            blade_data = propeller_data['Blades'][blade_key]
            all_pressures.extend(list(blade_data['Pressure Difference'].values()))
        
        max_abs_pressure = max(abs(min(all_pressures)), abs(max(all_pressures)))
        
        # Process blades and collect panel centers and pressures
        for blade_key in ['Blade_1', 'Blade_2']:
            blade_data = propeller_data['Blades'][blade_key]
            
            for panel_index, panel in blade_data['Panels'].items():
                panel_2d = np.array([(-p[1], p[0]) for p in panel])
                all_points.extend(panel_2d)
                
                # Calculate panel center
                center = np.mean(panel_2d, axis=0)
                points.append(center)
                
                # Normalize pressure
                pressure = blade_data['Pressure Difference'][panel_index]
                normalized_pressure = pressure / max_abs_pressure if max_abs_pressure != 0 else 0
                pressures.append(normalized_pressure)
        
        all_points = np.array(all_points)
        points = np.array(points)
        
        # Create a regular grid for interpolation
        margin = 0.05
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        
        width = x_max - x_min
        height = y_max - y_min
        
        grid_x, grid_y = np.mgrid[
            x_min - margin * width:x_max + margin * width:200j,
            y_min - margin * height:y_max + margin * height:200j
        ]
        
        # Interpolate pressures onto the regular grid
        grid_z = griddata(points, pressures, (grid_x, grid_y), method='cubic', fill_value=np.nan)
        
        # Create contour plot
        levels = np.linspace(-1, 1, 41)
        contour = ax.contourf(grid_x, grid_y, grid_z.T, levels=levels, cmap='RdBu_r', extend='both')
        
        # Add colorbar
        cax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
        cbar = plt.colorbar(contour, cax=cax, orientation='horizontal')
        cbar.set_label('Normalized Pressure Coefficient', size=11)
        cbar.ax.tick_params(labelsize=11)
        
        # Plot blade outlines
        for blade_key in ['Blade_1', 'Blade_2']:
            blade_data = propeller_data['Blades'][blade_key]
            for panel in blade_data['Panels'].values():
                panel_2d = np.array([(-p[1], p[0]) for p in panel])
                ax.plot(panel_2d[:, 0], panel_2d[:, 1], 'k-', linewidth=0.5, alpha=0.3)
        
        # Plot hub position
        ax.plot(-hub_position[1], hub_position[0], 'ko', markersize=6)
        
        ax.set_aspect('equal')
        ax.set_xlim(x_min - margin * width, x_max + margin * width)
        ax.set_ylim(y_min - margin * height, y_max + margin * height)
        
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        
        return fig

    def plot_quad_propeller_pressure(self, quad_propeller_mesh, omega, body_velocity=np.array([0., 0., 0.]), rho=1.225):
        """
        Plot the pressure coefficient distribution for all four propellers with a 3D-looking central ellipsoid fuselage.
        
        Parameters:
        -----------
        quad_propeller_mesh : dict
            Dictionary containing mesh data for all four propellers
        omega : array-like
            Angular velocity vector
        body_velocity : array-like
            Velocity vector of the body
        rho : float
            Air density
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.patches import Polygon, Ellipse
        from matplotlib.collections import PatchCollection

        
        # Create figure with square aspect ratio
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_axes([0.1, 0.2, 0.8, 0.7])
        
        # Lists to store all patches and pressure values
        all_patches = []
        all_pressure_values = []
        
        # Scale factor to bring propellers closer together (adjust as needed)
        scale_factor = 0.6  # Reduce this to bring propellers closer
        
        # Process all four propellers
        for propeller_key in quad_propeller_mesh.keys():
            propeller_data = quad_propeller_mesh[propeller_key]
            hub_position = np.array(propeller_data['Hub Position']) * scale_factor
            
            # Process both blades for each propeller
            for blade_key in ['Blade_1', 'Blade_2']:
                blade_data = propeller_data['Blades'][blade_key]
                
                for panel_index, panel in blade_data['Panels'].items():
                    # Scale and convert panel coordinates for top view
                    panel = np.array(panel) * scale_factor
                    panel_2d = [(-p[1], p[0]) for p in panel]
                    patch = Polygon(panel_2d, closed=True)
                    all_patches.append(patch)
                    
                    # Calculate pressure coefficient
                    panel_center = np.mean(panel, axis=0)
                    radius_vector = panel_center - hub_position
                    omegad = omega[propeller_key]
                    rotational_velocity = np.cross(omegad, radius_vector)
                    local_velocity = rotational_velocity + body_velocity
                    local_velocity_magnitude = np.linalg.norm(local_velocity)
                    
                    q_local = 0.5 * rho * local_velocity_magnitude**2
                    if 'Pressure Difference' in blade_data:
                        pressure = blade_data['Pressure Difference'][panel_index]
                        cp = pressure
                    else:
                        cp = 0
                        
                    all_pressure_values.append(cp)
                    
            # Plot hub position for each propeller
            ax.plot(-hub_position[1], hub_position[0], 'ko', markersize=6)
        
        # Create patch collection for all propellers
        collection = PatchCollection(all_patches, cmap='viridis', alpha=1.0)
        collection.set_array(np.array(all_pressure_values))
        ax.add_collection(collection)
        
        # Add 3D-looking fuselage with pressure gradient
        # Create gradient effect for fuselage
        n_ellipses = 20
        base_width = 0.4
        base_height = 0.2
        angle = 30  # Angle to create 3D effect
        
        # Create color gradient for fuselage
        cmap = plt.cm.viridis
        pressures = np.linspace(-2000, 2000, n_ellipses)
        
        # Add multiple ellipses with gradient to create 3D effect
        for i in range(n_ellipses):
            width = base_width * (1 - i/(2*n_ellipses))
            height = base_height * (1 - i/(2*n_ellipses))
            alpha = 0.1 if i == 0 else 0.05
            
            # Map pressure to color
            color = cmap((pressures[i] - min(pressures))/(max(pressures) - min(pressures)))
            
            fuselage = Ellipse((0, 0), width=width, height=height, angle=angle,
                             facecolor=color, alpha=alpha, zorder=0)
            ax.add_patch(fuselage)
        
        # Add colorbar
        cax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
        cbar = plt.colorbar(collection, cax=cax, orientation='horizontal',
                          label='Pressure Coefficient (Cp)')
        cbar.ax.tick_params(labelsize=8)
        
        # Set equal aspect ratio and adjust limits
        ax.set_aspect('equal')
        
        # Calculate plot limits including all propellers
        all_points = np.array([(-point[1], point[0]) 
                            for prop_data in quad_propeller_mesh.values()
                            for blade_data in prop_data['Blades'].values()
                            for panel in blade_data['Panels'].values()
                            for point in panel]) * scale_factor
        
        margin = 0.1
        x_min, y_min = np.min(all_points, axis=0)
        x_max, y_max = np.max(all_points, axis=0)
        width = x_max - x_min
        height = y_max - y_min
        
        ax.set_xlim(x_min - margin * width, x_max + margin * width)
        ax.set_ylim(y_min - margin * height, y_max + margin * height)
        
        # Remove axes elements
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
            
        # Add title
        ax.set_title('Quadcopter Propeller Pressure Distribution - Top View')
        
        plt.show()
        return fig

    def collect_thrust_data(self, quad_propeller_mesh, omega_dict, time_step, dt, debug=False):
        """
        Collect thrust data for all propellers and their blades at a given time step.
        
        Args:
            quad_propeller_mesh: Complete mesh data for all propellers
            omega_dict: Dictionary of angular velocities for each propeller
            time_step: Current time step
            dt: Time step size
            debug: Print debug information if True
        
        Returns:
            dict: Thrust data for all propellers at this time step
        """
        thrust_data = {}
        
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            # Calculate angle for this propeller
            angle = time_step * dt * omega_dict[propeller_key][2]
            
            # Initialize thrust data for this propeller
            thrust_data[propeller_key] = {
                'angle': angle,
                'blades': {}
            }
            
            # Process each blade in this propeller
            for blade_key, blade_data in propeller_data['Blades'].items():
                # Get radial segments and thrust values for this blade
                segments, thrust = self.get_radial_segments(blade_data, debug)
                
                # Store the data
                thrust_data[propeller_key]['blades'][blade_key] = {
                    'segments': segments,
                    'thrust': thrust
                }
                
                if debug:
                    print(f"\nPropeller {propeller_key}, {blade_key}:")
                    print(f"Angle: {np.degrees(angle):.1f} degrees")
                    print(f"Thrust values: {thrust}")
        
        return thrust_data
    
    def get_radial_segments(self, blade_data, debug=False):
        """
        Gets thrust values for radial segments starting from hub.
        
        Args:
            blade_data: Dictionary containing blade information
            debug: Boolean to print debug information
            
        Returns:
            tuple: (segment_lengths, thrust_values) as numpy arrays
        """
        if debug:
            print("\nCollecting radial segments and thrust values...")
        
        # Initialize storage
        segment_lengths = []
        thrust_values = []
        
        # Get unique spanwise indices
        spanwise_indices = sorted(set(idx[1] for idx in blade_data['Panel Forces'].keys()))
        
        for span_idx in spanwise_indices:
            # Get chord indices for this spanwise position
            chord_indices = sorted(idx[0] for idx in blade_data['Panel Forces'].keys() 
                                if idx[1] == span_idx)
            
            # Calculate total thrust for this segment
            total_thrust = 0.0
            for chord_idx in chord_indices:
                panel_idx = (chord_idx, span_idx)
                force = blade_data['Panel Forces'][panel_idx]
                total_thrust += force[2]  # z-component is thrust
                
                # For the first chord index, calculate segment length
                if chord_idx == chord_indices[0]:
                    panel = np.array(blade_data['Panels'][panel_idx])
                    
                    # Calculate panel length (average of leading and trailing edge)
                    edge1_length = np.linalg.norm(panel[3] - panel[0])
                    edge2_length = np.linalg.norm(panel[2] - panel[1])
                    panel_length = (edge1_length + edge2_length) / 2
                    segment_lengths.append(panel_length)
            
            thrust_values.append(total_thrust)
            
            if debug:
                print(f"\nSegment {len(segment_lengths)}:")
                print(f"Length: {panel_length:.6f} m")
                print(f"Thrust: {total_thrust:.6f} N")
        
        return np.array(segment_lengths), np.array(thrust_values)
    
    def plot_thrust_disk(self, thrust_history, n_steps_rev=1, R_hub=0.00624, debug=False):
        """
        Plot thrust distribution disk using stored thrust data for all propellers.
        Includes both radial and azimuthal interpolation for smooth visualization.
        
        Args:
            thrust_history: List of dictionaries containing thrust data for each time step
            n_steps_rev: Number of steps per revolution
            R_hub: Hub radius in meters
            debug: Print debug information
        """
        # Create subplots for each propeller
        n_propellers = len(thrust_history[0].keys())
        n_cols = min(2, n_propellers)
        n_rows = (n_propellers + n_cols - 1) // n_cols
        fig = plt.figure(figsize=(12*n_cols, 12*n_rows))
        
        # Create a grid specification with space for colorbar
        gs = fig.add_gridspec(n_rows, n_cols + 1, width_ratios=[1]*n_cols + [0.1])
        axes = []
        
        # Create subplot axes
        for i in range(n_rows):
            row_axes = []
            for j in range(n_cols):
                if i * n_cols + j < n_propellers:
                    ax = fig.add_subplot(gs[i, j], projection='polar')
                    row_axes.append(ax)
            axes.append(row_axes)
        axes = np.array(axes)
        
        # Calculate sweep angle for each step
        sweep_angle = 2 * np.pi / n_steps_rev
        
        # Get global min and max thrust values and maximum radius
        all_thrusts = []
        max_r = R_hub
        
        for step_data in thrust_history:
            for prop_data in step_data.values():
                for blade_data in prop_data['blades'].values():
                    all_thrusts.extend(blade_data['thrust'])
                    max_r = max(max_r, R_hub + sum(blade_data['segments']))
        
        vmin, vmax = min(all_thrusts), max(all_thrusts)
        
        # Process each propeller
        for prop_idx, propeller_key in enumerate(thrust_history[0].keys()):
            row = prop_idx // n_cols
            col = prop_idx % n_cols
            ax = axes[row, col]
            
            # Create finer grid for interpolation
            n_theta = 400  # Increased resolution
            n_r = 200      # Increased resolution
            theta = np.linspace(0, 2*np.pi, n_theta)
            r = np.linspace(R_hub, max_r, n_r)
            R, THETA = np.meshgrid(r, theta)
            Z = np.full_like(R, np.nan)
            
            # Create arrays to store scattered data for interpolation
            scatter_theta = []
            scatter_r = []
            scatter_thrust = []
            
            # For each time step
            for step_idx, step_data in enumerate(thrust_history[:n_steps_rev]):
                prop_data = step_data[propeller_key]
                n_blades = len(prop_data['blades'])
                blade_angle_offset = 2 * np.pi / n_blades
                
                # Process each blade
                for blade_idx, (blade_key, blade_data) in enumerate(prop_data['blades'].items()):
                    base_angle = step_idx * sweep_angle + blade_idx * blade_angle_offset
                    
                    # Calculate radial positions
                    r_positions = [R_hub]
                    r_start = R_hub
                    for length in blade_data['segments']:
                        r_positions.append(r_start + length)
                        r_start += length
                    r_positions = np.array(r_positions)
                    
                    # Create finer radial grid for each blade section
                    r_fine = np.linspace(R_hub, r_positions[-1], len(r_positions) * 4)
                    thrust_fine = np.interp(r_fine, 
                                        (r_positions[:-1] + r_positions[1:]) / 2, 
                                        blade_data['thrust'])
                    
                    # Create angular points for interpolation
                    # Use multiple angular positions within sweep angle for smooth transition
                    n_angular_points = 5
                    angles = np.linspace(base_angle, base_angle + sweep_angle, n_angular_points)
                    
                    # Add points for interpolation
                    for angle in angles:
                        scatter_theta.extend([angle] * len(r_fine))
                        scatter_r.extend(r_fine)
                        scatter_thrust.extend(thrust_fine)
                        
                        # Add points for periodic boundary condition
                        if angle < sweep_angle:
                            scatter_theta.extend([angle + 2*np.pi] * len(r_fine))
                            scatter_r.extend(r_fine)
                            scatter_thrust.extend(thrust_fine)
            
            # Convert to numpy arrays
            scatter_theta = np.array(scatter_theta)
            scatter_r = np.array(scatter_r)
            scatter_thrust = np.array(scatter_thrust)
            
            # Perform 2D interpolation using griddata
            from scipy.interpolate import griddata
            points = np.column_stack((scatter_theta, scatter_r))
            Z = griddata(points, scatter_thrust, (THETA, R), method='cubic', fill_value=np.nan)
            
            # Plot thrust distribution
            im = ax.pcolormesh(THETA, R, Z, shading='auto', cmap='jet', 
                            vmin=vmin, vmax=vmax)
            
            # Customize plot
            ax.grid(True)
            
            # Set angle labels with increased size and padding
            ax.set_thetagrids(np.arange(0, 360, 45), 
                            labels=[f'{x}°' for x in range(0, 360, 45)],
                            fontsize=40)
            ax.tick_params(axis='x', pad=32)
            
            # Hide radial labels but keep grid
            ax.set_rgrids(np.linspace(R_hub, max_r, 5), labels=[''] * 5)
        
        # Add single colorbar for all subplots
        cbar_ax = fig.add_subplot(gs[:, -1])
        cbar = plt.colorbar(im, cax=cbar_ax)
        cbar.set_label('Thrust (N)', fontsize=62, labelpad=22)
        cbar.ax.tick_params(labelsize=60)
        
        plt.tight_layout()
        return fig, axes