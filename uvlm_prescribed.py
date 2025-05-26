from propeller import PropellerGeometry
from mesh import PropellerMesh
from wind import WindField

from scipy.interpolate import UnivariateSpline
from scipy.io import savemat
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

import torch
import copy
import numpy as np
import cupy as cp
import numba as nb
from numba import cuda, float64, int32, jit, prange
import math


class UVLM:
    def __init__(self, propeller_mesh):
        """Initialize the UVLM solver with the given propeller mesh."""
        self.propeller_mesh = propeller_mesh
        self.wake_system = {}
        # self.initialize_wake_system(propeller_mesh)

    def initialize_wake_system(self, propeller_mesh):
        """Initialize wake system with specific control point indexing pattern."""
        for prop_key, propeller_data in propeller_mesh.items():
            self.wake_system[prop_key] = {}
            for blade_key, blade_data in propeller_data['Blades'].items():
                self.wake_system[prop_key][blade_key] = {}
                
                max_chordwise_index = max(idx[1] for idx in blade_data['Vortex Rings'].keys())
               
                for row in range(1):
                    for chordwise_idx in range(max_chordwise_index + 1):
                        cp_index = (row, chordwise_idx)
                        self.wake_system[prop_key][blade_key][cp_index] = {
                            'Gamma': None,
                            'Control Points': None,
                            'Time History': None,
                            'Vortex Rings': {
                                'Vertices': torch.zeros((4, 3), device=self.device)
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
    
    @staticmethod
    def biot_savart_wake(r1, r2, r0, dt, step, gamma):
        """Calculate induced velocity for wake segments."""
        # Use torch.linalg.cross for modern cross product calculation
        cross_r1_r2 = torch.linalg.cross(r1, r2)
        norm_cross_r1_r2 = torch.norm(cross_r1_r2)
        
        v = 1.48e-5  # viscosity
        eps = 1.25643  # Oseen parameter
        a = 1e-3
        n = 2  # Core model exponent
        
        rc = torch.tensor(2.92e-4, device=r1.device)
        if step >= 3:
            rc = torch.sqrt(rc**2 + (4 * eps * (1 + a * (gamma / v))) * v * dt * step)
            
        denominator = ((norm_cross_r1_r2**(2*n)) + ((torch.norm(rc*r0))**(2*n)))**(1/n)
        
        induced_velocity = (1.0 / (4 * torch.pi)) * (cross_r1_r2 / denominator) * \
                        torch.dot(r0, (r1/torch.norm(r1) - r2/torch.norm(r2)))
        
        return induced_velocity
    
    # def calculate_bound_to_bound_induced_velocity_matrix(self, quad_propeller_mesh, omega_dict):
    #     """Calculate the global induced velocity matrix using GPU acceleration."""
    #     global_matrices = {}
        
    #     for propeller_key, propeller_data in quad_propeller_mesh.items():
    #         hub_position = np.array(propeller_data['Hub Position'])
    #         control_points = []
    #         vortex_rings = []
            
    #         for blade_key, blade_data in propeller_data['Blades'].items():
    #             for cp_index, control_point in blade_data['Control Points'].items():
    #                 control_points.append((blade_key, cp_index, control_point))
    #                 vortex_rings.append((blade_key, cp_index, blade_data['Vortex Rings'][cp_index]))
            
    #         num_points = len(control_points)
    #         global_matrix = torch.zeros((num_points, num_points, 3), device=self.device)
            
    #         # Batch process vertices and compute induced velocities
    #         for i, (cp_blade_key, cp_index, control_point) in enumerate(control_points):
    #             for j, (vr_blade_key, vr_index, vortex_ring) in enumerate(vortex_rings):

    #                 vertices = vortex_ring['Vertices']

    #                # Initialize induced velocity for this pair
    #                 total_induced_velocity = np.zeros(3)
                    
    #                 for k in range(4):
    #                     vertex_start = vertices[k]
    #                     vertex_end = vertices[(k + 1) % 4]
                        
    #                     r1 = control_point - vertex_end
    #                     r2 = control_point - vertex_start
    #                     r0 = r1 - r2
                        
    #                     induced_velocity = self.biot_savart(r1, r2, r0, gamma=1.0)
    #                     total_induced_velocity += induced_velocity
                    
    #                 global_matrix[i, j] = total_induced_velocity
            
    #         global_matrices[propeller_key] = global_matrix
        
    #     return global_matrices
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

                        infinity_factor = 100  # Adjust based on domain size

                        
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
    
    def calculate_wake_to_bound_induced_velocity_matrix(self, quad_propeller_mesh, dt, time_step):
        """Calculate induced velocities from wake vortex rings onto bound control points"""
        final_induced_velocities = {}
        
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            control_points = []
            wake_vortex_rings = []
            wake_gamma_values = []  
            control_point_indices = []  # For debugging
            
            # Get bound control points
            for blade_key, blade_data in propeller_data['Blades'].items():
                for cp_index, control_point in blade_data['Control Points'].items():
                    control_points.append((blade_key, cp_index, control_point))

            # Get wake vortex rings and their gamma values
            wake_data = self.wake_system[propeller_key]
            for blade_key, blade_data in wake_data.items():
                for cp_index, wake_panel in blade_data.items():
                    if wake_panel['Vortex Rings']['Vertices'] is not None:
                        wake_vortex_rings.append((blade_key, cp_index, wake_panel['Vortex Rings']))
                        wake_gamma_values.append(wake_panel['Gamma'])
            
            # Convert gamma values to numpy array
            wake_gamma = np.array(wake_gamma_values).reshape(-1, 1)
            # Initialize matrix dimensions
            num_bound_points = len(control_points)
            num_wake_rings = len(wake_vortex_rings)
            influence_matrix = np.zeros((num_bound_points, num_wake_rings, 3))
            
            # Calculate influence coefficients
            for i, (cp_blade_key, cp_index, control_point) in enumerate(control_points):
                for j, (wake_blade_key, wake_index, wake_ring) in enumerate(wake_vortex_rings):
                    total_induced_velocity = np.zeros(3)
                    vertices = wake_ring['Vertices']
                    
                    for k in range(4):
                        vertex_start = np.array(vertices[k])
                        vertex_end = np.array(vertices[(k + 1) % 4])
                        
                        r1 = control_point - vertex_end
                        r2 = control_point - vertex_start
                        r0 = vertex_start - vertex_end

                        gamma = self.wake_system[propeller_key][wake_blade_key][wake_index]['Gamma']
                        time_history = self.wake_system[propeller_key][wake_blade_key][wake_index]['Time History']
                        step = time_step - time_history 

                        induced_velocity = self.biot_savart_wake(r1, r2, r0, dt, step, gamma=gamma)
                        total_induced_velocity += induced_velocity

                    influence_matrix[i, j] = total_induced_velocity
            # Calculate final induced velocities
            
            induced_velocities = np.einsum('ijk,j->ik', influence_matrix, wake_gamma.flatten())
            
            final_induced_velocities[propeller_key] = induced_velocities
     
        return final_induced_velocities
 
    def calculate_gamma(self, quad_propeller_mesh, bound_to_bound_global_matrices,  
                    wake_to_bound_induced_velocity_matrices, omega_dict, 
                    body_velocity, wind_field, com_position, time_step, roll, pitch, yaw, angle_corrections=None):
        """Calculate gamma (circulation strength) for each propeller."""
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

        # Initialize storage for results
        gamma_matrices = {}
        induced_velocities = {}
        R = rotation_matrix(roll, pitch, yaw)
        
        # Get wind function
        wind_func = WindField.update_wind_function(wind_field, com_position)
        
        # Process each propeller
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            # Get omega vector and hub position
            effective_omega = np.array(omega_dict[propeller_key])
            hub_position = np.array(propeller_data['Hub Position'])
            
            # Initialize lists for data collection
            control_points = []
            normals = []
            rhs = []
            
            # Process each blade
            for blade_key, blade_data in propeller_data['Blades'].items():
                
                for cp_index, control_point in blade_data['Control Points'].items():
                    control_point = np.array(control_point)
                    wind_velocity = -wind_func(control_point)

                    # Get normal vector
                    normal = np.array(blade_data['Normals'][cp_index])
                    
                    # Calculate rotated normal
                    rotated_normal = normal
                    normals.append(rotated_normal)
                    
                    # Calculate radius vector and cross product
                    radius_vector = control_point - hub_position
                    omega_cross_r = np.cross(effective_omega, radius_vector)
                    
                    # Store wind velocity
                    if "Wind Velocity" not in blade_data:
                        blade_data['Wind Velocity'] = {}
                    blade_data['Wind Velocity'][cp_index] = wind_velocity
                    
                    # Store omega x r
                    if "Omega_Cross_R" not in blade_data:
                        blade_data['Omega_Cross_R'] = {}
                    blade_data['Omega_Cross_R'][cp_index] = omega_cross_r
                    
                    # Calculate velocity term and RHS value
                    velocity_term = -omega_cross_r
                    rhs_value = -np.dot(velocity_term, normal)
                    original_rhs = rhs_value
                    
                    rhs.append(rhs_value)

                    control_points.append(control_point)
            
            # Convert lists to numpy arrays
            normals = np.array(normals)
            rhs = np.array(rhs).reshape(-1, 1)
            
            # Get influence matrix for this propeller
            bound_to_bound_induced_matrix = np.array(bound_to_bound_global_matrices[propeller_key])
            
            # Calculate influence coefficients
            Ab = np.einsum('ijk,ik->ij', bound_to_bound_induced_matrix, normals)
            
            # Handle wake effects
            if time_step > 2:
                wake_to_bound_induced_velocity_matrix = np.array(wake_to_bound_induced_velocity_matrices[propeller_key])
                Aw = np.zeros((len(normals), 1))
                for i in range(len(normals)):
                    Aw[i, 0] = np.dot(wake_to_bound_induced_velocity_matrix[i], normals[i])
            else:
                Aw = np.zeros_like(rhs)
            
            # Solve linear system for gamma
            gamma = np.linalg.solve(Ab, (rhs - Aw))
            gamma_matrices[propeller_key] = gamma.flatten()
            # print('gamma', gamma)
            # Calculate induced velocities
            induced_vel = np.zeros((len(normals), 3))
            for i in range(len(normals)):
                for j in range(len(normals)):
                    induced_vel[i] += bound_to_bound_induced_matrix[i, j] * gamma.flatten()[j]
            
            induced_velocities[propeller_key] = induced_vel
            
            # Update gamma values in the mesh
            gamma_index = 0
            for blade_key, blade_data in propeller_data['Blades'].items():
                blade_data['Gamma'] = {}
                blade_data['Induced_Velocities'] = {}
                blade_data['Wake_Induced_Velocities'] = {}
                
                for cp_index in blade_data['Control Points'].keys():
                    blade_data['Gamma'][cp_index] = float(gamma[gamma_index])
                    blade_data['Induced_Velocities'][cp_index] = induced_vel[gamma_index]
                    
                    if time_step > 2:
                        wake_induced_vel = wake_to_bound_induced_velocity_matrices[propeller_key][gamma_index]
                        blade_data['Wake_Induced_Velocities'][cp_index] = wake_induced_vel
                    
                    gamma_index += 1
        
        return gamma_matrices, induced_velocities
    
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
                    bound_to_bound_induced_velocity = blade_data['Induced_Velocities'][panel_index]
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
                    total_velocity =  -omega_cross_r
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
                    # pressure = rho * (gamma_current-gamma_previous_span) * np.linalg.cross((total_velocity + induced_velocity), tangent_span)
                    # pressure2 = rho * (gamma_current-gamma_previous_chord) * np.linalg.cross((total_velocity + induced_velocity), tangent_chord)
            
                    
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
        self.calculate_gamma(
            propeller_mesh,
            bound_to_bound_global_matrices,
            wake_to_bound_induced_velocity_matrices,
            omega,
            body_velocity,
            wind_field, 
            com_position, 
            time_step,
            roll=roll,
            pitch=pitch,
            yaw=yaw
        )

        # Step 3: Update the pressure differences for all panels
        self.pressure_difference(
            propeller_mesh,
            bound_to_bound_global_matrices,
            wake_to_bound_induced_velocity_matrices,
            body_velocity,
            omega,
            time_step,
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
                    control_point = blade_data['Control Points'][panel_index]
                    
                    # Calculate the normal vector and panel area
                    normal = blade_data['Normals'][panel_index]
                    
                    area_triangle_1 = 0.5 * np.linalg.norm(np.cross(panel_array[1] - panel_array[0], panel_array[3] - panel_array[0]))
                    area_triangle_2 = 0.5 * np.linalg.norm(np.cross(panel_array[2] - panel_array[1], panel_array[3] - panel_array[1]))

                    # panel_area =  0.5 * np.linalg.norm(
                    #     np.cross(panel_array[2] - panel_array[0], panel_array[3] - panel_array[1])
                    # )
                    panel_area = area_triangle_1 + area_triangle_2
                    
                    # Calculate vectors for Triangle 1
                    
                    # Force contribution for this pane
                    # force = (pressure)
                    force = np.dot(pressure * panel_area, normal)
                    
                    # moment_arm_local = panel_center - hub_position
                    # moment_local = np.cross(moment_arm_local, force)
                   
                    # # Moment contribution for this panel
                    # moment_arm_global = hub_position 
                    # moment_global = np.cross(moment_arm_global, force)
                    # moment = np.cross(moment_arm, force)

                    moment_arm_total = control_point  # Vector from origin to force application point
                    panel_moment = np.cross(moment_arm_total, force)
                    
                    # Store the force and moment for this panel
                    blade_data['Panel Forces'][panel_index] = force
                    blade_data['Panel Moments'][panel_index] = panel_moment

                    # Accumulate to total force and moment
                    total_force += force
                    total_moment += panel_moment

            # Store total force and moment for this propeller
            # print(total_force)
            total_forces_and_moments[propeller_key] = {'force': total_force, 'moment': total_moment}
            
        print("Forces", total_forces_and_moments['Propeller_1']['force'])
        # print("Forces", total_forces_and_moments['Propeller_2']['force'])
        # print("Forces", total_forces_and_moments['Propeller_3']['force'])
        # print("Forces", total_forces_and_moments['Propeller_4']['force'])
        return total_forces_and_moments

    def viscous_coupling(self, quad_propeller_mesh, omega_dict, 
                    body_velocity, wind_field, com_position, time_step, dt, roll, pitch, yaw,
                    rho=1.225, relaxation_factor=0.1, max_iterations=20, convergence_tolerance=1e-4):
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
        gamma_matrices, _ = self.calculate_gamma(
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
        
        # Calculate initial pressure and forces
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
        
        # Calculate initial forces for reference
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
        
        # Initialize storage for viscous coupling
        tip_radius = 0.11938  # Propeller tip radius (adjust as needed)
        
        # Process each propeller
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            hub_position = np.array(propeller_data['Hub Position'])
            omega_vector = np.array(omega_dict[propeller_key])
            propeller_axis = np.array([0, 0, 1])  # Assuming Z-axis is along propeller shaft
            
            # Process each blade
            for blade_key, blade_data in propeller_data['Blades'].items():
                # Initialize data structures
                if 'Local_AOA' not in blade_data:
                    blade_data['Local_AOA'] = {}
                    blade_data['Alpha_3D'] = {}
                    blade_data['Cl_inv'] = {}
                    blade_data['Cl_vis'] = {}
                    blade_data['r_over_R'] = {}
                    blade_data['Gamma_avg'] = {}  # To store average circulation for each spanwise section
                    blade_data['lambda_ratio'] = {}  # To store strength ratios
                    blade_data['Gtable'] = {}  # To store circulation from airfoil data
                    blade_data['Inflow_Velocity'] = {}  # To store inflow velocity components
                    
                # Step 2: Calculate average circulation strength for each spanwise section
                # Get unique spanwise indices by looking at control points
                # In this case, we group points by their spanwise position (cp_index[1])
                spanwise_sections = {}
                for cp_index, gamma_value in blade_data['Gamma'].items():
                    spanwise_idx = cp_index[1]  # Assuming format is (chordwise, spanwise)
                    if spanwise_idx not in spanwise_sections:
                        spanwise_sections[spanwise_idx] = []
                    
                    spanwise_sections[spanwise_idx].append(cp_index)
                
                # Calculate average circulation (Gavg,j) for each spanwise section
                # and strength ratios (λi,j) as per equation (15)
                for spanwise_idx, cp_indices in spanwise_sections.items():
                    # Calculate Gavg,j = (1/M) * Σ(Gi,j)
                    total_gamma = 0.0
                    for cp_index in cp_indices:
                        total_gamma += blade_data['Gamma'][cp_index]
                    
                    avg_gamma = total_gamma / len(cp_indices)
                    blade_data['Gamma_avg'][spanwise_idx] = avg_gamma
                    
                    # Calculate λi,j = Gi,j / Gavg,j
                    for cp_index in cp_indices:
                        if avg_gamma != 0:
                            lambda_ratio = blade_data['Gamma'][cp_index] / avg_gamma
                        else:
                            lambda_ratio = 1.0  # Default to 1 if average circulation is zero
                        
                        blade_data['lambda_ratio'][cp_index] = lambda_ratio
                
                # Step 3: Calculate local inflow velocity, inflow angle, and effective AOA
                # for each control point
                for cp_index, control_point in blade_data['Control Points'].items():
                    cp_position = np.array(control_point)
                    radius_vector = cp_position - hub_position
                    
                    # Calculate r/R value (normalized radial position)
                    radial_proj = radius_vector - np.dot(radius_vector, propeller_axis) * propeller_axis
                    r = np.linalg.norm(radius_vector)
                    r_over_R = r / tip_radius
                    blade_data['r_over_R'][cp_index] = r_over_R
                    
                    # Calculate local coordinate system
                    if np.linalg.norm(radial_proj) > 1e-10:
                        radial_dir = radial_proj / np.linalg.norm(radial_proj)
                    else:
                        radial_dir = np.array([1, 0, 0])
                    
                    tangential_dir = np.cross(propeller_axis, radial_dir)
                    if np.linalg.norm(tangential_dir) > 1e-10:
                        tangential_dir = tangential_dir / np.linalg.norm(tangential_dir)
                    
                    # Calculate velocity components
                    rotational_velocity = np.cross(omega_vector, radius_vector)
                    
                    # Get induced velocities
                    induced_velocity = np.zeros(3)
                    if 'Induced_Velocities' in blade_data and cp_index in blade_data['Induced_Velocities']:
                        induced_velocity = np.array(blade_data['Induced_Velocities'][cp_index])
                    
                    wake_induced_velocity = np.zeros(3)
                    if time_step > 2 and 'Wake_Induced_Velocities' in blade_data and cp_index in blade_data['Wake_Induced_Velocities']:
                        wake_induced_velocity = np.array(blade_data['Wake_Induced_Velocities'][cp_index])
                    
                    # Calculate total inflow velocity (Vinflow)
                    # Vinflow = V∞ - Ω × r + Vind.bound + Vind.wake
                    free_stream_velocity = np.zeros(3)  # V∞ is zero for hover
                    inflow_velocity = free_stream_velocity - rotational_velocity + induced_velocity + wake_induced_velocity
                    
                    # Store inflow velocity components for later use
                    blade_data['Inflow_Velocity'][cp_index] = {
                        'vector': inflow_velocity,
                        'a1': np.dot(inflow_velocity, tangential_dir),  # Tangential component
                        'a3': np.dot(inflow_velocity, propeller_axis)   # Axial component
                    }
                    
                    # Calculate inflow angle using the provided formula
                    # ainflow = tan^-1(Vinflow.a3 / Vinflow.a1)
                    a1 = blade_data['Inflow_Velocity'][cp_index]['a1']
                    a3 = blade_data['Inflow_Velocity'][cp_index]['a3']
                    
                    inflow_angle = np.arctan2(a3, a1)
                    
                    # Store Alpha_3D (3D inflow angle)
                    blade_data['Alpha_3D'][cp_index] = inflow_angle
                    
                    # Initialize Local_AOA to the 3D inflow angle if first iteration
                    if cp_index not in blade_data['Local_AOA'] or blade_data['Local_AOA'][cp_index] == 0:
                        blade_data['Local_AOA'][cp_index] = inflow_angle
        
        # Viscous coupling iteration
        for iteration in range(max_iterations):
            # Flag to track convergence
            converged = True
            max_gamma_diff = 0
            
            # For each propeller and blade, process spanwise sections
            for propeller_key, propeller_data in quad_propeller_mesh.items():
                for blade_key, blade_data in propeller_data['Blades'].items():
                    # Process each spanwise section
                    for spanwise_idx in blade_data['Gamma_avg'].keys():
                        # Get all control points at this spanwise position
                        cp_indices = [idx for idx in blade_data['Control Points'].keys() 
                                    if idx[1] == spanwise_idx]
                        
                        # Get a representative control point for this section
                        rep_idx = cp_indices[len(cp_indices)//2]
                        
                        # Step 4: Obtain lift coefficient from polar data
                        # Calculate effective angle of attack (Eq. 27)
                        # αe = Cl_inv/(2π) - αlocal + α3D
                        
                        # First, get inviscid lift coefficient for this section
                        # We'll use the average circulation for this calculation
                        avg_gamma = blade_data['Gamma_avg'][spanwise_idx]
                        
                        # Calculate local inflow velocity components
                        a1 = blade_data['Inflow_Velocity'][rep_idx]['a1']
                        a3 = blade_data['Inflow_Velocity'][rep_idx]['a3']
                        
                        # Calculate the total velocity magnitude
                        v_magnitude = np.sqrt(a1**2 + a3**2)
                        
                        # Calculate the sectional area
                        section_area = 0.0
                        for cp_index in cp_indices:
                            # Get panel area
                            panel = np.array(blade_data['Panels'][cp_index])
                            area_triangle_1 = 0.5 * np.linalg.norm(np.cross(panel[1] - panel[0], panel[3] - panel[0]))
                            area_triangle_2 = 0.5 * np.linalg.norm(np.cross(panel[2] - panel[1], panel[3] - panel[1]))
                            panel_area = area_triangle_1 + area_triangle_2
                            section_area += panel_area
                        
                        # Calculate inviscid lift coefficient based on circulation
                        # Cl_inv = 2 * Gavg / (Velocity * chord)
                        if v_magnitude > 1e-10 and section_area > 1e-10:
                            span = np.linalg.norm(panel[2] - panel[1])
                            chord = section_area / span if span > 1e-10 else 0.1
                            cl_inv = 2 * avg_gamma / (v_magnitude * chord)
                        else:
                            cl_inv = 0.0
                        
                        # Store Cl_inv for all points in this section
                        for cp_index in cp_indices:
                            blade_data['Cl_inv'][cp_index] = cl_inv
                        
                        # Get the blade twist angle for this section
                        twist_angle = 0.0
                        if 'Twist' in blade_data and rep_idx in blade_data['Twist']:
                            twist_angle = blade_data['Twist'][rep_idx]
                            
                        # Calculate effective angle of attack (αe)
                        local_aoa = blade_data['Local_AOA'][rep_idx]
                        alpha_3d = blade_data['Alpha_3D'][rep_idx]
                        
                        # αe = Cl_inv/(2π) - αlocal + α3D
                        # Note: the local_aoa already accounts for twist, as we set it to (inflow_angle - twist_angle)
                        alpha_e = cl_inv / (2 * np.pi) - local_aoa + alpha_3d
                        
                        # Convert effective AOA to degrees for lookup
                        alpha_e_deg = np.degrees(alpha_e)
                        
                        # Get lift coefficient from polar data
                        cl_vis = 0.0  # Default value
                        
                        # Find nearest polar data for this r/R value
                        r_over_R = blade_data['r_over_R'][rep_idx]
                        nearest_panel_idx = None
                        min_r_R_diff = float('inf')
                        
                        for idx, panel_r_R in blade_data['r_over_R'].items():
                            if idx[1] == spanwise_idx:  # Only consider panels in this section
                                r_R_diff = abs(panel_r_R - r_over_R)
                                if r_R_diff < min_r_R_diff:
                                    min_r_R_diff = r_R_diff
                                    nearest_panel_idx = idx
                        
                        # Access polar data to get Cl for the effective angle
                        if 'Polar Data' in blade_data and rep_idx in blade_data['Polar Data']:
                            polar = blade_data['Polar Data'][rep_idx]
                            if polar is not None:
                                # Find closest alpha value
                                alpha_arr = np.array(polar['alpha'])
                                alpha_idx = np.argmin(np.abs(alpha_arr - alpha_e_deg))
                                if alpha_idx < len(polar['cl']) and not np.isnan(polar['cl'][alpha_idx]):
                                    cl_vis = polar['cl'][alpha_idx]
                        
                        # Store Cl_vis for this section
                        for cp_index in cp_indices:
                            blade_data['Cl_vis'][cp_index] = cl_vis
                        
                        # Step 5: Calculate new circulation using Kutta-Joukowski theorem (Eq. 17)
                        # Gtable,j = (1/2 * (Vinflow.a1^2 + Vinflow.a3^2) * cl,j * dAj) / 
                        #            sqrt((Vinflow*l.a1)^2 + (Vinflow*l.a3)^2)
                        
                        inflow_velocity_squared = a1**2 + a3**2
                        dAj = section_area
                        
                        # For the denominator, we need to calculate (Vinflow*l.a1)^2 + (Vinflow*l.a3)^2
                        # where l appears to be a unit vector
                        # We'll approximate this as the inflow velocity magnitude for simplicity
                        denominator = np.sqrt(inflow_velocity_squared)
                        
                        if denominator > 1e-10:
                            # Calculate new circulation (Gtable,j)
                            g_table = 0.5 * inflow_velocity_squared * cl_vis * dAj / denominator
                        else:
                            g_table = 0.0
                        
                        # Store Gtable for this section
                        blade_data['Gtable'][spanwise_idx] = g_table
                        
                        # Compare with current average circulation
                        current_avg_gamma = blade_data['Gamma_avg'][spanwise_idx]
                        gamma_diff = abs(g_table - current_avg_gamma)
                        max_gamma_diff = max(max_gamma_diff, gamma_diff)
                        
                        # Check convergence for this section
                        if gamma_diff > convergence_tolerance * abs(current_avg_gamma) and abs(current_avg_gamma) > 1e-10:
                            converged = False
                        
                        # Step 6: Correct the circulation strength with an under-relaxation factor (Eq. 18)
                        if iteration == 0:
                            # First iteration: use initial gamma and table gamma
                            # Gk+1_j = Gk_avg,j + D * (Gk_table,j - Gk_avg,j)
                            updated_gamma = current_avg_gamma + relaxation_factor * (g_table - current_avg_gamma)
                        else:
                            # Later iterations: use current and previous table gamma
                            # Gk+1_j = Gk_table,j + D * (Gk_table,j - Gk-1_table,j)
                            prev_g_table = blade_data.get('Gtable_previous', {}).get(spanwise_idx, current_avg_gamma)
                            updated_gamma = g_table + relaxation_factor * (g_table - prev_g_table)
                        
                        # Store updated average circulation
                        blade_data['Gamma_avg'][spanwise_idx] = updated_gamma
                        
                        # Calculate relative error for convergence checking
                        if abs(current_avg_gamma) > 1e-10:
                            rel_error = abs(updated_gamma - current_avg_gamma) / abs(current_avg_gamma)
                            if rel_error > convergence_tolerance:
                                converged = False
                        
                        # Apply the updated circulation to individual elements using the strength ratios
                        for cp_index in cp_indices:
                            lambda_ratio = blade_data['lambda_ratio'][cp_index]
                            blade_data['Gamma'][cp_index] = updated_gamma * lambda_ratio
                    
                    # Store current Gtable values for next iteration
                    if 'Gtable_previous' not in blade_data:
                        blade_data['Gtable_previous'] = {}
                    blade_data['Gtable_previous'] = blade_data['Gtable'].copy()
                    
                    # Update local AOA for the next iteration
                    # This is a key step in the iterative process
                    for cp_index in blade_data['Control Points'].keys():
                        spanwise_idx = cp_index[1]
                        
                        # Get current cl values
                        cl_inv = blade_data['Cl_inv'][cp_index]
                        cl_vis = blade_data['Cl_vis'][cp_index]
                        
                        # Calculate lift coefficient difference
                        cl_diff = cl_inv - cl_vis
                        
                        # Update local AOA using relaxation
                        # local_aoa_update = local_aoa + relaxation_factor * (cl_diff / (2 * np.pi))
                        local_aoa = blade_data['Local_AOA'][cp_index]
                        local_aoa_update = local_aoa + relaxation_factor * (cl_diff / (2 * np.pi))
                        
                        # Store updated local AOA
                        blade_data['Local_AOA'][cp_index] = local_aoa_update
            
            # Track convergence
            convergence_history.append(max_gamma_diff)
            
            # Step 7: Check convergence criterion
            # Track convergence progress
            if iteration > 0:
                gamma_change = max_gamma_diff / max(abs(gval) for prop in quad_propeller_mesh.values() 
                                            for blade in prop['Blades'].values() 
                                            for gval in blade['Gamma'].values() if abs(gval) > 1e-10)
                convergence_history.append(gamma_change)
                print(f"Iteration {iteration+1}: Relative gamma change = {gamma_change:.6f}")
                
                # Check if convergence criterion is satisfied (within 0.001% as specified)
                if gamma_change < 0.00001:  # 0.001% expressed as decimal
                    converged = True
                    
            if converged:
                print(f"Viscous coupling converged after {iteration+1} iterations")
                break
            
            # If not converged and not the last iteration, recalculate pressure and forces
            if not converged and iteration < max_iterations - 1:
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
        
        # Step 8: Apply updated circulation to vortex ring elements and conduct wake convection
        # First, apply the updated circulation values throughout the mesh
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            for blade_key, blade_data in propeller_data['Blades'].items():
                # Apply the final circulation values using the strength ratios
                for cp_index in blade_data['Control Points'].keys():
                    spanwise_idx = cp_index[1]
                    
                    # Check if we have a valid average circulation for this spanwise section
                    if spanwise_idx in blade_data['Gamma_avg']:
                        # Get the strength ratio for this control point
                        lambda_ratio = blade_data['lambda_ratio'][cp_index]
                        
                        # Apply the ratio to get the final circulation value
                        final_gamma = blade_data['Gamma_avg'][spanwise_idx] * lambda_ratio
                        blade_data['Gamma'][cp_index] = final_gamma
        
        # Re-calculate pressure differences with final circulation values
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
        # This passes the updated circulation values to the wake system for time-marching
        if time_step > 0:
            self.update_wake(
                quad_propeller_mesh, 
                time_step, 
                dt, 
                body_velocity, 
                omega_dict, 
                wind_field, 
                com_position
            )
        
        if not converged:
            print(f"Viscous coupling did not converge after {max_iterations} iterations. Max gamma diff: {max_gamma_diff}")
        else:
            print("Viscous coupling completed successfully.")
            
        return forces_and_moments

    def update_wake(self, propeller_mesh, time_step, dt, body_velocity, omega_dict, wind_field, com_position):
        """Update wake system with GPU acceleration."""
        MAX_WAKE_LENGTH = 360
        wind_func = WindField.update_wind_function(wind_field, com_position)
        
        if time_step > 2:
            for propeller_key, propeller_data in self.wake_system.items():
                effective_omega = torch.tensor(omega_dict[propeller_key], device=self.device)
                hub_position = torch.tensor(propeller_mesh[propeller_key]['Hub Position'], device=self.device)
                
                for blade_key, blade_data in propeller_data.items():
                    existing_indices = list(blade_data.keys())
                    max_spanwise = max(idx[0] for idx in blade_data.keys())
                    
                    if max_spanwise >= MAX_WAKE_LENGTH - 1:
                        for chordwise_idx in set(idx[1] for idx in existing_indices if idx[0] == max_spanwise):
                            del blade_data[(max_spanwise, chordwise_idx)]
                        max_spanwise = MAX_WAKE_LENGTH - 2
                    
                    for spanwise in range(max_spanwise, -1, -1):
                        for chordwise_idx in set(idx[1] for idx in existing_indices if idx[0] == spanwise):
                            old_idx = (spanwise, chordwise_idx)
                            new_idx = (spanwise + 1, chordwise_idx)
                            
                            if old_idx in blade_data and new_idx[0] < MAX_WAKE_LENGTH:
                                blade_data[new_idx] = copy.deepcopy(blade_data[old_idx])
                
                gamma_index = 0
                for blade_key, blade_data in propeller_data.items():
                    for wake_idx in list(blade_data.keys()):
                        if wake_idx[0] >= MAX_WAKE_LENGTH:
                            continue
                            
                        if 'Control Points' in blade_data[wake_idx] and blade_data[wake_idx]['Control Points'] is not None:
                            vertices = torch.tensor(blade_data[wake_idx]['Vortex Rings']['Vertices'], device=self.device)
                            control_point = torch.tensor(blade_data[wake_idx]['Control Points'], device=self.device)
                            
                            wind_velocity = -torch.tensor(wind_func(control_point.cpu().numpy()), device=self.device)
                            
                            for k in range(4):
                                vertex = vertices[k]
                                radius_vector = vertex - hub_position
                                omega_cross_r = torch.linalg.cross(effective_omega, radius_vector)
                                
                                max_span = max(idx[0] for idx in propeller_mesh[propeller_key]['Blades'][blade_key]['Vortex Rings'].keys())
                                max_chord = wake_idx[1]
                                induced = torch.tensor(
                                    propeller_mesh[propeller_key]['Blades'][blade_key]['Induced_Velocities'][max_span, max_chord],
                                    device=self.device
                                )
                                
                                vertex_velocity = induced
                                blade_data[wake_idx]['Vortex Rings']['Vertices'][k] = \
                                    (vertices[k] + vertex_velocity * dt).cpu().numpy()
                            
                            control_point_velocity = induced
                            blade_data[wake_idx]['Control Points'] = \
                                (control_point + control_point_velocity * dt).cpu().numpy()
                            
                    gamma_index += 1

        # Create new wake panels
        for propeller_key, propeller_data in propeller_mesh.items():
            for blade_key, blade_data in propeller_data['Blades'].items():
                max_spanwise_index = max(idx[0] for idx in blade_data['Vortex Rings'].keys())
                
                for panel_index, vortex_ring in blade_data['Vortex Rings'].items():
                    if panel_index[0] == max_spanwise_index:
                        chordwise_idx = panel_index[1]
                        vertices = torch.tensor(vortex_ring['Vertices'], device=self.device)
                        
                        if time_step == 1:
                            self.wake_system[propeller_key][blade_key][(0, chordwise_idx)] = {
                                'Gamma': None,
                                'Control Points': None,
                                'Time History': int(time_step),
                                'Vortex Rings': {
                                    'Vertices': torch.stack([
                                        vertices[1],
                                        vertices[1],
                                        vertices[2],
                                        vertices[2]
                                    ]).cpu().numpy()
                                }
                            }
                        elif time_step >= 2:
                            wake_panel = self.wake_system[propeller_key][blade_key][(0, chordwise_idx)]
                            wake_vertices = torch.tensor(wake_panel['Vortex Rings']['Vertices'], device=self.device)
                            
                            new_vertices = torch.stack([
                                vertices[1],
                                wake_vertices[0],
                                wake_vertices[3],
                                vertices[2]
                            ])
                            
                            wake_panel['Vortex Rings']['Vertices'] = new_vertices.cpu().numpy()
                            
                            span_mid = (new_vertices[0] + new_vertices[3]) / 2
                            chord_mid = (new_vertices[1] + new_vertices[2]) / 2
                            control_point = span_mid + 0.5 * (chord_mid - span_mid)
                            
                            wake_panel['Control Points'] = control_point.cpu().numpy()
                            wake_panel['Gamma'] = blade_data['Gamma'][panel_index]
                            wake_panel['Time History'] = int(time_step)
                                 
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
                        gamma = blade_data['Gamma'][cp_index]
                        
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

    def plot_wake_system(self, quad_propeller_mesh, wake_system, propeller_key="Propeller_1", 
                        show_bound=True, azimuth=0, elevation=0, save_plot=False):
        """
        Plot the wake system showing only root wake panels.
        """
        # Create figure with proper spacing for colorbar
        fig = plt.figure(figsize=(15, 10))
        gs = plt.GridSpec(1, 20)
        ax = fig.add_subplot(gs[0, :19], projection='3d')
        cax = fig.add_subplot(gs[0, 19])

        # Plot bound system if requested
        if show_bound:
            propeller_data = quad_propeller_mesh[propeller_key]
            for blade_key, blade_data in propeller_data['Blades'].items():
                for panel_index, vortex_ring in blade_data['Vortex Rings'].items():
                    vertices = vortex_ring['Vertices']
                    for i in range(4):
                        start = vertices[i]
                        end = vertices[(i + 1) % 4]
                        ax.plot([start[0], end[0]], 
                                [start[1], end[1]], 
                                [start[2], end[2]], 
                                'b-', alpha=0.5, linewidth=0.5)

        # Plot wake system
        wake_data = wake_system[propeller_key]
        max_age = 0
        
        # Find maximum wake age for color scaling
        for blade_data in wake_data.values():
            for wake_index in blade_data.keys():
                max_age = max(max_age, wake_index[0])
        
        cmap = plt.cm.viridis
        
        for blade_key, blade_data in wake_data.items():
            for wake_index, wake_panel in blade_data.items():
                # Only plot root wake panels (wake_index[1] == 0 or 1)
             
                    
                vertices = wake_panel['Vortex Rings']['Vertices']
                color = cmap(wake_index[0] / max_age if max_age > 0 else 0)
                
                # Connect wake vertices with thinner lines
                for i in range(4):
                    start = vertices[i]
                    end = vertices[(i + 1) % 4]
                    ax.plot([start[0], end[0]], 
                            [start[1], end[1]], 
                            [start[2], end[2]], 
                            color=color, alpha=0.7, linewidth=0.5)

        # Add legend with line segments
        from matplotlib.lines import Line2D
        bound_line = Line2D([0], [0], color='blue', alpha=0.5, linewidth=0.5)
        wake_lines = [Line2D([0], [0], color=cmap(i/3), alpha=0.7, linewidth=0.5) 
                    for i in range(4)]
        ax.legend([bound_line] + wake_lines, 
                ['Bound Vortices', 'New Wake', 'Young Wake', 'Medium Wake', 'Old Wake'])

        # Customize plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Root Wake System Visualization - {propeller_key}\nAzimuth: {azimuth}°, Elevation: {elevation}°')
        
        ax.view_init(elev=elevation, azim=azimuth)
        
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

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_age))
        sm.set_array([])
        plt.colorbar(sm, cax=cax, label='Wake Age')
        
        plt.tight_layout()
        
        if save_plot:
            plt.savefig(f'root_wake_system_{propeller_key}_az{azimuth}_el{elevation}.png', 
                        dpi=300, bbox_inches='tight')
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
        n = 90
        rho = 1.07178
        d = 0.24

        # Extract z-component (thrust) from Propeller_1 force
        thrust = (forces_and_moments['Propeller_1']['force'][2]) / (rho * n**2  * d**4)
        
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
        for blade_key in ['Blade_1', 'Blade_2']:
            blade_data = propeller_data['Blades'][blade_key]
            all_pressures.extend(list(blade_data['Pressure Difference'].values()))
        
        max_abs_pressure = max(abs(min(all_pressures)), abs(max(all_pressures)))
        
        # Process blades with improved interpolation
        for blade_key in ['Blade_1', 'Blade_2']:
            blade_data = propeller_data['Blades'][blade_key]
            
            for panel_index, panel in blade_data['Panels'].items():
                panel_2d = [(-p[1], p[0]) for p in panel]
                patch = Polygon(panel_2d, closed=True)
                patches.append(patch)
                
                pressure = blade_data['Pressure Difference'][panel_index]
                normalized_pressure = pressure / max_abs_pressure if max_abs_pressure != 0 else 0
                pressure_values.append(normalized_pressure)
        
        # Use coolwarm colormap and adjust shading for smoother appearance
        collection = PatchCollection(patches, cmap='viridis', alpha=1.0, edgecolors='face', 
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
        all_points = np.array([(-point[1], point[0]) for blade_key in ['Blade_1', 'Blade_2'] 
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

    def get_radial_thrust_line(self, blade_data, propeller_key, quad_propeller_mesh, debug=False):
        """
        Gets a single line of thrust values along the blade radius, accounting for hub position.
        
        Args:
            blade_data: Dictionary containing blade information
            propeller_key: Key for the propeller (e.g., 'Propeller_1')
            quad_propeller_mesh: Complete mesh containing hub positions
            debug: Boolean to print debug information
        """
        if debug:
            print("\nCollecting radial thrust distribution...")
        
        # Get hub position for this propeller
        hub_position = np.array(quad_propeller_mesh[propeller_key]['Hub Position'])
        
        if debug:
            print(f"Hub position for {propeller_key}: {hub_position}")
        
        # Initialize storage for thrust at each spanwise position
        spanwise_thrust = {}
        spanwise_radial = {}
        
        # Get unique spanwise indices
        spanwise_indices = sorted(set(idx[1] for idx in blade_data['Panel Forces'].keys()))
        
        # Sum thrust along chord for each spanwise position
        for span_idx in spanwise_indices:
            spanwise_thrust[span_idx] = 0.0
            
            # Get all chord indices for this span position
            chord_indices = [idx[0] for idx in blade_data['Panel Forces'].keys() 
                            if idx[1] == span_idx]
            
            # Sum all chord positions for this spanwise location
            for chord_idx in chord_indices:
                panel_idx = (chord_idx, span_idx)
                force = blade_data['Panel Forces'][panel_idx]
                spanwise_thrust[span_idx] += force[2]  # z-component for thrust
                
                # Store radial position if not already stored
                if span_idx not in spanwise_radial:
                    control_point = blade_data['Control Points'][panel_idx]
                    # Subtract hub position before calculating radius
                    adjusted_point = control_point - hub_position
                    r = np.sqrt(adjusted_point[0]**2 + adjusted_point[1]**2)
                    spanwise_radial[span_idx] = r
                    
                    if debug:
                        print(f"\nSpan index {span_idx}:")
                        print(f"Control point: {control_point}")
                        print(f"Adjusted point: {adjusted_point}")
                        print(f"Radial position: {r:.6f} m")
            
            if debug:
                print(f"Total thrust at r = {spanwise_radial[span_idx]:.6f}: {spanwise_thrust[span_idx]:.6f} N")
        
        # Convert to arrays, sort by radial position
        radial_positions = np.array([spanwise_radial[idx] for idx in spanwise_indices])
        thrust_values = np.array([spanwise_thrust[idx] for idx in spanwise_indices])
        sort_idx = np.argsort(radial_positions)
        
        if debug:
            print("\nFinal sorted distribution:")
            for r, t in zip(radial_positions[sort_idx], thrust_values[sort_idx]):
                print(f"r = {r:.6f} m, thrust = {t:.6f} N")
        
        return radial_positions[sort_idx], thrust_values[sort_idx]
    
    def get_radial_segments(self, blade_data, debug=False):
        """
        Gets thrust values for radial segments starting from hub.
        
        Args:
            blade_data: Dictionary containing blade information
            debug: Boolean to print debug information
            
        Returns:
            segment_lengths: Length of each radial segment
            thrust_values: Thrust value for each segment
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
            chord_indices = [idx[0] for idx in blade_data['Panel Forces'].keys() 
                            if idx[1] == span_idx]
            
            # Calculate total thrust for this segment
            total_thrust = 0.0
            for chord_idx in chord_indices:
                panel_idx = (chord_idx, span_idx)
                force = blade_data['Panel Forces'][panel_idx]
                total_thrust += force[2]  # z-component
                
                # For the first chord index, calculate panel length
                if chord_idx == chord_indices[0]:
                    panel = np.array(blade_data['Panels'][panel_idx])
                    # Calculate panel length (average of leading and trailing edge)
                    edge1_length = np.linalg.norm(panel[2] - panel[0])
                    edge2_length = np.linalg.norm(panel[3] - panel[1])
                    panel_length = (edge1_length + edge2_length) / 2
                    segment_lengths.append(panel_length)
            
            thrust_values.append(total_thrust)
            
            if debug:
                print(f"\nSegment {len(segment_lengths)}:")
                print(f"Length: {panel_length:.6f} m")
                print(f"Thrust: {total_thrust:.6f} N")
        
        if debug:
            print("\nFinal distribution:")
            total_length = 0.00624  # Start from hub
            for i, (length, thrust) in enumerate(zip(segment_lengths, thrust_values)):
                print(f"Segment {i+1}:")
                print(f"Starts at r = {total_length:.6f} m")
                print(f"Length = {length:.6f} m")
                print(f"Thrust = {thrust:.6f} N")
                total_length += length
        
        return np.array(segment_lengths), np.array(thrust_values)



        # Function to collect thrust data during iteration
    
    def collect_thrust_data(self, blade_data_1, blade_data_2, angle, debug=False):
        """
        Collect thrust data for both blades at a given angle.
        
        Args:
            blade_data_1: Dictionary containing Blade_1 information
            blade_data_2: Dictionary containing Blade_2 information
            angle: Current angle in radians
            debug: Print debug information
            
        Returns:
            dict: Thrust data for this angle step
        """
        # Get thrust lines for both blades
        segments_1, thrust_1 = self.get_radial_segments(blade_data_1)
        segments_2, thrust_2 = self.get_radial_segments(blade_data_2)
        
        if debug:
            print(f"\nCollecting data at angle: {np.degrees(angle):.1f} degrees")
            print("Blade 1:", thrust_1)
            print("Blade 2:", thrust_2)
        
        return {
            'angle': angle,
            'blade_1': {'segments': segments_1, 'thrust': thrust_1},
            'blade_2': {'segments': segments_2, 'thrust': thrust_2}
        }

        