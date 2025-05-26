from propeller import PropellerGeometry
from mesh import PropellerMesh
from wind import WindField
from vpm_cuda import VPM

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

import copy
import numpy as np
import cupy as cp
import numba as nb
from numba import cuda, float64, int32
import math

@cuda.jit(device=True)
def biot_savart(r1, r2, r0, gamma, result): 
    """GPU-accelerated Biot-Savart calculation"""
    # Initialize local array for cross product
    cross_r1_r2 = cuda.local.array(3, nb.float64)
    
    # Manual cross product
    cross_r1_r2[0] = r1[1]*r2[2] - r1[2]*r2[1]
    cross_r1_r2[1] = r1[2]*r2[0] - r1[0]*r2[2]
    cross_r1_r2[2] = r1[0]*r2[1] - r1[1]*r2[0]
    
    norm_cross = math.sqrt(cross_r1_r2[0]**2 + cross_r1_r2[1]**2 + cross_r1_r2[2]**2)
    norm_r0 = math.sqrt(r0[0]**2 + r0[1]**2 + r0[2]**2)
    norm_r1 = math.sqrt(r1[0]**2 + r1[1]**2 + r1[2]**2)
    norm_r2 = math.sqrt(r2[0]**2 + r2[1]**2 + r2[2]**2)


    # Core parameters
    v = 1.48e-5  # viscosity
    eps = 1.25643  # Oseen parameter
    a = 2 * 10e-4
    n = 2  # Core model exponent
    # sigma = 1 + (a * (gamma / v))
    rc0 = 2.92e-4
    a_tol = 1e-6
    # Calculate core radius based on time step
    # if step_diff < 3:
    #     rc = 2.92e-4
    # else:
    #     rc = math.sqrt(rc0**2 + (4.0 * eps * sigma * v * step_diff))
    if norm_cross < a_tol or norm_r0 < a_tol:
        for i in range(3):
            result[i] = 0.0
        return
    
    if norm_r1 < a_tol or norm_r2 < a_tol:
        for i in range(3):
            result[i] = 0.0
        return
    
    rc = 2.92e-4
    n = 2
    
    denominator = ((norm_cross**(2*n)) + ((rc*norm_r0)**(2*n)))**(1/n)
    # denominator = ((norm_cross**(2)))


    h = norm_cross / norm_r0
    Kv = h**2 / math.sqrt(h**4 + rc**4)
    # Manual dot product
    norm_r1 = math.sqrt(r1[0]**2 + r1[1]**2 + r1[2]**2)
    norm_r2 = math.sqrt(r2[0]**2 + r2[1]**2 + r2[2]**2)
    dot_term = r0[0]*(r1[0]/norm_r1 - r2[0]/norm_r2) + \
               r0[1]*(r1[1]/norm_r1 - r2[1]/norm_r2) + \
               r0[2]*(r1[2]/norm_r1 - r2[2]/norm_r2)
    
    # Initialize result array
    induced_velocity = cuda.local.array(3, nb.float64)
    factor = (gamma / (4 * math.pi)) * dot_term / denominator
    
    # Compute final velocity
    result[0] = factor * cross_r1_r2[0]
    result[1] = factor * cross_r1_r2[1]
    result[2] = factor * cross_r1_r2[2]

@cuda.jit(device=True)
def biot_savart_wake(r1, r2, r0, dt, step_diff, gamma, result): 
    """GPU-accelerated Biot-Savart calculation with viscous core model for wake"""
    # Initialize local array for cross product
    cross_r1_r2 = cuda.local.array(3, nb.float64)
    
    # Manual cross product
    cross_r1_r2[0] = r1[1]*r2[2] - r1[2]*r2[1]
    cross_r1_r2[1] = r1[2]*r2[0] - r1[0]*r2[2]
    cross_r1_r2[2] = r1[0]*r2[1] - r1[1]*r2[0]
    
    norm_cross = math.sqrt(cross_r1_r2[0]**2 + cross_r1_r2[1]**2 + cross_r1_r2[2]**2)
    norm_r0 = math.sqrt(r0[0]**2 + r0[1]**2 + r0[2]**2)
    norm_r1 = math.sqrt(r1[0]**2 + r1[1]**2 + r1[2]**2)
    norm_r2 = math.sqrt(r2[0]**2 + r2[1]**2 + r2[2]**2)

    a_tol = 10e-6
    if norm_cross < a_tol or norm_r0 < a_tol:
        for i in range(3):
            result[i] = 0.0
        return
    
    if norm_r1 < a_tol or norm_r2 < a_tol:
        for i in range(3):
            result[i] = 0.0
        return
    # Core parameters
    v = 1.48e-5  # viscosity
    eps = 1.25643  # Oseen parameter
    a = 1 * 10e-4
    n = 2  # Core model exponent
    sigma = 1 + (a * (0.3 / v))

    # sigma = 200

    rc0 = 2.92e-4
    # rc0 = norm_r0
    # rc = math.sqrt(rc0**2 + (4.0 * eps * sigma * v * step_diff * dt))
    # Calculate core radius based on time step
    # if step_diff < 3:
    #     rc = 2.92e-4
    # else:
    #     rc = math.sqrt(rc0**2 + (4.0 * eps * sigma * v * step_diff * dt))

    # if norm_cross < a_tol:
    #     return 0.0
    rc = rc0
    # Denominator with viscous core correction
    # denominator = ((norm_cross**(2*n)) + ((rc*norm_r0)**(2*n)))**(1/n)
    denominator = ((norm_cross**(2)))
    test = 1 - 2.7814**(-eps * (norm_cross / (norm_r0 * rc))**2)

    h = norm_cross / norm_r0
    Kv = h**2 / math.sqrt(h**4 + rc**4)
    # Manual dot product
    norm_r1 = math.sqrt(r1[0]**2 + r1[1]**2 + r1[2]**2)
    norm_r2 = math.sqrt(r2[0]**2 + r2[1]**2 + r2[2]**2)
    dot_term = r0[0]*(r1[0]/norm_r1 - r2[0]/norm_r2) + \
               r0[1]*(r1[1]/norm_r1 - r2[1]/norm_r2) + \
               r0[2]*(r1[2]/norm_r1 - r2[2]/norm_r2)
    
    # Compute final velocity
    factor = Kv * (gamma / (4.0 * math.pi)) * dot_term / denominator
    result[0] = factor * cross_r1_r2[0]
    result[1] = factor * cross_r1_r2[1]
    result[2] = factor * cross_r1_r2[2]

@cuda.jit
def bound_velocity_kernel(control_points, vortex_rings, result_matrix):
    i, j = cuda.grid(2)
    
    if i < control_points.shape[0] and j < vortex_rings.shape[0]:
        r1 = cuda.local.array(3, nb.float64)
        r2 = cuda.local.array(3, nb.float64)
        r0 = cuda.local.array(3, nb.float64)
        total_induced = cuda.local.array(3, nb.float64)
        induced = cuda.local.array(3, nb.float64)
        
        # Initialize total induced velocity
        for k in range(3):
            total_induced[k] = 0.0
        
        # Process each vortex filament
        for k in range(4):
            # Get vertices
            for n in range(3):
                vertex_start_n = vortex_rings[j, k, n]
                vertex_end_n = vortex_rings[j, (k + 1) % 4, n]
                
                # Calculate vectors
                r1[n] = control_points[i, n] - vertex_end_n
                r2[n] = control_points[i, n] - vertex_start_n
                r0[n] = vertex_start_n - vertex_end_n
            
            # Pass induced array to biot_savart
            biot_savart(r1, r2, r0, 1.0, induced)
            
            # Add to total
            for n in range(3):
                total_induced[n] += induced[n]
        
        # Store result
        for k in range(3):
            result_matrix[i, j, k] = total_induced[k]

@cuda.jit
def wake_velocity_kernel(control_points, wake_vortex_rings, wake_gamma, wake_time_steps, time_step, dt, result_matrix):
    """GPU-accelerated kernel for wake to bound induced velocity calculation"""
    i, j = cuda.grid(2)
    
    if i < control_points.shape[0] and j < wake_vortex_rings.shape[0]:
        r1 = cuda.local.array(3, nb.float64)
        r2 = cuda.local.array(3, nb.float64)
        r0 = cuda.local.array(3, nb.float64)
        total_induced = cuda.local.array(3, nb.float64)
        induced = cuda.local.array(3, nb.float64)
        
        # Initialize total induced velocity
        for k in range(3):
            total_induced[k] = 0.0
        
        # Process each vortex filament
        for k in range(4):
            # Get vertices
            for n in range(3):
                vertex_start_n = wake_vortex_rings[j, k, n]
                vertex_end_n = wake_vortex_rings[j, (k + 1) % 4, n]
                
                # Calculate vectors
                r1[n] = control_points[i, n] - vertex_end_n
                r2[n] = control_points[i, n] - vertex_start_n
                r0[n] = vertex_start_n - vertex_end_n
            
            # Calculate time step difference
            step_diff = time_step - wake_time_steps[j]
            
            # Pass induced array to biot_savart with viscous core model
            biot_savart_wake(r1, r2, r0, dt, step_diff, wake_gamma[j], induced)
            # biot_savart(r1, r2, r0, wake_gamma[j], induced)
            # Add to total
            for n in range(3):
                total_induced[n] += induced[n]
        
        # Store result
        for k in range(3):
            result_matrix[i, j, k] = total_induced[k]

@cuda.jit
def bound_to_wake_velocity_kernel(wake_vertices, bound_vortex_rings, result_matrix, wake_gamma, wake_time_steps, dt, time_step):
    """GPU-accelerated kernel for calculating bound to wake vertex induced velocities"""
    i, j = cuda.grid(2)
    
    if i < wake_vertices.shape[0] and j < bound_vortex_rings.shape[0]:
        r1 = cuda.local.array(3, nb.float64)
        r2 = cuda.local.array(3, nb.float64)
        r0 = cuda.local.array(3, nb.float64)
        total_induced = cuda.local.array(3, nb.float64)
        induced = cuda.local.array(3, nb.float64)
        
        # Initialize total induced velocity
        for k in range(3):
            total_induced[k] = 0.0
        
        # Process each vortex filament
        for k in range(4):
            # Get vertices
            for n in range(3):
                vertex_start_n = bound_vortex_rings[j, k, n]
                vertex_end_n = bound_vortex_rings[j, (k + 1) % 4, n]
                
                # Calculate vectors
                r1[n] = wake_vertices[i, n] - vertex_start_n
                r2[n] = wake_vertices[i, n] -  vertex_end_n
                r0[n] = vertex_start_n - vertex_end_n
            
            step_diff = time_step - wake_time_steps[j]

            # Pass induced array to biot_savart
            # biot_savart_wake(r1, r2, r0, dt, step_diff, wake_gamma[j], induced)
            biot_savart(r1, r2, r0, 1.0, induced)

            # Add to total
            for n in range(3):
                total_induced[n] += induced[n]
        
        # Store result
        for k in range(3):
            result_matrix[i, j, k] = total_induced[k]

@cuda.jit
def wake_to_wake_velocity_kernel(wake_vertices, wake_vortex_rings, result_matrix, wake_gamma, wake_time_steps, current_step, dt):
    """GPU-accelerated kernel for calculating wake to wake vertex induced velocities"""
    i, j = cuda.grid(2)
    
    if i < wake_vertices.shape[0] and j < wake_vortex_rings.shape[0]:
        r1 = cuda.local.array(3, nb.float64)
        r2 = cuda.local.array(3, nb.float64)
        r0 = cuda.local.array(3, nb.float64)
        total_induced = cuda.local.array(3, nb.float64)
        induced = cuda.local.array(3, nb.float64)
        
        # Initialize total induced velocity
        for k in range(3):
            total_induced[k] = 0.0
        
        # Process each vortex filament
        for k in range(4):
            # Get vertices
            for n in range(3):
                vertex_start_n = wake_vortex_rings[j, k, n]
                vertex_end_n = wake_vortex_rings[j, (k + 1) % 4, n]
                
                # Calculate vectors
                r1[n] = wake_vertices[i, n] - vertex_end_n
                r2[n] = wake_vertices[i, n] - vertex_start_n
                # r0[n] = vertex_start_n - vertex_end_n
                r0[n] =  r1[n] - r2[n]

            step_diff = (current_step - wake_time_steps[j])

            gamma_value = (wake_gamma[j])
            
            biot_savart_wake(r1, r2, r0, dt, step_diff, gamma_value, induced)
            
            # Add to total
            for n in range(3):
                total_induced[n] += induced[n]
        
        # Store result
        for k in range(3):
            result_matrix[i, j, k] = total_induced[k]


class UVLM:
    def __init__(self, propeller_mesh):
        """Initialize the UVLM solver with the given propeller mesh."""
        self.propeller_mesh = propeller_mesh
        self.wake_system = {}
        # self.bound_to_bound_global_matrices = self.calculate_bound_to_bound_induced_velocity_matrix(propeller_mesh)
        # self.initialize_wake_system(propeller_mesh)         

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

    def calculate_bound_to_bound_induced_velocity_matrix(self, quad_propeller_mesh, omega_dict):
        global_matrices = {}
        
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            # Collect data
            control_points = []
            vortex_rings = []
            
            for blade_data in propeller_data['Blades'].values():
                for cp_index, control_point in blade_data['Control Points'].items():
                    control_points.append(control_point)
                    vortex_rings.append(blade_data['Vortex Rings'][cp_index]['Vertices'])
            
            # Convert to numpy arrays
            control_points = np.array(control_points, dtype=np.float64)
            vortex_rings = np.array(vortex_rings, dtype=np.float64)
            
            threadsperblock = (16, 16)  # Reduced from (32, 32)
        
            # Calculate grid size more conservatively
            max_threads = 512  # Maximum threads per block
            thread_count = threadsperblock[0] * threadsperblock[1]
            if thread_count > max_threads:
                threadsperblock = (16, 8)  # Further reduce if needed
                
            blockspergrid_x = math.ceil(control_points.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(vortex_rings.shape[0] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)

            
            # Allocate result matrix
            result_matrix = np.zeros((control_points.shape[0], vortex_rings.shape[0], 3), dtype=np.float64)
            
            # Transfer data to GPU
            d_control_points = cuda.to_device(control_points)
            d_vortex_rings = cuda.to_device(vortex_rings)
            d_result_matrix = cuda.to_device(result_matrix)
            
            # Launch kernel
            bound_velocity_kernel[blockspergrid, threadsperblock](
                d_control_points, d_vortex_rings, d_result_matrix
            )
            
            # Get result back
            result_matrix = d_result_matrix.copy_to_host()
            global_matrices[propeller_key] = result_matrix
        
        return global_matrices
  
    def calculate_wake_to_bound_induced_velocity_matrix(self, quad_propeller_mesh, dt, time_step):
        """Calculate induced velocities from wake vortex rings onto bound control points using CUDA"""
        final_induced_velocities = {}
        
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            # Collect bound control points
            control_points = []
            wake_vortex_rings = []
            wake_gamma_values = []
            wake_time_steps = []
            
            # Get bound control points
            for blade_data in propeller_data['Blades'].values():
                for control_point in blade_data['Control Points'].values():
                    control_points.append(control_point)
            
            # Get wake vortex rings and their gamma values
            wake_data = self.wake_system[propeller_key]
            for blade_data in wake_data.values():
                for wake_panel in blade_data.values():
                    if wake_panel['Vortex Rings']['Vertices'] is not None:
                        wake_vortex_rings.append(wake_panel['Vortex Rings']['Vertices'])
                        wake_gamma_values.append(wake_panel['Gamma'])
                        wake_time_steps.append(wake_panel['Time History'])
            
            # Convert to numpy arrays
            control_points = np.array(control_points, dtype=np.float64)
            wake_vortex_rings = np.array(wake_vortex_rings, dtype=np.float64)
            wake_gamma = np.array(wake_gamma_values, dtype=np.float64)
            wake_time_steps = np.array(wake_time_steps, dtype=np.float64)
            
            # Set up CUDA grid
            threadsperblock = (16, 16)
            blockspergrid_x = math.ceil(control_points.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(wake_vortex_rings.shape[0] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            
            # Allocate result matrix
            result_matrix = np.zeros((control_points.shape[0], wake_vortex_rings.shape[0], 3), dtype=np.float64)
            
            # Transfer data to GPU
            d_control_points = cuda.to_device(control_points)
            d_wake_vortex_rings = cuda.to_device(wake_vortex_rings)
            d_wake_gamma = cuda.to_device(wake_gamma)
            d_wake_time_steps = cuda.to_device(wake_time_steps)
            d_result_matrix = cuda.to_device(result_matrix)
            
            # Launch kernel
            wake_velocity_kernel[blockspergrid, threadsperblock](
                d_control_points, d_wake_vortex_rings, d_wake_gamma,
                d_wake_time_steps, time_step, dt, d_result_matrix
            )
            
            # Get result back
            result_matrix = d_result_matrix.copy_to_host()
            
            # Calculate final induced velocities
            induced_velocities = np.einsum('ijk->ik', result_matrix)
            final_induced_velocities[propeller_key] = induced_velocities
        
        return final_induced_velocities

    def calculate_bound_to_wake_induced_velocity_matrix(self, quad_propeller_mesh, dt, time_step):
        """
        Calculate induced velocities from bound vortex rings onto wake control points using CUDA
        """
        final_induced_velocities = {}
        
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            # Collect bound vortex rings and gamma values
            bound_vortex_rings = []
            bound_gamma_values = []
            wake_gamma_values = []
            wake_time_steps = []

            for blade_data in propeller_data['Blades'].values():
                for cp_index in blade_data['Control Points'].keys():
                    bound_vortex_rings.append(blade_data['Vortex Rings'][cp_index]['Vertices'])
                    
                    bound_gamma_values.append(blade_data['Gamma'][cp_index])
            
            # Get wake control points
            wake_vertices = []
            wake_data = self.wake_system[propeller_key]
            
            for blade_data in wake_data.values():
                for wake_panel in blade_data.values():
                    # if wake_panel['Vortex Rings']['Vertices'] is not None:
                    #     # Append all four vertices of the wake panel
                    for vertex in wake_panel['Vortex Rings']['Vertices']:
                        wake_vertices.append(vertex)
                        if wake_panel['Vortex Rings']['Vertices'] is not None:
                            wake_time_steps.append(wake_panel['Time History'])
                            wake_gamma_values.append(wake_panel['Gamma'])
            
                # Convert to numpy arrays
            wake_vertices = np.array(wake_vertices, dtype=np.float64)
            bound_vortex_rings = np.array(bound_vortex_rings, dtype=np.float64)
            bound_gamma = np.array(bound_gamma_values, dtype=np.float64)
            wake_time_steps = np.array(wake_time_steps, dtype=np.float64)
            wake_gamma = np.array(wake_gamma_values, dtype=np.float64)
            
            # Set up CUDA grid
            threadsperblock = (16, 16)
            blockspergrid_x = math.ceil(wake_vertices.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(bound_vortex_rings.shape[0] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            
            # Allocate result matrix
            result_matrix = np.zeros((wake_vertices.shape[0], bound_vortex_rings.shape[0], 3), dtype=np.float64)
            
            # Transfer data to GPU
            d_wake_vertices = cuda.to_device(wake_vertices)
            d_bound_vortex_rings = cuda.to_device(bound_vortex_rings)
            d_result_matrix = cuda.to_device(result_matrix)
            d_wake_time_steps = cuda.to_device(wake_time_steps)
            d_wake_gamma = cuda.to_device(wake_gamma)

            # Launch kernel
            bound_to_wake_velocity_kernel[blockspergrid, threadsperblock](
                d_wake_vertices, d_bound_vortex_rings, d_result_matrix, d_wake_gamma, d_wake_time_steps, dt, time_step
            )
            
            # Get result back
            result_matrix = d_result_matrix.copy_to_host()
            
            # Calculate final induced velocities
            induced_velocities = np.einsum('ijk,j->ik', result_matrix, bound_gamma.flatten())
            final_induced_velocities[propeller_key] = induced_velocities
        
        return final_induced_velocities
    
    def calculate_wake_to_wake_induced_velocity_matrix(self, dt, time_step):
        """
        Calculate induced velocities from wake vortex rings onto wake vertices using CUDA
        """
        final_induced_velocities = {}
        
        for propeller_key, wake_data in self.wake_system.items():
            # Collect wake vertices and vortex rings
            wake_vertices = []
            wake_vortex_rings = []
            wake_gamma_values = []
            wake_time_steps = []

            # First pass to collect all vertices
            for blade_data in wake_data.values():
                for wake_panel in blade_data.values():
                    # if wake_panel['Vortex Rings']['Vertices'] is not None:
                    #     # Add all four vertices from this wake panel
                    for vertex in wake_panel['Vortex Rings']['Vertices']:
                        wake_vertices.append(vertex)
                    wake_vortex_rings.append(wake_panel['Vortex Rings']['Vertices'])
                    wake_gamma_values.append(wake_panel['Gamma'])
                    wake_time_steps.append(wake_panel['Time History'])
            
            # Convert to numpy arrays
            wake_vertices = np.array(wake_vertices, dtype=np.float64)
            wake_vortex_rings = np.array(wake_vortex_rings, dtype=np.float64)
            wake_gamma = np.array(wake_gamma_values, dtype=np.float64)
            wake_time_steps = np.array(wake_time_steps, dtype=np.float64)
            
            # if len(wake_vertices) == 0:
            #     continue
            
            # Set up CUDA grid
            threadsperblock = (16, 16)
            blockspergrid_x = math.ceil(wake_vertices.shape[0] / threadsperblock[0])
            blockspergrid_y = math.ceil(wake_vortex_rings.shape[0] / threadsperblock[1])
            blockspergrid = (blockspergrid_x, blockspergrid_y)
            
            # Allocate result matrix
            result_matrix = np.zeros((wake_vertices.shape[0], wake_vortex_rings.shape[0], 3), dtype=np.float64)
            
            # Transfer data to GPU
            d_wake_vertices = cuda.to_device(wake_vertices)
            d_wake_vortex_rings = cuda.to_device(wake_vortex_rings)
            d_result_matrix = cuda.to_device(result_matrix)
            d_wake_time_steps = cuda.to_device(wake_time_steps)
            d_wake_gamma = cuda.to_device(wake_gamma)

            # Launch kernel
            wake_to_wake_velocity_kernel[blockspergrid, threadsperblock](
                d_wake_vertices, d_wake_vortex_rings, d_result_matrix, d_wake_gamma, d_wake_time_steps, time_step, dt,
            )
            
            # Get result back
            result_matrix = d_result_matrix.copy_to_host()
            
            # Calculate final induced velocities
            induced_velocities = np.einsum('ijk->ik', result_matrix)
            final_induced_velocities[propeller_key] = induced_velocities
        
        return final_induced_velocities
            
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
        
        wind_func = WindField.update_wind_function(wind_field, com_position)

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

                    wind_velocity = -wind_func(control_point)

                    # Rotate normal vector
                    normal = blade_data['Normals'][cp_index]
                    rotated_normal =  normal
                    normals.append(rotated_normal)
                    
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
                    velocity_term = -omega_cross_r
                    rhs_value = -np.dot(velocity_term, normal)
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
            
            if time_step>2:
                wake_to_bound_induced_velocity_matrix = wake_to_bound_induced_velocity_matrices[propeller_key]
                Aw = np.einsum('ij,ij->i', wake_to_bound_induced_velocity_matrix, normals).reshape(-1, 1)
            else:    
                Aw = np.zeros_like(rhs)
            
            gamma = np.linalg.solve(Ab, (rhs - Aw))
            
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
                    
                    if time_step > 2:
                        # Get wake-induced velocities for this control point
                        wake_induced_vel = wake_to_bound_induced_velocity_matrices[propeller_key][gamma_index]
                        blade_data['Wake_Induced_Velocities'][cp_index] = wake_induced_vel

                    gamma_index += 1
 
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
                max_spanwise_index = max(idx[0] for idx in blade_data['Vortex Rings'].keys())
                max_chordwise_index = max(idx[1] for idx in blade_data['Vortex Rings'].keys())

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

                    if time_step>2:
                        wake_to_bound_induced_velocity = blade_data['Wake_Induced_Velocities'][panel_index]
                    else:    
                        wake_to_bound_induced_velocity = np.zeros_like(omega_cross_r)

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
                    total_velocity =  -omega_cross_r + wake_to_bound_induced_velocity

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

                    # force_1 = rho * (gamma_current-gamma_previous_span) * np.linalg.cross(total_velocity + bound_to_bound_induced_velocity, tangent_span)
                    # force_2 = rho * (gamma_current-gamma_previous_chord) * np.linalg.cross(total_velocity + bound_to_bound_induced_velocity, tangent_chord)
                    # force_steady = force_1 + force_2
                    # force_unsteady = rho * gamma_dot *  panel_area * normal

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
                    add_force_1 =0
                    add_force_2 = 0
                    pressure = rho * (
                        np.dot(total_velocity, tangent_chord / np.linalg.norm(tangent_chord)  * gamma_diff_span) +
                        np.dot(total_velocity, tangent_span / np.linalg.norm(tangent_span) * gamma_diff_chord)  +
                        gamma_dot   
                    )

                    
                  
                    if panel_index[0] == max_spanwise_index:
                        gamma_previous_span = gamma_dot
                        gamma_diff_span = (gamma_previous - gamma_current) / np.linalg.norm(tangent_chord)
                        pressure_1 = rho * (
                        np.dot(total_velocity, tangent_chord / np.linalg.norm(tangent_chord)  * gamma_diff_span) +
                        np.dot(total_velocity, tangent_span / np.linalg.norm(tangent_span) * gamma_diff_chord)  +
                        gamma_dot   
                    )
                        add_force_1 = np.linalg.norm(rho * (gamma_previous - gamma_current) * np.linalg.cross(total_velocity + bound_to_bound_induced_velocity, tangent_span)) /  panel_area
                        # print('success')
                    if panel_index[1] == max_chordwise_index:
                        gamma_diff_chord = (0 - gamma_current) / np.linalg.norm(tangent_span)
                        pressure_2 = rho * (
                        np.dot(total_velocity, tangent_chord / np.linalg.norm(tangent_chord)  * gamma_diff_span) +
                        np.dot(total_velocity, tangent_span / np.linalg.norm(tangent_span) * gamma_diff_chord)  +
                        gamma_dot   
                    )
                        add_force_2 = np.linalg.norm(rho * (0 - gamma_current) * np.linalg.cross(total_velocity + bound_to_bound_induced_velocity, tangent_chord)) /  panel_area
                        

                    # pressure = (np.cross(rho *  total_velocity * (gamma_current - gamma_previous_span), (tangent_span))) / panel_area
                    # pressure = (rho * np.linalg.norm(total_velocity) * (gamma_current - gamma_previous_span) * np.linalg.norm(tangent_span))
                    # pressure = -2 * ((total_velocity + induced_velocity) / total_velocity)**2
                    # pressure = np.linalg.norm(total_velocity) * rho * (gamma_current) * np.linalg.norm(tangent_span)
                    # pressure = rho * (gamma_current-gamma_previous_span) * np.linalg.cross((total_velocity + induced_velocity), tangent_span)
                    # pressure2 = rho * (gamma_current-gamma_previous_chord) * np.linalg.cross((total_velocity + induced_velocity), tangent_chord)
            
                    
                    pressure_difference[panel_index] = (pressure + add_force_2 + add_force_1) 
                
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
        print('bound_to_bound_global_matrices')
        wake_to_bound_induced_velocity_matrices = None

        # # Only calculate wake influences after time step 2
        if time_step > 2:
            wake_to_bound_induced_velocity_matrices = self.calculate_wake_to_bound_induced_velocity_matrix(propeller_mesh, dt, time_step)

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

                    # moment_arm_total = control_point  # Vector from origin to force application point
                    # moment = np.cross(moment_arm_total, force)

                    # moment_arm = control_point
                    # panel_moment = np.cross(moment_arm, force)
                    
                    
                    # Store the force and moment for this panel
                    blade_data['Panel Forces'][panel_index] = force
                    blade_data['Panel Moments'][panel_index] = force

                    # Accumulate to total force and moment
                    total_force += force
                    total_moment += force

            # Store total force and moment for this propeller
            # print(total_force)
            total_forces_and_moments[propeller_key] = {'force': total_force, 'moment': total_moment}
            
        print("Force", total_forces_and_moments['Propeller_1']['force'])
        # print("Forces", total_forces_and_moments['Propeller_2']['moment'])
        # print("Forces", total_forces_and_moments['Propeller_3']['moment'])
        # print("Forces", total_forces_and_moments['Propeller_4']['moment'])
        return total_forces_and_moments

    def update_wake(self, propeller_mesh, time_step, dt, body_velocity, omega_dict, wind_field, com_position):
        """
        Update wake system with a maximum of 10 spanwise panels by:
        1. First convecting existing wake panels
        2. Then shedding new wake panels
        3. Removing oldest panels if exceeding maximum length
        """
        MAX_WAKE_LENGTH = 400 #Maximum number of spanwise panels to maintain
        wind_func = WindField.update_wind_function(wind_field, com_position)
        
        # Step 1: Convect existing wake panels
        if time_step > 2:
            for propeller_key, wake_data in self.wake_system.items():
                effective_omega = omega_dict[propeller_key]
                hub_position = np.array(propeller_mesh[propeller_key]['Hub Position'])
                
                # Calculate induced velocities
                wake_to_wake_velocities = self.calculate_wake_to_wake_induced_velocity_matrix(dt, time_step)
                
                bound_to_wake_velocities = self.calculate_bound_to_wake_induced_velocity_matrix(propeller_mesh, dt, time_step)
                # induced = 5
                
                gamma_index = 0
                # Update positions of all wake panels
                for blade_key, blade_data in wake_data.items():
                    
                    for wake_idx in list(blade_data.keys()):  # Create list to avoid runtime modification issues

                        if (wake_idx[0] >= MAX_WAKE_LENGTH):
                            continue  # Skip panels beyond maximum length

                        control_point = blade_data[wake_idx]['Control Points']
                        # wind_velocity = -wind_func(control_point)
                            
                        vertices = blade_data[wake_idx]['Vortex Rings']['Vertices']
                            
                        # Update vertices
                        vertex_offset = gamma_index * 4
                        
                        for k in range(4):
                            vertex = vertices[k]
                            radius_vector = vertex - hub_position
                            omega_cross_r = np.cross(effective_omega, radius_vector)
                            max_span = max(idx[0] for idx in propeller_mesh[propeller_key]['Blades'][blade_key]['Vortex Rings'].keys())
                            max_chord = wake_idx[1]
                            induced = propeller_mesh[propeller_key]['Blades'][blade_key]['Induced_Velocities'][max_span, max_chord]
                            # print('induced veloctiy', induced)
                            vertex_velocity = (   
                                wake_to_wake_velocities[propeller_key][vertex_offset + k] 
                                # + bound_to_wake_velocities[propeller_key][vertex_offset + k]  
                                # + wind_velocity 
                                # 0
                                
                            )
                            # print(vertex_velocity)
                            # Update vertex position
                            blade_data[wake_idx]['Vortex Rings']['Vertices'][k] += vertex_velocity * dt
                        # Update control point
                        # control_point = blade_data[wake_idx]['Control Points']
                        # radius_vector = control_point - hub_position
                        # omega_cross_r = np.cross(effective_omega, radius_vector)
                        
                        # control_point_velocity = ( 
                            # wake_to_wake_velocities[propeller_key][vertex_offset + k] 
                        #     # - bound_to_wake_velocities[propeller_key][vertex_offset + k] 
                        #     # induced 
                        # )
                        
                        # blade_data[wake_idx]['Control Points'] += control_point_velocity * dt
                        gamma_index += 1

                for blade_key, blade_data in wake_data.items(): 

                    # Find maximum spanwise index
                    existing_indices = list(blade_data.keys())

                    max_spanwise = max(idx[0] for idx in blade_data.keys())
                    
                    # Remove panels beyond MAX_WAKE_LENGTH
                    if max_spanwise >= MAX_WAKE_LENGTH - 1:
                        # Remove the oldest panels
                        for chordwise_idx in set(idx[1] for idx in existing_indices if idx[0] == max_spanwise):
                            del blade_data[(max_spanwise, chordwise_idx)]
                        max_spanwise = MAX_WAKE_LENGTH - 2  # Adjust max_spanwise after removal
                    
                    # Move panels from back to front to avoid overwriting
                    for spanwise in range(max_spanwise, -1, -1):
                        for chordwise_idx in set(idx[1] for idx in existing_indices if idx[0] == spanwise):
                            old_idx = (spanwise, chordwise_idx)
                            new_idx = (spanwise + 1, chordwise_idx)
                            
                            # Move panel data to new index if within MAX_WAKE_LENGTH
                            if old_idx in blade_data and new_idx[0] < MAX_WAKE_LENGTH:
                                blade_data[new_idx] = copy.deepcopy(blade_data[old_idx])

        # Step 2: Shed new wake panels from trailing edge
        for propeller_key, propeller_data in propeller_mesh.items():
            for blade_key, blade_data in propeller_data['Blades'].items():
                # Find trailing edge panels
                max_spanwise_index = max(idx[0] for idx in blade_data['Vortex Rings'].keys())
                # Process each trailing edge panel
                for panel_index, vortex_ring in blade_data['Vortex Rings'].items():
                    if panel_index[0] == max_spanwise_index:
                        chordwise_idx = panel_index[1]
                        vertices = vortex_ring['Vertices']
                        
                        if time_step == 1:
                            # First time step: initialize first wake panel
                            self.wake_system[propeller_key][blade_key][(0, chordwise_idx)] = {
                                'Gamma': None,
                                'Control Points': None,
                                'Time History': int(time_step),
                                'Vortex Rings': {
                                    'Vertices': np.array([
                                        vertices[1],  # Front left
                                        vertices[1],  # Back left 
                                        vertices[2],  # Back right
                                        vertices[2]   # Front right 
                                    ])
                                }
                            }

                        elif time_step >= 2:
                            # Update existing wake panel and create connection
                            te_left = vertices[1]
                            te_right = vertices[2]

                            wake_panel_0 = self.wake_system[propeller_key][blade_key][(0, chordwise_idx)]
                            wake_vertices_0 = wake_panel_0['Vortex Rings']['Vertices']
                            
                            # Create new wake panel with 0.3 scaling factor
                            new_vertices = np.array([
                                te_left,          # New front left from trailing edge
                                te_left + 0.3 * (wake_vertices_0[0] - te_left),     # 30% of the way to previous front left
                                te_right + 0.3 * (wake_vertices_0[3] - te_right),   # 30% of the way to previous front right
                                te_right           # New front right from trailing edge
                            ])
                            
                            # Update the panel at index 0
                            wake_panel_0['Vortex Rings']['Vertices'] = new_vertices
                            wake_panel_0['Control Points'] = np.mean(new_vertices, axis=0)
                            wake_panel_0['Gamma'] = blade_data['Gamma'][panel_index]
                            wake_panel_0['Time History'] = int(time_step)

                            if time_step > 3:
                                panel_idx_1 = (1, chordwise_idx)
                                # if panel_idx_1 in blade_data:
                                # Update its front vertices to match the back vertices of the newly shed panel
                                self.wake_system[propeller_key][blade_key][panel_idx_1]['Vortex Rings']['Vertices'][0] = new_vertices[1]  # Connect to back left
                                self.wake_system[propeller_key][blade_key][panel_idx_1]['Vortex Rings']['Vertices'][3] = new_vertices[2]  # Connect to back right
                                
                                # Update its control point
                                vertices_1 = self.wake_system[propeller_key][blade_key][panel_idx_1]['Vortex Rings']['Vertices']
                                self.wake_system[propeller_key][blade_key][panel_idx_1]['Control Points'] = np.mean(vertices_1, axis=0)
                                
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
                    if cp_index[0] == fixed_radial_index:  # Only take points at the fixed radial position
                        gamma = blade_data['Gamma'][cp_index]
                        
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
        
        # Extract z-component (thrust) from Propeller_1 force
        thrust = forces_and_moments['Propeller_1']['force'][2]
        
        # Store time step and thrust value
        self.thrust_history.append({
            'time_step': time_step,
            'thrust': thrust
        })
    
    def plot_thrust_history(self, thrust_history):
        """
        Plot the thrust history over time steps.
        """
        if not thrust_history:
            print("No thrust history data provided.")
            return
        
        # Extract time steps and thrust values
        time_steps = [data['time_step'] for data in thrust_history]
        thrust_values = [data['thrust'] for data in thrust_history]
        
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
