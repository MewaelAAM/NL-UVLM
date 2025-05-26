import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.animation import FuncAnimation, PillowWriter



class SixDOFDynamics:
    def __init__(self, mass, inertia_matrix, initial_position=None, initial_velocity_body=None,
                 initial_angles=None, initial_angular_rates=None):
        # Initialize history tracking
        self.history = {
            'time': [],
            'position': [],
            'velocity_body': [],
            'angles': [],
            'angular_rates': []
        }
        """
        Initialize 6-DOF rigid body dynamics with body-frame velocities.
        """
        self.mass = mass
        self.inertia_matrix = inertia_matrix
        self.inertia_inverse = np.linalg.inv(inertia_matrix)
        
        # Initialize states
        self.position = initial_position if initial_position is not None else np.zeros(3)
        self.velocity_body = initial_velocity_body if initial_velocity_body is not None else np.zeros(3)
        self.angles = initial_angles if initial_angles is not None else np.zeros(3)
        self.angular_rates = initial_angular_rates if initial_angular_rates is not None else np.zeros(3)
        
        # Store previous states
        self.previous_position = self.position.copy()
        self.previous_velocity_body = self.velocity_body.copy()
        self.previous_angles = self.angles.copy()
        self.previous_angular_rates = self.angular_rates.copy()

    def get_body_to_inertial_matrix(self):
        """
        Get transformation matrix from body to inertial frame using Euler angles.
        """
        phi, theta, psi = self.angles  # roll (ϕ), pitch (θ), yaw (ψ)
        
        return np.array([
            [np.cos(theta)*np.cos(psi), 
             np.cos(psi)*np.sin(theta)*np.sin(phi) - np.sin(psi)*np.cos(phi),
             np.cos(psi)*np.sin(theta)*np.cos(phi) + np.sin(psi)*np.sin(phi)],
            [np.sin(psi)*np.cos(theta),
             np.sin(psi)*np.sin(theta)*np.sin(phi) + np.cos(psi)*np.cos(phi),
             np.sin(psi)*np.sin(theta)*np.cos(phi) - np.cos(psi)*np.sin(phi)],
            [-np.sin(theta),
             np.cos(theta)*np.sin(phi),
             np.cos(theta)*np.cos(phi)]
        ])

    def update_states(self, forces_body, moments_body, dt, current_time, gravity=np.array([0, 0, -9.81], dtype=np.float64)):
        """
        Update all states using current forces and moments.
        
        Args:
            forces_body (np.ndarray): Forces in body frame [Fu, Fv, Fw]
            moments_body (np.ndarray): Moments in body frame [L, M, N]
            dt (float): Time step
            current_time (float): Current simulation time
            gravity (np.ndarray): Gravity vector in inertial frame
        """
        # Ensure all inputs are float64 to avoid type casting issues
        forces_body = np.array(forces_body, dtype=np.float64)
        moments_body = np.array(moments_body, dtype=np.float64)
        
        # Store previous states
        self.previous_position = self.position.copy()
        self.previous_velocity_body = self.velocity_body.copy()
        self.previous_angles = self.angles.copy()
        self.previous_angular_rates = self.angular_rates.copy()

        # Extract current states
        u, v, w = self.velocity_body
        p, q, r = self.angular_rates
        phi, theta, psi = self.angles  # roll (ϕ), pitch (θ), yaw (ψ)

        # 1. Update body-frame velocities (u, v, w)
        # Calculate gravity components in body frame
        g_body = np.array([
            -np.sin(theta),
            np.sin(phi)*np.cos(theta),
            np.cos(phi)*np.cos(theta)
        ], dtype=np.float64) * gravity[2]  # Only z component as gravity is [0, 0, -g]

        # Velocity update in body frame
        velocity_dot = np.array([
            r*v - q*w,           # ů
            -r*u + p*w,         # v̇
            q*u - p*v           # ẇ
        ], dtype=np.float64) + g_body + forces_body/self.mass

        # Update body-frame velocities using Euler integration
        self.velocity_body = self.velocity_body + velocity_dot * dt

        # 2. Transform body velocities to inertial frame
        R_body_to_inertial = self.get_body_to_inertial_matrix()
        position_dot = R_body_to_inertial @ self.velocity_body

        # Update position in inertial frame
        self.position = self.position + position_dot * dt
        
        # Store states in history
        self.history['time'].append(current_time)
        self.history['position'].append(self.position.copy())
        self.history['velocity_body'].append(self.velocity_body.copy())
        self.history['angles'].append(self.angles.copy())
        self.history['angular_rates'].append(self.angular_rates.copy())

        # 3. Update angular motion
        # Calculate angular acceleration
        angular_momentum = self.inertia_matrix @ self.angular_rates
        gyroscopic_moment = -np.cross(self.angular_rates, angular_momentum)
        angular_acceleration = self.inertia_inverse @ (moments_body + gyroscopic_moment)
        
        # Update angular rates
        self.angular_rates = self.angular_rates + angular_acceleration * dt
        
        # Update Euler angles
        phi, theta = self.angles[0:2]  # Current roll and pitch
        
        # Check for gimbal lock
        if abs(np.cos(theta)) < 1e-6:
            # Near gimbal lock, use simplified update
            euler_rates = np.array([
                self.angular_rates[0],
                0,  # Lock pitch rate
                self.angular_rates[2]  # Allow yaw to continue
            ], dtype=np.float64)
            print(f"Warning: Near gimbal lock at t={current_time:.2f}s, theta={np.rad2deg(theta):.1f}°")
        else:
            # Normal case
            euler_rates_matrix = np.array([
                [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
                [0, np.cos(phi), -np.sin(phi)],
                [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
            ], dtype=np.float64)
            euler_rates = euler_rates_matrix @ self.angular_rates
        
        # Update Euler angles
        self.angles = self.angles + euler_rates * dt

    def get_state_dict(self):
        """Return current states as a dictionary."""
        return {
            'position': self.position.copy(),
            'velocity_body': self.velocity_body.copy(),
            'angles': self.angles.copy(),
            'angular_rates': self.angular_rates.copy()
        }

    def update_mesh_transformations(propeller_mesh, dynamics, dt, omega_dict):
        """
        Update mesh geometry based on both body motion and propeller rotation.

        """
        def rotate_vector(vector, rotation_matrix):
            """Apply rotation matrix to a vector."""
            return rotation_matrix @ vector

        def rotate_around_point(point, center, rotation_matrix):
            """Rotate a point around a center point using rotation matrix."""
            relative_pos = point - center
            rotated_relative = rotation_matrix @ relative_pos
            return center + rotated_relative

        # Get body rotation matrix from dynamics
        body_rotation_matrix = dynamics.get_body_to_inertial_matrix()
        
        body_displacement = dynamics.position - dynamics.previous_position
        
        for propeller_key, propeller_data in propeller_mesh.items():
            # Get propeller's hub position relative to COM
            hub_position = np.array(propeller_data['Hub Position'])
            
            # Get propeller's rotation matrix for this time step
            omega = omega_dict[propeller_key]
            angle = np.linalg.norm(omega) * dt
            if angle > 0:
                axis = omega / np.linalg.norm(omega)  # Normalize to get the rotation axis
                c, s = np.cos(angle), np.sin(angle)
                v = 1 - c
                prop_rotation_matrix = np.array([
                    [axis[0] * axis[0] * v + c, axis[0] * axis[1] * v - axis[2] * s, axis[0] * axis[2] * v + axis[1] * s],
                    [axis[1] * axis[0] * v + axis[2] * s, axis[1] * axis[1] * v + c, axis[1] * axis[2] * v - axis[0] * s],
                    [axis[2] * axis[0] * v - axis[1] * s, axis[2] * axis[1] * v + axis[0] * s, axis[2] * axis[2] * v + c]
                ])
            else:
                prop_rotation_matrix = np.eye(3)  # No rotation if angle is zero

            # Update hub position with body motion
            new_hub_position = rotate_vector(hub_position, body_rotation_matrix) + body_displacement
            propeller_data['Hub Position'] = new_hub_position.tolist()

            # Update each blade
            for blade_key, blade_data in propeller_data['Blades'].items():
                # 1. Update panel vertices
                for panel_index, panel in blade_data['Panels'].items():
                    panel_array = np.array(panel)
                    transformed_panel = np.zeros_like(panel_array)
                    
                    for i, vertex in enumerate(panel_array):
                        # First rotate around hub due to propeller rotation
                        rotated_vertex = rotate_around_point(vertex, hub_position, prop_rotation_matrix)
                   
                        transformed_vertex = rotate_vector(rotated_vertex, body_rotation_matrix) + body_displacement
                        transformed_panel[i] = transformed_vertex
                        
                    blade_data['Panels'][panel_index] = transformed_panel.tolist()


                # 2. Update control points
                for cp_index, control_point in blade_data['Control Points'].items():
                    cp_array = np.array(control_point)
                    # Propeller rotation
                    
                    rotated_cp = rotate_around_point(cp_array, hub_position, prop_rotation_matrix)
                    # Body motion
                    transformed_cp = rotate_vector(rotated_cp, body_rotation_matrix) + body_displacement
                    blade_data['Control Points'][cp_index] = transformed_cp
                   

                # 3. Update vortex ring vertices
                for vr_index, vortex_ring in blade_data['Vortex Rings'].items():
                    vertices = np.array(vortex_ring['Vertices'])
                    transformed_vertices = np.zeros_like(vertices)
                    
                    for i, vertex in enumerate(vertices):
                        
                        rotated_vertex = rotate_around_point(vertex, hub_position, prop_rotation_matrix)
                
                        transformed_vertex = rotate_vector(rotated_vertex, body_rotation_matrix) + body_displacement
                        transformed_vertices[i] = transformed_vertex
                    
                    blade_data['Vortex Rings'][vr_index]['Vertices'] = transformed_vertices.tolist()

                # 4. Update normal vectors (only rotation, no translation)
                for normal_index, normal in blade_data['Normals'].items():
                    normal_array = np.array(normal)
                    # print('normal', normal_array)
                    # Rotate with propeller
                    rotated_normal = prop_rotation_matrix @ normal_array
                    # Rotate with body
                    transformed_normal = body_rotation_matrix @ rotated_normal
                    blade_data['Normals'][normal_index] = transformed_normal

                # 5. Update tangential vectors
                for tangential_index, tangential_vectors in blade_data['Tangential Vectors'].items():
                    # Handle both tangential vectors
                    for key in ['Tangential i', 'Tangential j']:
                        tangent = np.array(tangential_vectors[key])
                        # Rotate with propeller
                        rotated_tangent = prop_rotation_matrix @ tangent
                        # Rotate with body
                        transformed_tangent = body_rotation_matrix @ rotated_tangent
                        tangential_vectors[key] = transformed_tangent
        
    def plot_trajectories(dynamics):
        """
        Plot various trajectories and states from the dynamics history.
        
        Args:
            dynamics: SixDOFDynamics instance with history data
        """

        
        # Convert history lists to numpy arrays for easier plotting
        time = np.array(dynamics.history['time'])
        positions = np.array(dynamics.history['position'])
        angles = np.array(dynamics.history['angles'])
        
        # Create subplots
        fig = plt.figure(figsize=(15, 10))
        
        # 1. 3D Trajectory
        ax1 = fig.add_subplot(231, projection='3d')
        ax1.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', label='Path')
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title('3D Trajectory')
        ax1.grid(True)
        
        # 2. Lateral Trajectory (X-Y)
        ax2 = fig.add_subplot(232)
        ax2.plot(positions[:, 0], positions[:, 1], 'g-')
        ax2.set_xlabel('X (m)')
        ax2.set_ylabel('Y (m)')
        ax2.set_title('Lateral Trajectory')
        ax2.grid(True)
        ax2.axis('equal')
        
        # 3. Longitudinal Trajectory (X-Z)
        ax3 = fig.add_subplot(233)
        ax3.plot(positions[:, 0], positions[:, 2], 'r-')
        ax3.set_xlabel('X (m)')
        ax3.set_ylabel('Z (m)')
        ax3.set_title('Longitudinal Trajectory')
        ax3.grid(True)
        
        # 4. Altitude vs Time
        ax4 = fig.add_subplot(234)
        ax4.plot(time, positions[:, 2], 'b-')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Altitude (m)')
        ax4.set_title('Altitude vs Time')
        ax4.grid(True)
        
        # 5. Euler Angles vs Time
        ax5 = fig.add_subplot(235)
        ax5.plot(time, np.rad2deg(angles[:, 0]), 'r-', label='Roll')
        ax5.plot(time, np.rad2deg(angles[:, 1]), 'g-', label='Pitch')
        ax5.plot(time, np.rad2deg(angles[:, 2]), 'b-', label='Yaw')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Angle (deg)')
        ax5.set_title('Euler Angles vs Time')
        ax5.grid(True)
        ax5.legend()
        
        # 6. Position Components vs Time
        ax6 = fig.add_subplot(236)
        ax6.plot(time, positions[:, 0], 'r-', label='X')
        ax6.plot(time, positions[:, 1], 'g-', label='Y')
        ax6.plot(time, positions[:, 2], 'b-', label='Z')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Position (m)')
        ax6.set_title('Position Components vs Time')
        ax6.grid(True)
        ax6.legend()
        
        plt.tight_layout()
        plt.show()