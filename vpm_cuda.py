import numpy as np
import cupy as cp
import numba as nb
from numba import cuda, float64, int32
import math

class VPM:
    """
    Implementation of the Lagrangian Vortex Particle Method (VPM) for wake modeling.
    """
    def __init__(self):
        self.vortex_particles = {}
    
    def initialize_vpm_system(self):
        """Initialize wake system with specific control point indexing pattern."""
        self.vortex_particles = {}
                
    def create_vortex_particles_from_line(self, propeller_key, blade_key, start_point, end_point, gamma, direction, time_step, n_particles=1):
        """
        Create vortex particles from a vortex line segment.
        
        Args:
            propeller_key: Key of the propeller
            blade_key: Key of the blade
            start_point: Starting point of the vortex line
            end_point: Ending point of the vortex line
            gamma: Circulation strength
            direction: Direction vector of the vortex line
            time_step: Current time step
            n_particles: Number of particles to create along the line
        """
        # Initialize nested dictionary structure if it doesn't exist
        if propeller_key not in self.vortex_particles:
            self.vortex_particles[propeller_key] = {}
        if blade_key not in self.vortex_particles[propeller_key]:
            self.vortex_particles[propeller_key][blade_key] = []
            
        # Calculate line length
        line_length = np.linalg.norm(end_point - start_point)
        
        if line_length < 1e-10:
            return  # Skip if line is too short
        
        # Normalize direction vector
        if np.linalg.norm(direction) > 1e-10:
            direction = direction / np.linalg.norm(direction)
        else:
            direction = (end_point - start_point) / line_length
        
        # Particle strength formula: Γp = ΔΓ·dl/n
        particle_strength = gamma * line_length / n_particles
        
        # Create particles along the line
        for i in range(n_particles):
            # Position along the line (evenly distributed)
            alpha = (i + 0.5) / n_particles
            position = start_point + alpha * (end_point - start_point)
            
            # Create particle
            particle = {
                'position': position,
                'strength': particle_strength,
                'direction': direction,
                'time_created': time_step,
                'age': 0
            }
            
            # Add to the particle list
            self.vortex_particles[propeller_key][blade_key].append(particle)

    
    def calculate_particle_to_bound_induced_velocity_matrix(self, quad_propeller_mesh):
        """
        Calculate induced velocities from vortex particles onto bound control points.
        Uses the same Biot-Savart approach as the original UVLM code.
        
        Args:
            quad_propeller_mesh: Dictionary containing the propeller mesh data
            
        Returns:
            Dictionary of induced velocity matrices, keyed by propeller_key
        """
        global_matrices = {}
        
        # For each propeller
        for propeller_key, propeller_data in quad_propeller_mesh.items():
            # Collect all control points for this propeller
            control_points = []
            
            for blade_data in propeller_data['Blades'].values():
                for cp_index, control_point in blade_data['Control Points'].items():
                    control_points.append(control_point)
            
            # Convert to numpy array
            control_points = np.array(control_points, dtype=np.float64)
            
            # Initialize result matrix for this propeller
            n_points = len(control_points)
            result_matrix = np.zeros((n_points, 3), dtype=np.float64)
            
            # Check if we have particles for this propeller
            if propeller_key not in self.vortex_particles:
                global_matrices[propeller_key] = result_matrix
                continue
            
            # Collect all particles for this propeller
            particles_positions = []
            particles_strengths = []
            particles_directions = []
            
            for blade_key in self.vortex_particles[propeller_key]:
                for particle in self.vortex_particles[propeller_key][blade_key]:
                    particles_positions.append(particle['position'])
                    particles_strengths.append(particle['strength'])
                    particles_directions.append(particle['direction'])
            
            # If no particles, return zeros
            if not particles_positions:
                global_matrices[propeller_key] = result_matrix
                continue
            
            # Convert to numpy arrays
            particles_positions = np.array(particles_positions, dtype=np.float64)
            particles_strengths = np.array(particles_strengths, dtype=np.float64)
            particles_directions = np.array(particles_directions, dtype=np.float64)
            
            # Calculate GPU grid dimensions
            threads_per_block = 256
            blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block
            
            # Transfer data to GPU
            d_control_points = cuda.to_device(control_points)
            d_particles_positions = cuda.to_device(particles_positions)
            d_particles_strengths = cuda.to_device(particles_strengths)
            d_particles_directions = cuda.to_device(particles_directions)
            d_result_matrix = cuda.to_device(result_matrix)
            
            # Launch kernel
            particle_to_bound_kernel[blocks_per_grid, threads_per_block](
                d_control_points, 
                d_particles_positions, 
                d_particles_strengths, 
                d_particles_directions,
                d_result_matrix
            )
            
            # Get result back
            result_matrix = d_result_matrix.copy_to_host()
            global_matrices[propeller_key] = result_matrix
        
        return global_matrices
    
    def plot_vortex_particles(self, propeller_key=None, age_filter=None, azimuth=0, elevation=30, save_plot=False):
        """
        Plot vortex particles in 3D with color mapping based on strength.
        
        Args:
            propeller_key: Optional key to filter for a specific propeller
            age_filter: Optional maximum age to display (in time steps)
            azimuth: View azimuth angle (degrees)
            elevation: View elevation angle (degrees)
            save_plot: Whether to save the plot to a file
        """
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import numpy as np
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Define colormap for strength
        cmap = plt.cm.viridis
        
        # Collect all particle strengths to normalize color mapping
        all_strengths = []
        max_age = 0
        
        # Filter propellers if specified
        propeller_keys = [propeller_key] if propeller_key else self.vortex_particles.keys()
        
        # First pass to collect statistics
        for prop_key in propeller_keys:
            if prop_key not in self.vortex_particles:
                continue
                
            for blade_key, particles in self.vortex_particles[prop_key].items():
                for particle in particles:
                    # Apply age filter if specified
                    if age_filter is not None and particle['age'] > age_filter:
                        continue
                        
                    all_strengths.append(abs(particle['strength']))
                    max_age = max(max_age, particle['age'])
        
        # Avoid division by zero
        max_strength = max(all_strengths) if all_strengths else 1.0
        
        # Plot each particle
        for prop_key in propeller_keys:
            if prop_key not in self.vortex_particles:
                continue
                
            for blade_key, particles in self.vortex_particles[prop_key].items():
                # Initialize arrays for vectorized plotting
                positions = []
                colors = []
                sizes = []
                
                for particle in particles:
                    # Apply age filter if specified
                    if age_filter is not None and particle['age'] > age_filter:
                        continue
                    
                    # Get particle position
                    pos = particle['position']
                    strength = abs(particle['strength'])
                    age = particle['age']
                    
                    # Store position
                    positions.append(pos)
                    
                    # Determine color based on strength
                    normalized_strength = strength / max_strength
                    colors.append(cmap(normalized_strength))
                    
                    # Determine size based on strength (adjust scale as needed)
                    # Also make particles fade out with age
                    base_size = 30 * normalized_strength
                    age_factor = 1.0 - (age / max_age if max_age > 0 else 0)
                    sizes.append(base_size * max(0.2, age_factor))
                
                # Convert to numpy arrays for scatter plot
                if positions:
                    positions = np.array(positions)
                    
                    # Scatter plot
                    ax.scatter(
                        positions[:, 0], positions[:, 1], positions[:, 2],
                        c=colors, s=sizes, alpha=0.6, edgecolors='none'
                    )
                    
                    # Plot direction vectors for selected particles (every nth particle)
                    n = 10  # Plot every 10th particle's direction
                    for i in range(0, len(positions), n):
                        if i < len(particles):
                            particle = particles[i]
                            pos = particle['position']
                            direction = particle['direction']
                            strength = abs(particle['strength'])
                            
                            # Scale direction vector based on strength
                            scale = 0.01 * strength / max_strength
                            end_point = pos + direction * scale
                            
                            # Plot direction vector as a line
                            ax.plot(
                                [pos[0], end_point[0]],
                                [pos[1], end_point[1]],
                                [pos[2], end_point[2]],
                                'r-', linewidth=1, alpha=0.5
                            )
        
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, max_strength))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, label='Vortex Strength')
        
        # Set axis labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        title = 'Vortex Particle Visualization'
        if propeller_key:
            title += f' - {propeller_key}'
        if age_filter is not None:
            title += f' (Age <= {age_filter})'
        ax.set_title(title)
        
        # Set view angle
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
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.2), 
                markersize=10, label='Weak Vortex'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=cmap(0.8), 
                markersize=10, label='Strong Vortex'),
            Line2D([0], [0], color='r', lw=1, label='Direction Vector')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        # Save if requested
        if save_plot:
            filename = 'vortex_particles'
            if propeller_key:
                filename += f'_{propeller_key}'
            if age_filter is not None:
                filename += f'_age{age_filter}'
            plt.savefig(f'{filename}.png', dpi=300, bbox_inches='tight')
        
        plt.tight_layout()
        plt.show()
        
        return fig
    

    def plot_wake_spanwise_gamma(self, uvlm_solver, propeller_mesh, propeller_key="Propeller_1", blade_key="Blade_1", save_plot=False):
        """
        Plot the spanwise gamma distribution in the wake, showing how circulation
        varies from hub to tip with higher values expected at the tip.
        
        Args:
            uvlm_solver: The UVLM solver instance containing wake information
            propeller_mesh: Dictionary containing propeller mesh data
            propeller_key: Key of the propeller to visualize
            blade_key: Key of the blade to visualize
            save_plot: Whether to save the plot to a file
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from mpl_toolkits.mplot3d import Axes3D
        
        # Create figure with 2 subplots: 3D visualization and 2D spanwise plot
        fig = plt.figure(figsize=(15, 8))
        
        # 3D plot for spatial visualization
        ax1 = fig.add_subplot(121, projection='3d')
        
        # 2D plot for spanwise gamma distribution
        ax2 = fig.add_subplot(122)
        
        # Get hub position for reference
        hub_position = np.array(propeller_mesh[propeller_key]['Hub Position'])
        
        # Extract wake panels and their gamma values
        wake_data = uvlm_solver.wake_system[propeller_key][blade_key]
        
        # Group wake panels by chordwise index (we want to see variation along span)
        chordwise_indices = set()
        for index in wake_data.keys():
            chordwise_indices.add(index[1])
        
        # List to store radial position and gamma value for each panel
        radial_positions = []
        gamma_values = []
        wake_panels_xyz = []
        wake_panels_gamma = []
        
        # Find the tip radius for normalization
        tip_radius = 0
        for panel_index, panel in wake_data.items():
            vertices = panel['Vortex Rings']['Vertices']
            for vertex in vertices:
                # Calculate radial position from hub
                rel_pos = vertex - hub_position
                radius = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
                tip_radius = max(tip_radius, radius)
        
        # Process each wake panel
        for panel_index, panel in wake_data.items():
            if panel['Gamma'] is None:
                continue
                
            vertices = panel['Vortex Rings']['Vertices']
            gamma = panel['Gamma']
            
            # Calculate average position of wake panel
            center = np.mean(vertices, axis=0)
            
            # Calculate radial position from hub
            rel_pos = center - hub_position
            radius = np.sqrt(rel_pos[0]**2 + rel_pos[1]**2)
            
            # Normalize radius by tip radius
            r_R = radius / tip_radius if tip_radius > 0 else 0
            
            # Store data for plotting
            radial_positions.append(r_R)
            gamma_values.append(gamma)
            
            # Store panel for 3D visualization
            wake_panels_xyz.append(vertices)
            wake_panels_gamma.append(gamma)
        
        # 3D visualization of wake panels colored by gamma
        cmap = plt.cm.viridis
        
        # Find min/max gamma for color normalization
        all_gammas = np.array(gamma_values)
        vmin = np.min(all_gammas) if len(all_gammas) > 0 else 0
        vmax = np.max(all_gammas) if len(all_gammas) > 0 else 1
        
        # Normalize the gamma values for color mapping
        norm_gammas = (all_gammas - vmin) / (vmax - vmin) if vmax > vmin else all_gammas
        
        # Plot wake panels in 3D
        for i, vertices in enumerate(wake_panels_xyz):
            # Plot each edge of the panel
            for j in range(4):
                start = vertices[j]
                end = vertices[(j + 1) % 4]
                
                # Color based on gamma
                color = cmap(norm_gammas[i] if i < len(norm_gammas) else 0)
                
                ax1.plot([start[0], end[0]], 
                        [start[1], end[1]], 
                        [start[2], end[2]], 
                        color=color, alpha=0.7, linewidth=1)
        
        # Add hub position for reference
        ax1.scatter(hub_position[0], hub_position[1], hub_position[2], 
                color='red', s=50, marker='o', label='Hub')
        
        # Customize 3D plot
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')
        ax1.set_title(f'Wake Panels - {propeller_key} {blade_key}')
        ax1.view_init(elev=30, azim=45)
        
        # Set equal aspect ratio for 3D plot
        max_range = np.array([
            ax1.get_xlim3d()[1] - ax1.get_xlim3d()[0],
            ax1.get_ylim3d()[1] - ax1.get_ylim3d()[0],
            ax1.get_zlim3d()[1] - ax1.get_zlim3d()[0]
        ]).max() / 2.0
        
        mid_x = (ax1.get_xlim3d()[1] + ax1.get_xlim3d()[0]) / 2.0
        mid_y = (ax1.get_ylim3d()[1] + ax1.get_ylim3d()[0]) / 2.0
        mid_z = (ax1.get_zlim3d()[1] + ax1.get_zlim3d()[0]) / 2.0
        
        ax1.set_xlim3d([mid_x - max_range, mid_x + max_range])
        ax1.set_ylim3d([mid_y - max_range, mid_y + max_range])
        ax1.set_zlim3d([mid_z - max_range, mid_z + max_range])
        
        # Plot 2D spanwise gamma distribution
        radial_positions = np.array(radial_positions)
        gamma_values = np.array(gamma_values)
        
        # Sort by radial position for plotting
        sort_idx = np.argsort(radial_positions)
        radial_positions = radial_positions[sort_idx]
        gamma_values = gamma_values[sort_idx]
        
        # Plot the spanwise gamma distribution
        ax2.plot(radial_positions, gamma_values, 'bo-', linewidth=2, markersize=6)
        
        # Add trend line (polynomial fit)
        if len(radial_positions) > 2:
            try:
                z = np.polyfit(radial_positions, gamma_values, 3)
                p = np.poly1d(z)
                x_smooth = np.linspace(min(radial_positions), max(radial_positions), 100)
                ax2.plot(x_smooth, p(x_smooth), 'r--', linewidth=1.5, 
                    label='Trend (Polynomial Fit)')
            except:
                pass  # Skip trend line if fitting fails
        
        # Customize 2D plot
        ax2.set_xlabel('Normalized Radial Position (r/R)')
        ax2.set_ylabel('Gamma (Circulation)')
        ax2.set_title(f'Spanwise Gamma Distribution - {propeller_key} {blade_key}')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Add horizontal line at gamma = 0 for reference
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # Set y-axis limits to emphasize tip values
        y_range = max(abs(np.max(gamma_values)), abs(np.min(gamma_values))) if len(gamma_values) > 0 else 1
        ax2.set_ylim([-y_range * 0.1, y_range * 1.1])
        
        if len(radial_positions) > 0:
            ax2.set_xlim([0, 1.05])  # Normalized radial position should be 0 to 1
        
        # Add legend
        ax2.legend()
        
        # Add colorbar for gamma values
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin, vmax))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax1, label='Gamma (Circulation)')
        
        plt.tight_layout()
        
        # Save plot if requested
        if save_plot:
            plt.savefig(f'wake_spanwise_gamma_{propeller_key}_{blade_key}.png', 
                    dpi=300, bbox_inches='tight')
        
        plt.show()
        return fig