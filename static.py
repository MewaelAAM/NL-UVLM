import numpy as np

class QuadcopterForceMoments:
    def __init__(self, arm_length, com_position):
        """
        Initialize the QuadcopterForceMoments calculator.
        
        Args:
            arm_length (float): Length of the quadcopter arms from center to propeller
            com_position (np.ndarray): Position of the center of mass (x, y, z)
        """
        self.arm_length = arm_length
        self.com_position = np.array(com_position)
        
        # Define propeller positions relative to COM
        # Using standard X configuration
        self.propeller_positions = {
            'Propeller_1': np.array([arm_length/np.sqrt(2), arm_length/np.sqrt(2), 0]),  # Front right
            'Propeller_2': np.array([arm_length/np.sqrt(2), -arm_length/np.sqrt(2), 0]), # Front left
            'Propeller_3': np.array([-arm_length/np.sqrt(2), arm_length/np.sqrt(2), 0]),# Rear left
            'Propeller_4': np.array([-arm_length/np.sqrt(2), -arm_length/np.sqrt(2), 0])  # Rear right
        }

    def calculate_total_force_and_moment(self, forces_dict):
        """
        Calculate total force and moment from all propellers.
        
        Args:
            forces_dict (dict): Dictionary containing forces from each propeller
                              Format: {'Propeller_1': {'force': np.array([fx, fy, fz]), 
                                                     'moment': np.array([mx, my, mz])},
                                     'Propeller_2': {...}, ...}
        
        Returns:
            tuple: (total_force, total_moment) where each is a numpy array [x, y, z]
        """
        total_force = np.zeros(3)
        total_moment = np.zeros(3)
        
        for propeller_key, force_data in forces_dict.items():
            # Add to total force
            propeller_force = force_data['force']
            total_force += propeller_force
            
            # Calculate moment from force
            moment_arm = self.propeller_positions[propeller_key]
            force_moment = np.cross(moment_arm, propeller_force)
            
            # Add propeller's own moment if provided
            if 'moment' in force_data:
                force_moment += force_data['moment']
            
            total_moment += force_moment
            
        return total_force, total_moment

    def get_propeller_contribution_analysis(self, forces_dict):
        """
        Analyze individual propeller contributions to total force and moment.
        
        Args:
            forces_dict (dict): Dictionary containing forces from each propeller
        
        Returns:
            dict: Analysis of each propeller's contribution
        """
        analysis = {}
        total_force, total_moment = self.calculate_total_force_and_moment(forces_dict)
        
        for propeller_key, force_data in forces_dict.items():
            propeller_force = force_data['force']
            moment_arm = self.propeller_positions[propeller_key]
            force_moment = np.cross(moment_arm, propeller_force)
            
            if 'moment' in force_data:
                force_moment += force_data['moment']
            
            # Calculate contribution percentages
            force_magnitude = np.linalg.norm(propeller_force)
            moment_magnitude = np.linalg.norm(force_moment)
            total_force_magnitude = np.linalg.norm(total_force)
            total_moment_magnitude = np.linalg.norm(total_moment)
            
            analysis[propeller_key] = {
                'force_contribution': propeller_force,
                'moment_contribution': force_moment,
                'force_percentage': (force_magnitude / total_force_magnitude * 100 
                                   if total_force_magnitude > 0 else 0),
                'moment_percentage': (moment_magnitude / total_moment_magnitude * 100 
                                    if total_moment_magnitude > 0 else 0)
            }
            
        return analysis

    def plot_force_moment_distribution(self, forces_dict):
        """
        Create visualization of force and moment distribution.
        
        Args:
            forces_dict (dict): Dictionary containing forces from each propeller
        """
        import matplotlib.pyplot as plt
        
        # Create figure with 3D subplot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot COM
        ax.scatter([0], [0], [0], color='red', s=100, label='COM')
        
        # Plot propellers and their force vectors
        colors = {'Propeller_1': 'blue', 'Propeller_2': 'green', 
                 'Propeller_3': 'orange', 'Propeller_4': 'purple'}
        
        for propeller_key, force_data in forces_dict.items():
            pos = self.propeller_positions[propeller_key]
            force = force_data['force']
            scale = 0.01  # Scale factor for force vectors visualization
            
            # Plot propeller position
            ax.scatter(pos[0], pos[1], pos[2], 
                      color=colors[propeller_key], s=50, label=propeller_key)
            
            # Plot force vector
            ax.quiver(pos[0], pos[1], pos[2],
                     force[0]*scale, force[1]*scale, force[2]*scale,
                     color=colors[propeller_key], alpha=0.6)
            
            # Plot moment arm
            ax.plot([0, pos[0]], [0, pos[1]], [0, pos[2]], 
                   '--', color=colors[propeller_key], alpha=0.3)
        
        # Set equal aspect ratio and labels
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Quadcopter Force and Moment Distribution')
        
        # Add legend
        ax.legend()
        
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