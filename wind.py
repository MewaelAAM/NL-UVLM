import numpy as np
from scipy.spatial import cKDTree

class WindField:
    def __init__(self, mesh_data):
        """
        Initialize wind field using direct mesh data with pre-computed data structures
        """
        self.mesh = mesh_data
        
        # Pre-compute and store all points and velocities
        self.points = mesh_data.points
        self.velocities = mesh_data.get_array('U')
        
        # Create KD-tree for fast nearest neighbor lookup
        self.kdtree = cKDTree(self.points)
    
    def get_wind_velocity(self, position):
        """
        Get wind velocity using pre-computed KD-tree for fast lookup
        """
        # Find nearest point using KD-tree
        distance, idx = self.kdtree.query(position)
        
        return self.velocities[idx]

    @staticmethod
    def update_wind_function(wind_field, com_position):
        """
        Creates a wind function that provides wind velocity relative to the COM position.
        """
        def wind_velocity(position):
            absolute_position = position + com_position
            
            return wind_field.get_wind_velocity(absolute_position)


        return wind_velocity