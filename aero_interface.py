import numpy as np
import sys
import time
import inspect
import importlib
import pyvista as pv

# Add timestamp to track reloading
MODULE_LOAD_TIME = time.time()
print(f"aero_interface.py loaded at: {time.ctime(MODULE_LOAD_TIME)}")
print(f"File location: {__file__}")

# Reset module state on reload
_is_initialized = False
aero_interface = None

# Add a function to explicitly reset the module state
def reset_module():
    """Reset the module state explicitly"""
    global _is_initialized, aero_interface
    print(f"Explicitly resetting aero_interface module state at {time.ctime()}")
    _is_initialized = False
    aero_interface = None
    return True

# Import your custom modules
try:
    # Force reload of dependent modules
    if 'vlm' in sys.modules:
        importlib.reload(sys.modules['vlm'])
    if 'propeller' in sys.modules:
        importlib.reload(sys.modules['propeller'])
    if 'mesh' in sys.modules:
        importlib.reload(sys.modules['mesh'])
    if 'wind' in sys.modules:
        importlib.reload(sys.modules['wind'])

    from propeller import PropellerGeometry
    from mesh import PropellerMesh
    from vlm import VLM
    from wind import WindField

    print("Successfully imported propeller, mesh, wind, and VLM modules")
except ImportError as e:
    print(f"Error importing modules: {e}")
    print(f"Python path: {sys.path}")


class AerodynamicsInterface:
    def __init__(self):
        print(f"Initializing Python Aerodynamics Interface at {time.ctime()}")
        
        # Initialize geometry and mesh only once
        self.propeller_geometry = PropellerGeometry(
            airfoil_distribution_file='DJI9443_airfoils.csv',
            chorddist_file='DJI9443_chorddist.csv',
            pitchdist_file='DJI9443_pitchdist.csv',
            sweepdist_file='DJI9443_sweepdist.csv',
            heightdist_file='DJI9443_heightdist.csv',
            R_tip=0.11938,
            R_hub=0.00624,
            num_blades=2
        )
        
        self.propeller_mesh_system = PropellerMesh(self.propeller_geometry, arm_length=0.175, com=(0, 0, 0))
        print('Propeller Loaded Successfully')
        
        self.quad_propeller_mesh = self.propeller_mesh_system.generate_quad_propeller_mesh()
        print('Mesh Generated Successfully')
        
        # Initialize the UVLM solver
        self.uvlm_solver = VLM(self.quad_propeller_mesh)
        print('UVLM Solver Generated Successfully')
        
        # Try to get the file location of the VLM class
        try:
            vlm_file = inspect.getfile(self.uvlm_solver.__class__)
            print(f"VLM class loaded from: {vlm_file}")
        except Exception as e:
            print(f"Could not get VLM file location: {e}")
        
        # Use simple numpy array for wind field
        mesh = pv.read(r"bp_50percent_75.vtk") 
        self.wind_field = WindField(mesh)
        print('Wind Field Initialized')
        
        # Constants
        self.rho = 1.071778
        self.n_steps_rev = 20
        self.dt = 2*np.pi / (565.9 * self.n_steps_rev)
        
        # Initialize motor speeds
        self.omega_dict = {
            'Propeller_1': np.array([0, 0, 565.9]),
            'Propeller_2': np.array([0, 0, -565.9]),
            'Propeller_3': np.array([0, 0, -565.9]),
            'Propeller_4': np.array([0, 0, 565.9])
        }
        
        self.time_step = 0
        print("Python Aerodynamics Interface ready!")

    def step(self, position, velocity, angles, angular_rates, motor_speeds):
        """Function called from Simulink at each time step"""
    
        self.time_step += 1
        
        # Update motor speeds
        motor_speeds_arr = np.array(motor_speeds).flatten()
        self.omega_dict = {
            'Propeller_1': np.array([0, 0, motor_speeds_arr[0]]),
            'Propeller_2': np.array([0, 0, -motor_speeds_arr[1]]),
            'Propeller_3': np.array([0, 0, -motor_speeds_arr[2]]),
            'Propeller_4': np.array([0, 0, motor_speeds_arr[3]])
        }
        
        # Convert inputs to numpy arrays
        position_arr = np.array(position).flatten()
        velocity_arr = np.array(velocity).flatten()
        angles_arr = np.array(angles).flatten()
        
        print(f"Position: {position_arr}, Velocity: {velocity_arr}, Angles: {angles_arr}")
        
        # Call the VLM solver
        result = self.uvlm_solver.calculate_total_forces_and_moments(
            propeller_mesh=self.quad_propeller_mesh,
            dt=self.dt,
            rho=self.rho,
            time_step=self.time_step,
            body_velocity=velocity_arr,
            omega=self.omega_dict,
            wind_field=self.wind_field,
            com_position=position_arr,
            roll=angles_arr[0],
            pitch=angles_arr[1],
            yaw=angles_arr[2]
        )
        
        print(f"Result type: {type(result)}")
        print(result)
        print('succesful')
        # It's a dictionary with propeller data
        prop1_force = result['Propeller_1']['force']
        prop1_moment = result['Propeller_1']['moment']
        prop2_force = result['Propeller_2']['force']
        prop2_moment = result['Propeller_2']['moment']
        prop3_force = result['Propeller_3']['force']
        prop3_moment = result['Propeller_3']['moment']
        prop4_force = result['Propeller_4']['force']
        prop4_moment = result['Propeller_4']['moment']
    
        
        # Combine all outputs
        all_outputs = np.concatenate([
            prop1_force, prop1_moment,
            prop2_force, prop2_moment,
            prop3_force, prop3_moment,
            prop4_force, prop4_moment
        ])
        
        print(f"Time step {self.time_step}: Returning forces and moments")
        return all_outputs.tolist()
     

# Create a single instance
aero_interface = AerodynamicsInterface()

def initialize():
    """Function to call from MATLAB to ensure initialization"""
    print("Python module loaded and initialized!")
    return True

def step(position, velocity, angles, angular_rates, motor_speeds):
    """Function called from MATLAB at each time step"""
    return aero_interface.step(position, velocity, angles, angular_rates, motor_speeds)