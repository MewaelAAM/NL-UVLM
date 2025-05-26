'''
Enhanced plotting module for VLM Quadcopter Simulation
Provides advanced visualization of flight data with wind effects

Save this file as enhanced_plotting.py to enable the enhanced plotting features
'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.gridspec import GridSpec
from matplotlib import cm


def plot_enhanced_results(results):
    """
    Plot enhanced simulation results including wind effects
    
    Args:
        results: Dictionary containing simulation results
    """
    time_index = results['time']
    position = results['position']
    angle = results['angle']
    wind_velocity = results.get('wind_velocity', None)
    
    # Create figure for enhanced plots
    fig = plt.figure(figsize=(18, 10), dpi=80, facecolor='w', edgecolor='k')
    gs = GridSpec(3, 4, figure=fig)
    
    # 3D Flight path
    ax_3d = fig.add_subplot(gs[0:2, 0:2], projection='3d')
    ax_3d.plot(position[0], position[1], position[2], linewidth=2)
    ax_3d.set_title('Flight Path with Wind Vectors')
    ax_3d.set_xlabel('x (m)')
    ax_3d.set_ylabel('y (m)')
    ax_3d.set_zlabel('z (m)')
    
    # Add reference position
    if len(position[0]) > 0:
        ref_point = [position[0][0], position[1][0], position[2][0]]
        ax_3d.scatter(ref_point[0], ref_point[1], ref_point[2], color='r', marker='o', s=100, label='Starting Position')
    
    # Add wind vectors if available
    if wind_velocity is not None and len(wind_velocity) > 0:
        # Sample wind vectors at regular intervals
        sample_interval = max(1, len(time_index) // 10)
        for i in range(0, len(time_index), sample_interval):
            if i < len(wind_velocity):
                wind_vec = wind_velocity[i]
                scale = 5.0  # Scale factor for visualization
                if np.linalg.norm(wind_vec) > 0:
                    ax_3d.quiver(position[0][i], position[1][i], position[2][i],
                              wind_vec[0] * scale, wind_vec[1] * scale, wind_vec[2] * scale,
                              color='b', alpha=0.5)
    
    ax_3d.legend()
    
    # Altitude and hover error
    ax_alt = fig.add_subplot(gs[0, 2])
    ax_alt.plot(time_index, position[2], 'g-', linewidth=2)
    ax_alt.set_title('Altitude vs. Time')
    ax_alt.set_xlabel('Time (s)')
    ax_alt.set_ylabel('Altitude (m)')
    ax_alt.grid(True)
    
    # Add reference line for desired altitude
    if len(position[2]) > 0:
        ref_altitude = position[2][0]  # Assuming initial altitude is the reference
        ax_alt.axhline(y=ref_altitude, color='r', linestyle='--', label=f'Ref: {ref_altitude}m')
        ax_alt.legend()
    
    # 2D position trace (top-down view)
    ax_xy = fig.add_subplot(gs[0, 3])
    ax_xy.plot(position[0], position[1], 'b-', linewidth=2)
    ax_xy.set_title('Top-Down View (XY Plane)')
    ax_xy.set_xlabel('X Position (m)')
    ax_xy.set_ylabel('Y Position (m)')
    ax_xy.grid(True)
    
    # Add starting point marker
    if len(position[0]) > 0:
        ax_xy.plot(position[0][0], position[1][0], 'ro', markersize=8, label='Start')
        ax_xy.legend()
    
    # Euler angles
    ax_angles = fig.add_subplot(gs[1, 2])
    ax_angles.plot(time_index, angle[0], 'r-', linewidth=2, label='Roll')
    ax_angles.plot(time_index, angle[1], 'g-', linewidth=2, label='Pitch')
    ax_angles.plot(time_index, angle[2], 'b-', linewidth=2, label='Yaw')
    ax_angles.set_title('Euler Angles vs. Time')
    ax_angles.set_xlabel('Time (s)')
    ax_angles.set_ylabel('Angle (deg)')
    ax_angles.legend()
    ax_angles.grid(True)
    
    # Motor thrusts
    ax_thrust = fig.add_subplot(gs[1, 3])
    for i in range(4):
        ax_thrust.plot(time_index, results['motor_thrust'][i], linewidth=2, label=f'Motor {i+1}')
    ax_thrust.set_title('Motor Thrust vs. Time')
    ax_thrust.set_xlabel('Time (s)')
    ax_thrust.set_ylabel('Thrust (N)')
    ax_thrust.legend()
    ax_thrust.grid(True)
    
    # Total thrust
    ax_total_thrust = fig.add_subplot(gs[2, 0])
    ax_total_thrust.plot(time_index, results['total_thrust'], 'k-', linewidth=2)
    ax_total_thrust.set_title('Total Thrust vs. Time')
    ax_total_thrust.set_xlabel('Time (s)')
    ax_total_thrust.set_ylabel('Thrust (N)')
    ax_total_thrust.grid(True)
    
    # Position error
    ax_error = fig.add_subplot(gs[2, 1])
    ax_error.plot(time_index, results['total_error'], 'm-', linewidth=2)
    ax_error.set_title('Position Error vs. Time')
    ax_error.set_xlabel('Time (s)')
    ax_error.set_ylabel('Error (m)')
    ax_error.grid(True)
    
    # Wind speed
    if wind_velocity is not None and len(wind_velocity) > 0:
        ax_wind = fig.add_subplot(gs[2, 2])
        wind_magnitude = [np.linalg.norm(w) for w in wind_velocity]
        ax_wind.plot(time_index[:len(wind_magnitude)], wind_magnitude, 'c-', linewidth=2)
        ax_wind.set_title('Wind Magnitude vs. Time')
        ax_wind.set_xlabel('Time (s)')
        ax_wind.set_ylabel('Wind Speed (m/s)')
        ax_wind.grid(True)
    
    # Body torques
    ax_torque = fig.add_subplot(gs[2, 3])
    ax_torque.plot(time_index, results['body_torque'][0], 'r-', linewidth=2, label='Roll')
    ax_torque.plot(time_index, results['body_torque'][1], 'g-', linewidth=2, label='Pitch')
    ax_torque.plot(time_index, results['body_torque'][2], 'b-', linewidth=2, label='Yaw')
    ax_torque.set_title('Body Torques vs. Time')
    ax_torque.set_xlabel('Time (s)')
    ax_torque.set_ylabel('Torque (N·m)')
    ax_torque.legend()
    ax_torque.grid(True)
    
    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)
    plt.show()


def plot_wind_analysis(results):
    """
    Plot detailed wind analysis and quadcopter response
    
    Args:
        results: Dictionary containing simulation results
    """
    time_index = results['time']
    position = results['position']
    angle = results['angle']
    wind_velocity = results.get('wind_velocity', [])
    wind_files = results.get('wind_file', [])
    
    fig = plt.figure(figsize=(15, 10), dpi=80, facecolor='w', edgecolor='k')
    gs = GridSpec(2, 3, figure=fig)
    
    # Wind components over time
    ax_wind_comp = fig.add_subplot(gs[0, 0])
    if wind_velocity and len(wind_velocity) > 0:
        wx = [w[0] for w in wind_velocity]
        wy = [w[1] for w in wind_velocity]
        wz = [w[2] for w in wind_velocity]
        
        ax_wind_comp.plot(time_index[:len(wx)], wx, 'r-', linewidth=2, label='Wind X')
        ax_wind_comp.plot(time_index[:len(wy)], wy, 'g-', linewidth=2, label='Wind Y')
        ax_wind_comp.plot(time_index[:len(wz)], wz, 'b-', linewidth=2, label='Wind Z')
        
        # Add vertical lines to indicate wind file changes
        if wind_files and len(wind_files) > 0:
            current_file = wind_files[0]
            for i, file in enumerate(wind_files):
                if file != current_file:
                    ax_wind_comp.axvline(x=time_index[i], color='k', linestyle='--', alpha=0.5)
                    current_file = file
        
    ax_wind_comp.set_title('Wind Components vs. Time')
    ax_wind_comp.set_xlabel('Time (s)')
    ax_wind_comp.set_ylabel('Wind Velocity (m/s)')
    ax_wind_comp.legend()
    ax_wind_comp.grid(True)
    
    # Attitude response to wind
    ax_attitude = fig.add_subplot(gs[0, 1])
    ax_attitude.plot(time_index, angle[0], 'r-', linewidth=2, label='Roll')
    ax_attitude.plot(time_index, angle[1], 'g-', linewidth=2, label='Pitch')
    
    # Add wind magnitude on secondary y-axis if wind data available
    if wind_velocity and len(wind_velocity) > 0:
        ax_wind = ax_attitude.twinx()
        wind_magnitude = [np.linalg.norm(w) for w in wind_velocity]
        ax_wind.plot(time_index[:len(wind_magnitude)], wind_magnitude, 'b--', linewidth=1.5, label='Wind Magnitude')
        ax_wind.set_ylabel('Wind Magnitude (m/s)', color='b')
        ax_wind.tick_params(axis='y', labelcolor='b')
        
        # Add vertical lines to indicate wind file changes
        if wind_files and len(wind_files) > 0:
            current_file = wind_files[0]
            for i, file in enumerate(wind_files):
                if file != current_file:
                    ax_attitude.axvline(x=time_index[i], color='k', linestyle='--', alpha=0.5)
                    current_file = file
    
    ax_attitude.set_title('Attitude Response to Wind')
    ax_attitude.set_xlabel('Time (s)')
    ax_attitude.set_ylabel('Angle (deg)')
    ax_attitude.legend(loc='upper left')
    ax_attitude.grid(True)
    
    # Motor thrust response to wind
    ax_motor_resp = fig.add_subplot(gs[0, 2])
    for i in range(4):
        ax_motor_resp.plot(time_index, results['motor_thrust'][i], linewidth=2, label=f'Motor {i+1}')
    
    # Add wind magnitude on secondary y-axis if wind data available
    if wind_velocity and len(wind_velocity) > 0:
        ax_wind2 = ax_motor_resp.twinx()
        wind_magnitude = [np.linalg.norm(w) for w in wind_velocity]
        ax_wind2.plot(time_index[:len(wind_magnitude)], wind_magnitude, 'k--', linewidth=1.5, label='Wind Magnitude')
        ax_wind2.set_ylabel('Wind Magnitude (m/s)', color='k')
        ax_wind2.tick_params(axis='y', labelcolor='k')
        
        # Add vertical lines to indicate wind file changes
        if wind_files and len(wind_files) > 0:
            current_file = wind_files[0]
            for i, file in enumerate(wind_files):
                if file != current_file:
                    ax_motor_resp.axvline(x=time_index[i], color='k', linestyle='--', alpha=0.5)
                    current_file = file
    
    ax_motor_resp.set_title('Motor Thrust Response to Wind')
    ax_motor_resp.set_xlabel('Time (s)')
    ax_motor_resp.set_ylabel('Thrust (N)')
    ax_motor_resp.legend(loc='upper left')
    ax_motor_resp.grid(True)
    
    # 3D Path with wind vectors
    ax_3d = fig.add_subplot(gs[1, 0:2], projection='3d')
    
    # Color the path segments based on wind files
    if wind_files and len(wind_files) > 0:
        unique_files = []
        for file in wind_files:
            if file not in unique_files:
                unique_files.append(file)
        
        colors = plt.cm.jet(np.linspace(0, 1, len(unique_files)))
        color_map = {file: colors[i] for i, file in enumerate(unique_files)}
        
        current_file = wind_files[0]
        start_idx = 0
        
        for i, file in enumerate(wind_files):
            if file != current_file or i == len(wind_files) - 1:
                # Plot segment with specific color
                end_idx = i
                if i == len(wind_files) - 1:
                    end_idx = i + 1  # Include the last point
                
                ax_3d.plot(position[0][start_idx:end_idx], 
                         position[1][start_idx:end_idx], 
                         position[2][start_idx:end_idx], 
                         color=color_map[current_file], 
                         linewidth=2, 
                         label=f'Wind File: {current_file}')
                
                # Update for next segment
                current_file = file
                start_idx = i
    else:
        # Plot regular path if no wind files
        ax_3d.plot(position[0], position[1], position[2], 'k-', linewidth=2, label='Flight Path')
    
    # Add wind vectors along flight path if available
    if wind_velocity and len(wind_velocity) > 0:
        # Sample wind vectors at regular intervals
        sample_interval = max(1, len(time_index) // 15)
        for i in range(0, len(time_index), sample_interval):
            if i < len(wind_velocity):
                wind_vec = wind_velocity[i]
                scale = 5.0  # Scale factor for visualization
                if np.linalg.norm(wind_vec) > 0:
                    ax_3d.quiver(position[0][i], position[1][i], position[2][i],
                              wind_vec[0] * scale, wind_vec[1] * scale, wind_vec[2] * scale,
                              color='b', alpha=0.5)
    
    ax_3d.set_title('Flight Path with Wind Vectors')
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    
    # Create a custom legend for unique wind files
    if wind_files and len(wind_files) > 0:
        handles = []
        labels = []
        for file in unique_files:
            handles.append(plt.Line2D([0], [0], color=color_map[file], lw=2))
            labels.append(f'Wind: {file}')
        ax_3d.legend(handles, labels, loc='upper right')
    else:
        ax_3d.legend()
    
    # Hover performance with wind
    ax_hover = fig.add_subplot(gs[1, 2])
    altitude_error = [abs(position[2][i] - position[2][0]) for i in range(len(position[2]))]
    lateral_error = [np.sqrt((position[0][i] - position[0][0])**2 + (position[1][i] - position[1][0])**2) 
                    for i in range(len(position[0]))]
    
    ax_hover.plot(time_index, altitude_error, 'g-', linewidth=2, label='Altitude Error')
    ax_hover.plot(time_index, lateral_error, 'm-', linewidth=2, label='Lateral Error')
    
    # Add wind magnitude if available
    if wind_velocity and len(wind_velocity) > 0:
        ax_wind3 = ax_hover.twinx()
        wind_magnitude = [np.linalg.norm(w) for w in wind_velocity]
        ax_wind3.plot(time_index[:len(wind_magnitude)], wind_magnitude, 'b--', linewidth=1.5, label='Wind Magnitude')
        ax_wind3.set_ylabel('Wind Magnitude (m/s)', color='b')
        ax_wind3.tick_params(axis='y', labelcolor='b')
        
        # Add vertical lines to indicate wind file changes
        if wind_files and len(wind_files) > 0:
            current_file = wind_files[0]
            for i, file in enumerate(wind_files):
                if file != current_file:
                    ax_hover.axvline(x=time_index[i], color='k', linestyle='--', alpha=0.5)
                    current_file = file
    
    ax_hover.set_title('Hover Performance with Wind')
    ax_hover.set_xlabel('Time (s)')
    ax_hover.set_ylabel('Error (m)')
    ax_hover.legend(loc='upper left')
    ax_hover.grid(True)
    
    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)
    plt.show()
', linewidth=2, label='Wind X')
        ax_wind_comp.plot(time_index[:len(wy)], wy, 'g-', linewidth=2, label='Wind Y')
        ax_wind_comp.plot(time_index[:len(wz)], wz, 'b-', linewidth=2, label='Wind Z')
        
    ax_wind_comp.set_title('Wind Components vs. Time')
    ax_wind_comp.set_xlabel('Time (s)')
    ax_wind_comp.set_ylabel('Wind Velocity (m/s)')
    ax_wind_comp.legend()
    ax_wind_comp.grid(True)
    
    # Attitude response to wind
    ax_attitude = fig.add_subplot(gs[0, 1])
    ax_attitude.plot(time_index, angle[0], 'r-', linewidth=2, label='Roll')
    ax_attitude.plot(time_index, angle[1], 'g-', linewidth=2, label='Pitch')
    
    # Add wind magnitude on secondary y-axis if wind data available
    if wind_velocity and len(wind_velocity) > 0:
        ax_wind = ax_attitude.twinx()
        wind_magnitude = [np.linalg.norm(w) for w in wind_velocity]
        ax_wind.plot(time_index[:len(wind_magnitude)], wind_magnitude, 'b--', linewidth=1.5, label='Wind Magnitude')
        ax_wind.set_ylabel('Wind Magnitude (m/s)', color='b')
        ax_wind.tick_params(axis='y', labelcolor='b')
    
    ax_attitude.set_title('Attitude Response to Wind')
    ax_attitude.set_xlabel('Time (s)')
    ax_attitude.set_ylabel('Angle (deg)')
    ax_attitude.legend(loc='upper left')
    ax_attitude.grid(True)
    
    # Motor thrust response to wind
    ax_motor_resp = fig.add_subplot(gs[0, 2])
    for i in range(4):
        ax_motor_resp.plot(time_index, results['motor_thrust'][i], linewidth=2, label=f'Motor {i+1}')
    
    # Add wind magnitude on secondary y-axis if wind data available
    if wind_velocity and len(wind_velocity) > 0:
        ax_wind2 = ax_motor_resp.twinx()
        wind_magnitude = [np.linalg.norm(w) for w in wind_velocity]
        ax_wind2.plot(time_index[:len(wind_magnitude)], wind_magnitude, 'k--', linewidth=1.5, label='Wind Magnitude')
        ax_wind2.set_ylabel('Wind Magnitude (m/s)', color='k')
        ax_wind2.tick_params(axis='y', labelcolor='k')
    
    ax_motor_resp.set_title('Motor Thrust Response to Wind')
    ax_motor_resp.set_xlabel('Time (s)')
    ax_motor_resp.set_ylabel('Thrust (N)')
    ax_motor_resp.legend(loc='upper left')
    ax_motor_resp.grid(True)
    
    # 3D Path with wind vectors
    ax_3d = fig.add_subplot(gs[1, 0:2], projection='3d')
    ax_3d.plot(position[0], position[1], position[2], 'k-', linewidth=2, label='Flight Path')
    
    # Add wind vectors along flight path if available
    if wind_velocity and len(wind_velocity) > 0:
        # Sample wind vectors at regular intervals
        sample_interval = max(1, len(time_index) // 15)
        for i in range(0, len(time_index), sample_interval):
            if i < len(wind_velocity):
                wind_vec = wind_velocity[i]
                scale = 5.0  # Scale factor for visualization
                if np.linalg.norm(wind_vec) > 0:
                    ax_3d.quiver(position[0][i], position[1][i], position[2][i],
                              wind_vec[0] * scale, wind_vec[1] * scale, wind_vec[2] * scale,
                              color='b', alpha=0.5)
    
    ax_3d.set_title('Flight Path with Wind Vectors')
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.legend()
    
    # Hover performance with wind
    ax_hover = fig.add_subplot(gs[1, 2])
    altitude_error = [abs(position[2][i] - position[2][0]) for i in range(len(position[2]))]
    lateral_error = [np.sqrt((position[0][i] - position[0][0])**2 + (position[1][i] - position[1][0])**2) 
                    for i in range(len(position[0]))]
    
    ax_hover.plot(time_index, altitude_error, 'g-', linewidth=2, label='Altitude Error')
    ax_hover.plot(time_index, lateral_error, 'm-', linewidth=2, label='Lateral Error')
    
    # Add wind magnitude if available
    if wind_velocity and len(wind_velocity) > 0:
        ax_wind3 = ax_hover.twinx()
        wind_magnitude = [np.linalg.norm(w) for w in wind_velocity]
        ax_wind3.plot(time_index[:len(wind_magnitude)], wind_magnitude, 'b--', linewidth=1.5, label='Wind Magnitude')
        ax_wind3.set_ylabel('Wind Magnitude (m/s)', color='b')
        ax_wind3.tick_params(axis='y', labelcolor='b')
    
    ax_hover.set_title('Hover Performance with Wind')
    ax_hover.set_xlabel('Time (s)')
    ax_hover.set_ylabel('Error (m)')
    ax_hover.legend(loc='upper left')
    ax_hover.grid(True)
    
    plt.tight_layout(pad=1.0, w_pad=2.0, h_pad=2.0)
    plt.show()


def plot_azimuth_effects(results, time_step_index=None):
    """
    Plot effects of different propeller azimuth positions on forces and moments
    
    Args:
        results: Dictionary containing simulation results
        time_step_index: Index of time step to analyze (if None, uses middle of simulation)
    """
    if 'azimuth_data' not in results:
        print("No azimuth data available for analysis")
        return
    
    azimuth_data = results['azimuth_data']
    
    # If no specific time step is provided, use the middle of the simulation
    if time_step_index is None:
        time_step_index = len(results['time']) // 2
    
    # Get data for the selected time step
    if str(time_step_index) not in azimuth_data:
        print(f"No azimuth data available for time step {time_step_index}")
        return
    
    data = azimuth_data[str(time_step_index)]
    azimuth_angles = data['azimuth_angles']
    forces = data['forces']
    moments = data['moments']
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    
    # Forces vs azimuth
    ax1 = fig.add_subplot(221, projection='polar')
    ax1.plot(azimuth_angles, [f[0] for f in forces], 'r-', label='Force X')
    ax1.plot(azimuth_angles, [f[1] for f in forces], 'g-', label='Force Y')
    ax1.plot(azimuth_angles, [f[2] for f in forces], 'b-', label='Force Z')
    ax1.set_title('Forces vs Azimuth Angle')
    ax1.legend()
    
    # Moments vs azimuth
    ax2 = fig.add_subplot(222, projection='polar')
    ax2.plot(azimuth_angles, [m[0] for m in moments], 'r-', label='Moment X')
    ax2.plot(azimuth_angles, [m[1] for m in moments], 'g-', label='Moment Y')
    ax2.plot(azimuth_angles, [m[2] for m in moments], 'b-', label='Moment Z')
    ax2.set_title('Moments vs Azimuth Angle')
    ax2.legend()
    
    # Total force magnitude vs azimuth
    ax3 = fig.add_subplot(223, projection='polar')
    force_magnitudes = [np.linalg.norm(f) for f in forces]
    ax3.plot(azimuth_angles, force_magnitudes, 'k-')
    ax3.set_title('Force Magnitude vs Azimuth Angle')
    
    # Total moment magnitude vs azimuth
    ax4 = fig.add_subplot(224, projection='polar')
    moment_magnitudes = [np.linalg.norm(m) for m in moments]
    ax4.plot(azimuth_angles, moment_magnitudes, 'k-')
    ax4.set_title('Moment Magnitude vs Azimuth Angle')
    
    plt.tight_layout()
    plt.show()


def create_animation_data(results):
    """
    Prepare data for animation of the quadcopter simulation
    
    Args:
        results: Dictionary containing simulation results
    
    Returns:
        Dictionary with data prepared for animation
    """
    # Extract relevant data
    position = np.array([results['position'][0], results['position'][1], results['position'][2]]).T
    angles = np.array([results['angle'][0], results['angle'][1], results['angle'][2]]).T
    wind = np.array(results.get('wind_velocity', [[0, 0, 0]] * len(position)))
    
    # Create animation data dictionary
    animation_data = {
        'position': position,
        'angles': angles,
        'time': results['time'],
        'wind': wind
    }
    
    return animation_data