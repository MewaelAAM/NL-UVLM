% testPythonInterface.m - Test script for Python aerodynamics interface
clear py.aero_interface;  % Clear any previous instances
clear py.sys.modules; 
clear all;
clc;
% Ensure Python path includes our module directory


py.importlib.reload(py.importlib.import_module('aero_interface'));
disp('Python module reloaded successfully');
result = py.aero_interface.initialize();


% Test with sample data
position = double([0; 0; 0]);
velocity = double([0; 0; 0]);
angles = double([0; 0; 0]);
angular_rates = double([0; 0; 0]);
motor_speeds = double([565.9; 565.9; 565.9; 565.9]);
disp('Sending test data to Python...');

outputsPy = py.aero_interface.step(position, velocity, angles, angular_rates, motor_speeds);

% Convert Python list to MATLAB array
outputs = double(py.array.array('d', outputsPy));

% Extract individual propeller forces and moments
prop1_force = outputs(1:3);
prop1_moment = outputs(4:6);
prop2_force = outputs(7:9);
prop2_moment = outputs(10:12);
prop3_force = outputs(13:15);
prop3_moment = outputs(16:18);
prop4_force = outputs(19:21);
prop4_moment = outputs(22:24);

% Display results
disp('Test completed successfully!');
disp('Propeller 1 Force:');
disp(prop1_force);
disp('Propeller 1 Moment:');
disp(prop1_moment);
