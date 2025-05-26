function [prop1_force, prop1_moment, prop2_force, prop2_moment, prop3_force, prop3_moment, prop4_force, prop4_moment] = callPythonAerodynamics(position, velocity, angles, angular_rates, motor_speeds)
    % Add coder.extrinsic declarations to bypass code generation for Python functions
    coder.extrinsic('py.aero_interface.step', 'py.aero_interface.initialize');
    coder.extrinsic('py.array.array', 'py.importlib.import_module');
    coder.extrinsic('py.sys.path', 'cell');
    
    % Initialize outputs to zeros in case of failure
    prop1_force = zeros(3,1);
    prop1_moment = zeros(3,1);
    prop2_force = zeros(3,1);
    prop2_moment = zeros(3,1);
    prop3_force = zeros(3,1);
    prop3_moment = zeros(3,1);
    prop4_force = zeros(3,1);
    prop4_moment = zeros(3,1);
    
    % Persistent flag for initialization
    persistent is_init;
    if isempty(is_init)
        is_init = false;
    end
    
    % Only initialize if needed
    if ~is_init
        % Add path to Python modules
        if coder.target('MATLAB')
            % Only execute in MATLAB, not in generated code
            if count(py.sys.path, pwd) == 0
                insert(py.sys.path, int32(0), pwd);
            end
            py.importlib.import_module('aero_interface');
            py.aero_interface.initialize();
            is_init = true;
        end
    end
    
    % Convert inputs to proper format for Python
    position_flat = double(position(:)');      % ensure row vector
    velocity_flat = double(velocity(:)');
    angles_flat = double(angles(:)');
    angular_rates_flat = double(angular_rates(:)');
    motor_speeds_flat = double(motor_speeds(:)');
    
    % Call Python only in MATLAB environment
    if coder.target('MATLAB')
        % Call the Python function
        outputsPy = py.aero_interface.step(position_flat, velocity_flat, angles_flat, angular_rates_flat, motor_speeds_flat);
        
        % Convert Python output to MATLAB array
        outputs = double(py.array.array('d', outputsPy));
        
        % Only process if we got enough outputs
        if length(outputs) == 24
            prop1_force = reshape(outputs(1:3), [3,1]);
            prop1_moment = reshape(outputs(4:6), [3,1]);
            prop2_force = reshape(outputs(7:9), [3,1]);
            prop2_moment = reshape(outputs(10:12), [3,1]);
            prop3_force = reshape(outputs(13:15), [3,1]);
            prop3_moment = reshape(outputs(16:18), [3,1]);
            prop4_force = reshape(outputs(19:21), [3,1]);
            prop4_moment = reshape(outputs(22:24), [3,1]);
        end
    end
end