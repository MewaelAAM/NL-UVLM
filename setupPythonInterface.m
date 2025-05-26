function setupPythonInterface()
    % Add the folder with your Python script to the Python path
    if count(py.sys.path, pwd) == 0
        insert(py.sys.path, int32(0), pwd);
    end
    
    % Check if Python module is accessible
    try
        py.importlib.import_module('aero_interface');
        disp('Successfully imported Python aerodynamics interface module');
    catch e
        error('Error loading Python module: %s', e.message);
    end
    
    % Initialize the Python interface
    try
        result = py.aero_interface.initialize();
        if result
            disp('Python aerodynamics interface initialized successfully');
        else
            error('Failed to initialize Python aerodynamics interface');
        end
    catch e
        error('Error initializing Python interface: %s', e.message);
    end
end