function initial_var = get_initial_variables(my_path, my_iniFile)

% my_path = '/Users/manuel/Documents/MATLAB/DeleteMe/'; % path to initial variables
% my_iniFile = 'iniConfig.m'; % file containiing initial variables

    runtmp = fullfile(my_path, my_iniFile);
    run(runtmp) % load variables into workspace
    clear functions % clear the script from memory so further changes can be read 
    clear my_path my_iniFile runtmp

    all_myVars = who; % name af all variables in worksapce.
    for i=1:length(all_myVars)    
        myVar = all_myVars{i};
        initial_var.(myVar) = eval(myVar); % strcuture holding all variables name and value
    end
    disp(['Variables loaded = ', num2str(length(all_myVars) )])
    clearvars -except initial_var

    %disp('The following variables were read: ');
    %disp(initial_var)
    %disp('done!!!!!!')

end
