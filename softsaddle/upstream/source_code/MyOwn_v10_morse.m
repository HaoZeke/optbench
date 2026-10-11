function [x_min, x_for_plot, x_trials, energy, v_min, ...
    minimizer_func_calls, hess_func_calls, func_calls, run_count, LS_total_iter, CG_total_iter, x_0 ] =...
    MyOwn_v10_morse(x,n, N ,saddle_search, folder_index, r_c, run_count, project_name, my_paths, xmin, iniVar, current_fmin)
%% MyOwn script! Minimum Mode Following method for Nonlinear optimization
% Version 2.0 (Notice than main changes are in the child functions)
%           It is optimize and in 2d is able to plot the path on top of
%           landscape for some functions. This feature is removed in newer
%           version (higer than version 2.0)
% Version 3.0 (Notice than main changes are in the child functions)
%           Plotting landscape is not include. the code is more friendy to
%           deal with higer dimensions.
% Version 4.0 
%           It works loading shared libraies created from the c++ codes of
%           lanczos method and morse potential
% Version 5.0
%           Calls an external function (iTOUGH2)
% Version 6.0
%           Creates a subspace to deal with parameters at the boundary.
%           If by following the gradient, the boundary is crossed,  LS and
%           CG are reset and a new line search begings in a (n-1)
%           dimenisonal subsapce.
% Version 8.0
%           Write all data to shared memory and then move it to HDD at the
%           end of simulation.
% Version 9.0
%           tmp folder allows to execute multiple matlab sesions simultaniusly
%           without conflicting each other. (tmp folder is in shared memory and
%           uses and unique random ID) (NOTE: version 9.0 works with 
%           globalsadd_v78_morse and newer version)
% Version 10.0
%           Read library from iniconfig file.
%
% xmin -> receives the coordinates of the minimum, to be used in
% SP-searches to estimate the direction to go.


% ************* Begin Initialization *********
    plot_path = false; % Record solution path: true or false (If true two
                      % variables are created: plot_path and plot_trials 

    lanc_iter           = iniVar.lanc_iter;
    lanc_abs_tol        = iniVar.lanc_abs_tol;
    lanc_finite         = iniVar.lanc_finite;
    min_method          = iniVar.min_method; % Decide what method to use for the minimization:
    write_out_to_hdd    = iniVar.write_out_to_hdd; % write screen output to a file                   
    libreria            = iniVar.lib_name;

 if write_out_to_hdd==true 
        % folder estructure for minimization from SPs
        %my_path_to_minimum = [my_paths{2}, '/0_0_0/']; % path to load the minimum in CG_nr_ function 
        if r_c==21 % r_c takes alnly this values: 0, 1, 2 and 21 and 22

            local_project_name = [num2str(folder_index),'_1_' , num2str(run_count), '/'];

            my_paths{1} = [my_paths{1}, project_name, local_project_name, 'min1/']; % ??????
            %my_paths{2} = [my_paths{2}, num2str(run_count), 'min1/'];                     
            %my_paths{2} = [my_paths{2}, num2str(folder_index), '_1_' , num2str(run_count), '/min1/'];                     
            my_paths{2} = [my_paths{2},'/', local_project_name, '/min1/'];                     

        elseif r_c==22
            local_project_name = [num2str(folder_index), '_1_' , num2str(run_count), '/'];  

            my_paths{1} = [my_paths{1}, project_name, local_project_name, 'min2/']; % ??????
            %my_paths{2} = [my_paths{2}, num2str(run_count), 'min2/'];
            %my_paths{2} = [my_paths{2}, num2str(folder_index), '_1_' , num2str(run_count), '/min2/'];
            my_paths{2} = [my_paths{2},'/', local_project_name, '/min2/'];

        else

            % Modify the paths (for parallel runs)     
            my_paths{1} = [my_paths{1}, project_name, num2str(folder_index),...
                            '_', num2str(r_c), '_' , num2str(run_count), '/'];
            my_paths{2} = [my_paths{2}, '/',num2str(folder_index), '_' , ...
                                num2str(r_c), '_' , num2str(run_count), '/'];            

        end  
 else
        t = getCurrentTask();% t.ID has unique ID for each worker. If I run
                             % only one minimization (not in parallel)from initial guess then t is empty
        if isempty(t)
            % Modify the paths (for parallel runs)     
            my_paths{2} = [my_paths{2},'/0/'];     
        else    
            % Modify the paths (for parallel runs)     
            my_paths{2} = [my_paths{2},'/',  num2str(t.ID), '/'];     
        end     
 end
% ************* End Initialization ********* 

% ************* BEGIN create a temporal folder
    % create working and tmp folder
    err_code2 = system(['mkdir -p ', my_paths{2}]);
    if (err_code2 ~= 0) 
        disp('Warning! creating temporal/working folder..')
    end
    oldFolder = cd(my_paths{2});% changing to tmp work directory
% ************* END create a temporal folder

% ************* Begin call functions *********

% *** BEGING load shared library with lanczos and morse   
%     if ~libisloaded('liblanczos')   
%         loadlibrary('liblanczos.so', 'matlab.h')
%     end   

    if ~libisloaded(libreria)  
        %loadlibrary([iniVar.lib_name, '.so'], 'matlab.h')
        loadlibrary(libreria, 'matlab.h','alias','lib')
    end 
    
    
    if ~libisloaded('libmorse')   
        loadlibrary('libmorse.so', 'morse.h')
    end    
    % calls initialize function to initialize lanczos

    report = 0; % 0=> false, 1=> true
    % Simulation domain (box length in x y and z)
    box_l = [19.2088; 19.0118; 30]; % for non-cubic
    boxx= zeros(1,9);
    boxx(1)=box_l(1); boxx(5)=box_l(2);boxx(9)=box_l(3);
    %calllib('liblanczos', 'initialise', boxx, lanc_abs_tol, lanc_iter, lanc_finite, report, x, 3*N)
    calllib('lib', 'initialise', boxx, lanc_abs_tol, lanc_iter, lanc_finite, report, x, 3*N)
% *** END load shared library with lanczos and morse                
 
    if strcmp(min_method, 'LBFGS')

        % Minimization algorithm => LBFGS
        % Line search method => Newtom-Rhapson
        [x_min, x_for_plot, x_trials, energy, v_min,minimizer_func_calls...
        , hess_func_calls, func_calls, LS_total_iter, CG_total_iter, x_0 ] = LBFGS_morse_v6...
        (x, n, saddle_search, plot_path, boxx, my_paths, run_count, xmin, current_fmin, iniVar); 

    elseif strcmp(min_method, 'CG')
        % Minimization algorithm => conjugate gradient
        % Line search method => Newtom-Rhapson
        [x_min, x_for_plot, x_trials, energy, v_min,minimizer_func_calls...
        , hess_func_calls, func_calls, LS_total_iter, CG_total_iter, x_0 ] = CG_nr_v22_morse...
        (x, n, saddle_search, plot_path, boxx, my_paths, run_count, xmin, current_fmin, iniVar); 

    else % default method CG
        % Minimization algorithm => conjugate gradient
        % Line search method => Newtom-Rhapson
        [x_min, x_for_plot, x_trials, energy, v_min,minimizer_func_calls...
        , hess_func_calls, func_calls, LS_total_iter, CG_total_iter, x_0 ] = CG_nr_v22_morse...
        (x, n, saddle_search, plot_path, boxx, my_paths, run_count, xmin, current_fmin, iniVar); 
        
    end
    % If not convergence empty run_count variable
% 	if isempty(energy)
% 	 run_count = [];
%      % Do nothing... I need to read run_count
% 	end
% ************* End call functions *********
    unloadlibrary lib
    unloadlibrary libmorse

    cd(oldFolder);% changing back to matlab default directory

% ************* END delete a temporal folder


 
