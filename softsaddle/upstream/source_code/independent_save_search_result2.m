function independent_save_search_result2(working_path, n_searches, n_free_a)
% Save a report about the SP searches in a txt file 
%
   % n_seraches = 1500
   % n_free_a = 175 -> (56_atoms *3_layers + 7_atoms)*3_components = 525 variables
% Particular cases:
    % load the minimum:
    min0 = load(fullfile(working_path, '0_0_0/obj.con'));
% memory allocation:
    run_count = (1:n_searches)'; % index of searches (column vector)
    f_v_list = zeros(size(run_count)); % SP - energies
    sp_fcs = f_v_list; % sp func calls
    min_fcs = f_v_list; % min func calls
    comp = 3*n_free_a; % number of atom componnents
    v_list = zeros(comp, n_searches); % matrix which colums represent each atom position
    %f_v_list(:) = NaN; % bad searches identify by NaN.
% Initialization:
j= 1; % counter for index of real entries in energy array
tic;

% Validate variables:
    for i=1:n_searches
       path_to_read = fullfile(working_path, ['0_1_', num2str(i)] ); 
       %disp(i) % error tracking
       
      % SP - position and energy
       v_list_tmp = load(fullfile(path_to_read, 'sp.con')); % position
       if isempty(v_list_tmp)
           
            v_list(:,i) = NaN; % position
            
            % mark it as a bad search 'NaN'
            f_v_list(i) = NaN; % energy
            %disp(['Warning! Empty energy at ', num2str(i)])
           
       else
           
            v_list(:,i) = v_list_tmp(1:comp); % takes only position free atoms
            
            % SP - energy 
            %f_v_list_tmp = load([path_to_read, 'obj.con']); % energy
            % save the energy value
            f_v_list(i) = load(fullfile(path_to_read, 'obj.con')); % energy
            i_n(j) = i; % index to acces real values from the array
            j = j + 1;
           
       end
              
      % SP - function calls
       sp_fcs(i) = load(fullfile(path_to_read, 'fcs.con')); % function calls
       
    end
    f_v_list = f_v_list - min0; % energy barrier
    
    % save the array of function calls for getting standar deviation
    save([working_path, 'sp_fcs_array.mat'], 'sp_fcs');
    
% 1.    
disp('SPs energies, fcs and indexs data loaded successfuly! - 1/5' )

% Validate variables for determining connection
    load([working_path, 'connected.con']); % load variable connected -> 3 columns matrix (run_count| connected or not | energy)
    i_unique = connected(:,1)';
    i_to_connected = i_unique(logical(connected(:,2))); % index of connected SPs
    
% 2. 
disp('SPs connection data loaded successfuly! - 2/5' )

% Validate variables for determining min_fcs
    for i=i_unique
        path_to_read1 = [working_path, '0_1_', num2str(i), '/min1/' ]; 
        path_to_read2 = [working_path, '0_1_', num2str(i), '/min2/' ];

        % min - function calls
        min1 = load([path_to_read1, 'fcs.con']); % func calc minimization No1 
        min2 = load([path_to_read2, 'fcs.con']); % func calc minimization No2
        min_fcs(i) = min1 + min2;    
    end
 
%3.        
disp('minimization fcs data loaded successfuly! - 3/5' )    

% Determnine unique vectors
    absTol = 0.1;

    [ind_ex, ~, ~, ~, corrected_run_count, same ]=...
                                    are_vectors_equal_morse_v6...
    (v_list(:,i_n), f_v_list((i_n)), absTol, v_list(:,i_n), run_count(i_n));

    ind_ex = int32(ind_ex);% convert from double to integer to save memory 
    
%4.
disp('Data processed successfuly! - 4/5' ) 

% Save a report
    save_search_resultV5...
     (working_path, n_searches, run_count, f_v_list, sp_fcs, min_fcs, ind_ex, same, corrected_run_count, i_to_connected, i_n)

% 5.
disp('Report created and saved successfuly! - 5/5' ) 
disp('All done!!')

toc; 
 

end
        
