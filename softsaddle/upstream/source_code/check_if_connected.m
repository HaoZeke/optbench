function [ x_min1, x_min2, energy1, energy2, connected_sp,...
m_fc_m1, h_fc_m1, T_fc_m1, m_LS_total_iter1, m_CG_total_iter1, ...
m_fc_m2, h_fc_m2, T_fc_m2, m_LS_total_iter2, m_CG_total_iter2] =...
              check_if_connected...
( v_min_TMP, x_sp_f, delta, absTol, current_min, l, N,run_count_f, ...
                                            project_name, my_paths, iniVar)
%check_if_connected Minimizes from both sides of the SPs and determines
%whether or not the SP is connected to the minimum passed in 'current_min'
%   

i=0; %This variables only change in goust. In globalsad remains cts. 

[r, n_vectors] = size(v_min_TMP); % reads number of vectors passed and number of componnents.

n_var = 3*N;% total number of variables due to all atoms in the system
if r<n_var % if vectors has less componnents that number of variables
    
    % fill with zeros the v_min components for frozen atoms   
    bottom_v_min = zeros(n_var-r, n_vectors); % matrix of zeros to be appendex to the bottom of v_min array
    % NOTE: the result matrix will be [v_min_TMP; bottom_v_min]
    v_min_TMP_ = [v_min_TMP; bottom_v_min];
    %disp(['Size of v_min_TMP_ = ', num2str(size(v_min_TMP_))])
else
    
    v_min_TMP_ = v_min_TMP;
    
end

% Initialization;
%disp(['Already inside check_if_connected.m, n_vectors =  ', num2str(n_vectors)])

if n_vectors == 0
    % Only validate the output variables:
    x_min1 = []; x_min2 = []; energy1 = []; energy2 = []; connected_sp = [];...
    m_fc_m1= 0; h_fc_m1 = 0; T_fc_m1= 0; m_LS_total_iter1= 0; m_CG_total_iter1= 0; ...
    m_fc_m2= 0; h_fc_m2= 0; T_fc_m2= 0; m_LS_total_iter2= 0; m_CG_total_iter2= 0;    
else
    % 3) Minimizations from displacements along min mode:            
    % ************************************ 
    % Minimize from all unique SPs    
    parfor k = 1:n_vectors
    
        % displacement along min mode
        [dis1_sp, dis2_sp] = min_mod_dis_generator...
        (v_min_TMP_(:,k), x_sp_f(:,k), delta);  

        % Minimization No 1
        [x_min1(:,k), ~, ~, energ1, ~, m_fc_m1(k), h_fc_m1(k), T_fc_m1(k), ~, m_LS_total_iter1(k), m_CG_total_iter1(k), ~] = MyOwn_v10_morse...
          (dis1_sp, l, N, false,i , 21, run_count_f(k), project_name, my_paths, [], iniVar, []);
        if isempty( energ1 )
            energ1 = Inf; % means minimization did not converged!
        end
        energy1(k) = energ1;
%disp(['done x_min1 in folder ->', num2str(run_count_f(k)) ])

        % Minimization No 2
        [x_min2(:,k), ~, ~, energ2, ~, m_fc_m2(k), h_fc_m2(k), T_fc_m2(k), ~, m_LS_total_iter2(k), m_CG_total_iter2(k), ~] = MyOwn_v10_morse...
          (dis2_sp, l, N, false,i , 22, run_count_f(k), project_name, my_paths, [], iniVar, []);
        if isempty( energ2 )
            energ2 = Inf; % means minimization did not converged!
        end
        energy2(k) = energ2;
%disp(['done x_min2 in folder ->', num2str(run_count_f(k)) ])
        %Check wheteher the SPs are connected to the initial statate or not.
%         connected_sp(k) = any([...
%            is_visited_morse(current_min, x_min1(:,k), absTol)...
%            is_visited_morse(current_min, x_min2(:,k), absTol)]);
        m1 = is_visited_morse(current_min, x_min1(:,k), absTol);
        m2 = is_visited_morse(current_min, x_min2(:,k), absTol);

        connected_sp(k) = any([m1 m2]);

    end
end
end

