function [M, S, sp_fcs] = get_standard(working_path, n_searches)

% Read SPs function call and save them into a colum array for further
% processing like standar deviation
% M -> average  S -> standar deviation
%
tic;
% memory allocation:
    sp_fcs = zeros(1,n_searches); % sp func calls

% Validate variables:
    for i=1:n_searches
       path_to_read = [working_path, '0_1_', num2str(i), '/' ]; 
       % SP-function calls
       sp_fcs(i) = load([path_to_read, 'fcs.con']); % function calls
    end
    
    % save the array of function calls for getting standar deviation
    save([working_path, 'sp_fcs_array.mat'], 'sp_fcs');
   
    disp('SPs funcion calls array saved successfuly! (sp_fcs_array.mat)' )
   
    % Calculus:
    M = mean(sp_fcs); % mean or average value
    S = std(sp_fcs); % Standar deviation
        
    fprintf('%-6s\t', 'min_f','average_f', 'max_f', 'std');
    fprintf('\n');
    fprintf('%3.2f\t',[min(sp_fcs) M max(sp_fcs) S]);
    fprintf('\n');     
    disp('******=*****')
    
%     disp(['min SP_fcs = ', num2str(min(sp_fcs)) ]); % minimum
%     
%     disp(['average SP_fcs = ', num2str(M) ]);
%     
%     disp(['max SP_fcs = ', num2str(max(sp_fcs)) ]);
%     
%     disp(['standard deviation of SP_fcs = ', num2str(S) ]);
    
    disp('All done!')
    
toc;    
    
end

    