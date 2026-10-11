function save_search_resultV5...
  (working_path, n_vectors, run_count, f_v_list, sp_fcs, min_fcs, ind_ex, same, corrected_run_count, i_to_connected, i_n)

%% function 'save_search_result.m' saves to HDD information about SP 
% seraches including which SP is unique and wich is repeated.
% Input:
%   working_path = path to save file 'mysearch_result.txt'
%   n_vectors -> number of vectors in the list
%   run_count -> index for every SP searches
%   f_v_list  -> energy barrier
%   sp_fcs    -> function evaluation to reach the SP
%   min_fcs   -> function evaluation in the minimization(if any) to 
%                determine whether the SP is unique or not.
%   From function 'are_vectors_equal_morse_v3':
%                                 ind_ex, same,  corrected_run_count
% 
% Output:
%   file 'mysearch_result.txt' is saved on HDD
       
%% initialization:
    Myfile = [working_path, 'ind_batch_results.mat'];
    resul = cell(n_vectors,1); % array of strings to store batch

%% Validation:
    batch_results = [run_count f_v_list double(sp_fcs)]; % Record all searches
        
      % get index NaN from corrected_run_count
        i_nan = ~isnan(corrected_run_count);% logic index of real entries
        i_d = find(i_nan); % index of real entries in 'corrected_run_count' 
        
      % Find repeated SPs
        repeated_par = ind_ex(same, :); 
        [~, iii, ~] = unique(repeated_par(:,2), 'first');
        par = repeated_par(iii,:);% array of SP-search index(colum No.2 repets colum No1)      
        
      % register unique SPs in 'resul' 
        for i=1:sum(i_nan)
            resul{corrected_run_count(i_d(i))} = ['Unique-', num2str( i )];
            %resul{i_d(i)} = ['Unique-', num2str( i )];           
        end 
        

        run_count_ = run_count(i_n); % index of good searches
      % register repeated SPs in 'resul' 
        for i=1:length(par(:,1))
            resul{run_count_( par(i,2) )} = ['repeat-',num2str(run_count_( par(i,1) )) ];
        end 
       
       % register connected SPs in 'resul' : 
        %con_ind = corrected_run_count(i_to_connected); % index of connected SPs
        for i=i_to_connected
            resul{i} = [resul{i}, '-Connected'];
            %resul_old{~con_ind} = [resul_old{~con_ind}, '-NotConnected'];
        end
        
%% Save results:

        save(Myfile, 'batch_results' )
        save([working_path, 'resul.mat'], 'resul' ) 
                               
        % Variables for statistical analisys:
          % SPs repetition
            sp_repetition = zeros(n_vectors,2); %
            sp_repetition(:,1) = NaN; % bad searches marked with NaN
            sp_repetition(corrected_run_count(i_d),1) = corrected_run_count(i_d); % uniques
            sp_repetition(run_count_( par(:,2) ),2 ) = run_count_( par(:,1) ); % column No.2 repeats colum No1
            sp_repetition(run_count_( par(:,2) ),1 ) = run_count_( par(:,2) ); % column No.2 repeats colum No1
          
%             sp_repetition = zeros(n_vectors,1); %
%             sp_repetition(:) = NaN; % bad searches marked with NaN
%             %a = sp_repetition(:,1);
%             sp_repetition(corrected_run_count(i_d)) = 1:sum(i_nan); % Unique
%             sp_repetition(run_count_( par(:,2) )) = run_count_( par(:,1) ); % repeated
            sp_repetition(isnan(sp_repetition(:,1)), :) = []; % remove NaN entries
            
            
            %sp_repetition = [run_count_( par(:,1) )  run_count_( par(:,2) )]; % colum No.2 repeats colum No1
            save([working_path, 'sp_repetition.mat'], 'sp_repetition' ) % to make histogram

          % searches vs (finding & repetation)
            t = 1:sum(i_nan);
            sp_finding = zeros(length(batch_results(:,1)), 4); % memory allocation
            sp_finding(:,1) = batch_results(:,1); % gets all seach indexes
            sp_finding(corrected_run_count(i_d(t)),2) = t; % record unique findings
            sp_finding(:,4) = sp_finding(:,2); % consecutive index of unique SPs
            sp_finding(corrected_run_count(i_d(t)),3) = 1; % 1 means new finding, zero menas no new finding to plot search vs. new finding
            sp_finding_tmp = sp_finding(:,3);
            for k=1:n_vectors
              sp_finding_tmp(k) =  sum( sp_finding( (1:k),3 ) ); 
            end
            sp_finding(:,3) = sp_finding_tmp;
            sp_finding(run_count_( par(:,2) ),2) = sp_finding(run_count_( par(:,1)),2); % record repeated findings
            save([working_path, 'sp_finding.mat'], 'sp_finding' ) % to make search vs finding new        
        
       % Write a text file report:

        fileID = fopen([working_path, 'mysearch_result.txt'],'w');

        fprintf(fileID,'%-5s\t %-8s\t %-8s\t %-8s\t %-8s\n', 'index', 'barrier', 'sp_fcs', 'min_fcs', 'results');
        fprintf(fileID,'\n');

        % Write to file
        for i=1:n_vectors

            fprintf(fileID,'%4u\t %9.4f\t %9.1f\t %9.1f\t %-9s\n',...
             batch_results(i,1), batch_results(i,2), batch_results(i,3), min_fcs(i), resul{i});

        end        

        fclose(fileID); % close the file tha keeps screen output        
        
%% group all initial displacement that found same SP
        
        rep_array = [run_count_( par(:,2) ) run_count_( par(:,1) )]; %  colum No.2 repeats colum No1
        
        for i=1:length(par(:,1))
            
          load_from_here = [working_path, '0_1_', num2str(rep_array(i,1)), '/disp.con'];
          copy_to_here = [working_path, '0_1_', num2str(rep_array(i,2)), '/disp_', num2str(rep_array(i,1)), '.con' ];
          system(['cp ', load_from_here, ' ', copy_to_here]);          
          
        end
        

        
end
        