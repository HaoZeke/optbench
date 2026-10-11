function [corrected_v_list, corrected_f_v_list, corrected_v_min, corrected_run_count, corrected_x_0]= ...
                    are_vectors_equal_morse_v6...
                (v_list, f_v_list, absTol, v_min, run_count, x_0)
            
% ************************************************************************ 
% *  GlobalSadd                                                          *
% *  Minimum mode following method for Saddle point searches.            *
% *    are_vectors_equal_goust -> look for similarities in a group of    *
% *    vectors. Sort the output according to the energy value            *
% *                                                                      *
% *      Author: Manuel Plasencia Gutierrez, 2013                        *
% *      email: mpg2@hi.is                                               *
% ************************************************************************
            
% Attempt to save memory managment
% update 15-march-2015:
%  check vectors by their enrgies only first and analize position of only
%  those having same energy. 
%  calls external function to make pairswise comparisson. (attempting to 
%  save memory)
% check parameters pased as function arguments
% if nargin<5 % means minima list is passed, v_min not passed

absTol_E = 0.01; % same value used in eon.(http://theory.cm.utexas.edu/eon/structure_comparison.html#structure-comparison)
  
    c = length(f_v_list); % get array dimensions c = number of min/sp
    myIndex = false(1,c);
    
    if (c >= 2) % 
        
        % compares all Energies first:
        % a)
        %[ ind_ex, same ] = compare_pairwise( f_v_list, absTol );
        %myIndex( ind_ex(same, :) ) = true;
        [ ind_ex2 ] = compare_pairwise_v2( f_v_list, absTol_E );
        myIndex( ind_ex2 ) = true;
        
        c_equal = sum(myIndex); % number of vectors with equal E.
                
        if (c_equal==0) % No vectors with same E so they are all different
            % No analisys is needed:
            %disp('All E are different:')
            corrected_v_list    = v_list;
            corrected_f_v_list  = f_v_list;
            corrected_v_min     = v_min;
            corrected_run_count = run_count;
            corrected_x_0       = x_0;
            
        else
            %disp(['number of equal E found =  ', num2str(c_equal)])
            equals_f_v_list = f_v_list( myIndex );
            equals_v_list = v_list( :, myIndex); 
            equals_v_min = v_min( :, myIndex );
            equals_run_count = run_count( myIndex );
            equals_x_0 = x_0( :, myIndex );
            
            NOT_equals_f_v_list = f_v_list( ~myIndex );
            NOT_equals_v_list = v_list( :, ~myIndex);
            NOT_equals_v_min = v_min( :, ~myIndex);
            NOT_equals_run_count = run_count( ~myIndex );
            NOT_equals_x_0 = x_0( :, ~myIndex);
            
            % b) update same and ind_ex:
            %[ ind_ex, same ] = compare_pairwise( equals_v_list, absTol );
            [ ind_ex2 ] = compare_pairwise_v2( equals_v_list', absTol );

            % remove from list repeated vectors:
            %equals_v_list(:, ind_ex(same, 2)) = NaN; % vectors
            %equals_f_v_list(ind_ex(same, 2)) = NaN; % E
            %equals_v_min(:, ind_ex(same, 2)) = NaN; % min mode
            %equals_run_count(ind_ex(same, 2)) = NaN; % sp-index
            equals_v_list(:, ind_ex2(:, 2)) = NaN; % vectors
            equals_f_v_list(ind_ex2(:, 2)) = NaN; % E
            equals_v_min(:, ind_ex2(:, 2)) = NaN; % min mode
            equals_run_count(ind_ex2(:, 2)) = NaN; % sp-index
            equals_x_0(:, ind_ex2(:, 2)) = NaN; % lamda zero
            
            
            % Merge non-equal list with the processed equal list:
            corrected_v_list = [equals_v_list NOT_equals_v_list];
            corrected_v_min = [equals_v_min NOT_equals_v_min];            
            corrected_f_v_list = [equals_f_v_list; NOT_equals_f_v_list];
            corrected_x_0 = [equals_x_0 NOT_equals_x_0];
                        
            if isrow(run_count)
                corrected_run_count = [equals_run_count NOT_equals_run_count];
            else
                corrected_run_count = [equals_run_count; NOT_equals_run_count];
            end

        end
 
        % ******* Sort the list ascending ****
            [corrected_f_v_list, in_dex] = sort(corrected_f_v_list);
            corrected_v_list = corrected_v_list(:,in_dex);
            corrected_v_min = corrected_v_min(:,in_dex);
            corrected_run_count = corrected_run_count(in_dex);
            corrected_x_0 = corrected_x_0(:,in_dex);
            
    else % (c<2) =>Only one vector in the comparison list

        %disp('Warning! Only one vector in the SP list to compare ')
        % validate the function and exit
        corrected_v_list = v_list;
        corrected_f_v_list = f_v_list;
        corrected_v_min = v_min;
        corrected_run_count = run_count;
        corrected_x_0 = x_0;
        
    end

   



