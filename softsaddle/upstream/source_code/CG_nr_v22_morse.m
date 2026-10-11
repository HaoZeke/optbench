	function [x_min, x_for_plot, x_trials, final_energy, v_min,...
    minimizer_func_calls, hess_func_calls, func_calls, LS_total_iter, i, x_0 ] = CG_nr_v22_morse(x, n,...
    saddle_search, plot_path, box, my_paths, labindex, xmin, current_min, iniVar)

%%   Version x.x   Newton-Raphson for the line search
%
% What is new in version 8:
%   . max_step_size can be use to abandon negative curvature area (on/off)
%   . Abandon SP search if the function is larger than a given value.
% What is new in version 9:
%   . when max_step_size is used no trial step is given to increase
%   efficiency
%   . bug related to aborting SP serach due to hight energy corrected
% What is new in version 11:
%   . Writes all files in memory during execution. Move to HDD at the end. 
% What is new in version 13:
%   . efficient algorithm to remember stept lenght to determine whether to
%   use old minmod or recalculated. 
% What is new in version 14:
%   . New resetting strategy for not avandon basin of atraction.
%   . Uses same formula once the positive area is exited once.
% What is new in version 15:
%   . dont calculate min mode in + area
% What is new in version 17:
%   . save lamda zero points for better exploring the surface of the
%   function.
% What is new in version 18:
%   . Restart a SP search that had converged to a minimum.
% What is new in version 19:
%   . Several approach to calculate beta for Conjugate Gradient method
% What is new in version 20:
%   . Modified formaula for determining the search direction in the CG method 
% What is new in version 21:
%   . It can be choosen whether or not calculate minimum mode during the LS
%   . Add LS stopping criterium: dot(g,d)<cg_err
%   . fixed bug related to LS print out
% What is new in version 22:
%   . All ouput and saving to HDD can be turn off.
%   . The minimum 'E' is passed in the function argument rather than its
%   position.
% I have removed 'lam' from the input argument of function JaMaHa
%
%       author: Manuel Plasencia Gutierrez

%% BEGIN Initialization
finite_diff_step        = iniVar.finite_diff_step; % Finite diff to stimate 2nd order derivative 
cg_iter_max             = iniVar.cg_iter_max; % Maximun number of CG iterations     
ls_iter_max             = iniVar.ls_iter_max; % Maximum number of Line search iterations
cg_err                  = iniVar.cg_err; % CG error tolerance 
ls_err                  = iniVar.ls_err; % Line search error tolerance
max_step_size           = iniVar.max_step_size ; % delta is defined in globalsadd
RESET                   = iniVar.RESET; % BEGING RESET may be switched only for comparisson 
                            % and must be kept true in final code
get_min_mode_           = iniVar.get_min_mode_; % true => clasical approach of the minimum mode
                            % false => improved approach of the minimum mode
                            % first time gradient is requested in a sp 
                            % search we assume we are in area with 
                            % positive curvature  
normalize_direction    = iniVar.normalize_direction; % if true the search direction is normalized
save_lanczos_call      = iniVar.save_lanczos_call; % if true the old min mode is used every time
                            % the proposed step lenght is smaller than a 
                            % tolerance. True RECOMMENDED for expensive func
save_lanczos_factor     = iniVar.save_lanczos_factor; % Values between 0 and 20
                            % every propose step size smaller than 
                            % 'save_lanczos_factor' percent of 'max_step_size'  
                            % will use the oldeigenvalue.
maximum_speed           = iniVar.maximum_speed; % true means max_step_size will be use in areas with 
                            % negative curvature. This implies for SP-searches
                            % avandon the positive area using max_step_size
                            % instead of calculating the step lenght.
dE_max                  = iniVar.dE_max;
obligate_move           = iniVar.obligate_move; % in SP-searches the gradient at the initial 
                            % displacement can be very small therefore
                            % the system will not progress, to avoid such a 
                            % non-progress situation the gradient is multiply by 
                            % a factor when obligate move is true 
use_same_formula        = iniVar.use_same_formula;% true means using same min. mode formula when first exit + area
alf                     = iniVar.alf; % if equal zero -gradient is used as direction
n_intentos              = iniVar.n_intentos; % Attempts to force the system staying inside SP basin of attraction.
avoid_stuck_in_min      = iniVar.avoid_stuck_in_min; % if true the CG algorithm will reject 
                            % convergence in area with e_min>0 and force the
                            % system to keep searching for SPs
beta_formula            = iniVar.beta_formula; % 1->Hestenes and Stiefel, 2->Polak-Ribiere, 3->Liu and Storey
                            % 4->Dai-Yuan, 5->Fletcher-Reeves, 6->conjugate descent
                            % 7->Hestenes-Stiefel + Dai-Yuan, 8->Polak-Ribiere + Fletcher-Reeves, 
                            % 9->Liu-Storey + conjugate descent
modified_CG             = iniVar.modified_CG; % Uses a modified formula to compute search direction.
e_tol                   = iniVar.e_tol; % tolerance to evaluate uniform descent condition in CG
get_min_mode_lineSearch = iniVar.get_min_mode_lineSearch;% false-> min mode is never calculated along the line
write_to_file           = iniVar.write_out_to_hdd; % write screen output to a file

save_l_zero = true; % save data about the point where lowest eigen value is zero.   
track_energy = false; % true keeps record of energy values
%write_to_file = true; % write screen output to a file

% ----------- DO NOT CHANGE FOLLOWING VALUES -----------------------------------
write_this = ''; % initialize a text variable
write_this2 = '';
bandera = false;% flag to bypass trial step whe initial disp is very close to the minimum
contador = 0; % for checking min mode
sum_distance = 0; % for checking min mode
cross_bound = false; % for the saddle point search
k=0; % Counter for resetting CG when k=n
i=0; % Counter for CG iterations
LS_total_iter = 0; % LS iteration counter
func_calls = 0; % counter for number of function evaluations
hess_func_calls = 0; % counter for number of fun eval to get min mode
minimizer_func_calls = 0; % counter for number of fun evalfrom minimizer
discarded_steps = 0; % count number of step posible discarded due to atempt to exit basin of atraction 
discarded_steps_ = 0; % count number of step actually discarded due to atempt to exit basin of atraction 
in_vecinity_ini_minimum = true; % true means in the positive area and false means negative area
j=0; % LS counter
%count_crossing = 0; % index for file naming x_lamda_zero
%x_0 = []; % x_0 is evaluated only if x_lamda_zero is requested.
x_0 = xmin; % x_0 is over written only if x_lamda_zero is requested.(this is to avoid crash when search beging in negative area)
abort_now = false; % flag to abort the search if needed
cg=tic;% start the clock only for CG_sec
%fcn=Inf;%error track to see where inside the loop F is evaluated.

% END Initialization

 %disp(pwd) % error track
 
%% Other initialization  

    if get_min_mode_ == true
            get_min_mode = true;
    else
            get_min_mode = false;
    end
    
    if save_lanczos_call % Old min mode is used.
       % define tolerance
        tol = ls_err + (max_step_size-ls_err)*save_lanczos_factor/100;
        %disp(['save lanczos tol = ', num2str(tol)]) % error track
    end
    
    if track_energy 
        energy = zeros(cg_iter_max*ls_iter_max,1);
        ener_index = 0;
    else
        energy = [];
    end
    
    working_path = my_paths{2}; % shared memory
    
    if write_to_file
        fileID = fopen([working_path, 'scr', '.out'],'w');
        write_to_hdd(x, working_path, 'disp.con')  % initial guess/displacement       
    end    
    
    if plot_path % prealocate variables
        nn = length(x);
        x_trials = zeros(nn, cg_iter_max*ls_iter_max);
        col = 1; % column index to store in the x_trials array

        x_for_plot = zeros(nn, cg_iter_max);
        col_plot = 1; % column index to store in the x_for_plot array
        
        % takes the inital guess 
        x_for_plot(:,col_plot) = x; 
        col_plot =col_plot + 1;

        x_trials(:,col) = x;
        col =col + 1;        
        
    else % empty variables
        x_trials = [];
        x_for_plot = [];
    end
    
%% getting the function, gradient, and approximated hessian             

    % ======== Function call #1 ===============================================         
    % ======== Function call #1 ===============================================     
    [f, g, e_min, v_min, fcount, hess_count] = get_func_and_grad_v8_morse...
        (x, saddle_search, n, box, get_min_mode,[],[], in_vecinity_ini_minimum, i+j); % call #1
    % ======== Function call #1 =============================================== 
    % ======== Function call #1 =============================================== 
    %fcn=1;%error track to see where inside the loop F is evaluated.
    %fprintf(fileID,'%2u\t',fcn);
    %fprintf(fileID,'\n');
    
    minimizer_func_calls = minimizer_func_calls +1;
    func_calls = func_calls +fcount;
    hess_func_calls = hess_func_calls + hess_count; 
    
    % keep record of eigenvalue and eigenvector 
    e_min_old = e_min;
    v_min_old = v_min;

    if saddle_search % saddle search requested then relax stopping criteria
        
       % absolute tolerance gradient
        %abs_grad_tol = 0.01;  % morse
        abs_grad_tol = cg_err;
        if (max(abs(g)) <= abs_grad_tol) && (obligate_move==true)
            %g = g*1000;
            g = 1000*(g/norm(g)); % normalize g and makes it very large
            write_this = 'Despite small gradient, the system will be forced to move. {g = 1000*(g/norm(g))}';
            bandera = true;% flag to bypass trial step
        end
        if (e_min<0) 
            write_this2 = 'Warning! Inital guess in area with negative lowest eigenvalue';
            get_min_mode = true;
            if (use_same_formula == true) %  initial displacement was placed in the negative area
                % NOTE: in the case use_same_formula = false it makes no sense
                % to set in_vecinity_ini_minimum = false
                in_vecinity_ini_minimum = false;
            end
            
        end
        
    else % minimization requested
        
       % absolute tolerance gradient 
        %abs_grad_tol = 0.01; % morse
        abs_grad_tol = cg_err;
       % Overwrite maximum energy value
        dE_max = Inf;
       % do Line search
        %ls_iter_max = 10;
       % Decrease max_step_size
        %max_step_size = max_step_size/3;    
    end    

    % load the energy at the minimum where simulation began
    E_0 = current_min;
    if isempty(E_0), E_0 = f; end
    
%     % load the energy at the minimum where simulation began
%     if (labindex ~= 0) %&& (saddle_search==true)
%         %load('../0_0_0/obj.mat')
%         %disp(['obj laoded successfully ', num2str(obj)]) % error track
%         %E_0 = obj;
%         %E_0 = load('/dev/shm/MyOwn/0_0_0/obj.con');
%         %E_0 = load([my_path_to_minimum, '0_0_0/obj.con']);
%         E_0 = load([my_path_to_minimum, 'obj.con']);
%      else
%          E_0 = f;
%     end
        
   % Keeps track of the energy
    if track_energy 
        ener_index = ener_index + 1;
        energy(ener_index) = f;   
    end 

    % Initializing CG
    r = -g; % Residual  
    if (saddle_search==true) && (get_min_mode==false) % get search direction 'd'
        %d = JaMaHa(xmin, x, -g, lam, 0, n);
        d = JaMaHa(xmin, x, -g, 0, n);
    else
        if normalize_direction % Search direction    
            d = r/norm(r);
        else
            d = r;
        end        
    end
    
    ganma_new = dot(r,r);
    %ganma_0 = ganma_new;
    r_old = r;    
    
    % prints in the screen/file heading of table with progress 
    if write_to_file    
        if saddle_search
            fprintf(fileID,'Begining saddle search from initial guess');
            fprintf(fileID,'\n');
            fprintf(fileID,write_this);
            fprintf(fileID,'\n');
            fprintf(fileID,write_this2);
            fprintf(fileID,'\n');
            fprintf(fileID,'%-6s\t', 'iter');
            %fprintf(fileID,'%-8s\t', 'step size', 'delta_E', 'norm_force', 'e_min');
            fprintf(fileID,'%-8s\t', 'step size', 'delta_E', 'max force', 'e_min');
            fprintf(fileID,'\n');
            fprintf(fileID,'%2u\t',i);
            %fprintf(fileID,'%6f\t', [0 (f-E_0) norm(g) e_min]);
            fprintf(fileID,'%6f\t', [0 (f-E_0) max(abs(g)) e_min]);
            fprintf(fileID,'\n');  
        else
            fprintf(fileID,'Begining minimization from initial guess');
            fprintf(fileID,'\n');
            fprintf(fileID,'\n');
            fprintf(fileID,'%-6s\t', 'iter', 'step size', 'max force', 'energy');
            fprintf(fileID,'\n');
            fprintf(fileID,'%2u\t',i);
            fprintf(fileID,'%6f\t', [0 max(abs(g)) f]);
            fprintf(fileID,'\n');
        end     
    end

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++%
                % Begin Conjugate Gradient loop %
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++%

while (i < cg_iter_max) && (max(abs(g)) > abs_grad_tol) && (f-E_0 < dE_max) && (abort_now==false)
                         
    j =0; % Counter for the Line Search.
    ganma_d = dot(d, d);
    step_size = zeros(n,1);  
    
    % *******************BEGIN LINE SEARCH*************************%
    % *******************BEGIN LINE SEARCH*************************%
    while true && (cross_bound==false) && (f-E_0 < dE_max) % Do LS
        
        if (in_vecinity_ini_minimum == false) && (discarded_steps >= n_intentos) && (use_same_formula==true) % do not reset CG if negative area is exit
            RESET = false;
        end

        % determine value of alfa(TRIAL STEP might be needed)
        if (saddle_search) && (e_min>0) && (maximum_speed==true) || (bandera==true) % bypass trial step (func eval #2) set alfa=max_step_size
            
            %if this code validates then e_min is positive
            alfa = max_step_size;% Set alfa = max_step+size 
            bandera = false;
            if normalize_direction==false
                % set propose step. The new step is weighted by the value
                % of propose_displace and the final step is smaller
                % than the max_step_size.

                % overwrite propose_displace
                propose_displace = 100*max_step_size;

                % keeps sign of alfa
                si = sign(alfa);
                porcent = 20;
                w = (porcent*max_step_size/100) + ...
                    (-porcent*max_step_size/100*...
                    (propose_displace - max_step_size))/(99*max_step_size);

                alfa = si *(1/norm(d))*(max_step_size - w);       

            end 

        else % do trial step (func eval #2) and calculate alfa

            %if this code validates then e_min is either positive or negative
            alfa = finite_diff_step; % 
            displace = alfa*d;%   
            x_finite = ([x(1:n)+displace; x(n+1:length(x))]);        
            % In trial step I use old min mod except if atempting to enter basin of attraction:
            if plot_path 
                % Adds the finite diff steps to trial steps record
                x_trials(:,col) = x_finite;
                col =col + 1;
            end

            % ======== Function call #2 ===============================================     
            % ======== Function call #2 ===============================================
            [~, g1, e_min, v_min, fcount, hess_count] = ...
            get_func_and_grad_v8_morse...
            (x_finite,saddle_search, n, box, get_min_mode, e_min, v_min,...
                                    in_vecinity_ini_minimum, i+j);% call #2
            % ======== Function call #2 ===============================================     
            % ======== Function call #2 =============================================== 
            %fcn=2;%error track to see where inside the loop F is evaluated.
            %fprintf(fileID,'%2u\t',fcn);
            %fprintf(fileID,'\n');

            minimizer_func_calls = minimizer_func_calls +1;
            func_calls = func_calls +fcount;
            hess_func_calls = hess_func_calls + hess_count;

            H_x_d =(g1-g)/finite_diff_step;% diagonal elements of the Hessian matrix                 
            nominator = dot(g,d);
            denominator = d'*H_x_d;% approximation of matrix-vector multiplication H*d

            alfa =  -nominator/denominator;            

            if (denominator < 0) % DUDAAA REVISAARRRRRR

                % ** denominator < 0 means "+" curvature in the original function, **
                % ** if min mode transformation is applied and if the function
                % is multiply by -1

                %disp('LS failure due to Negative curvature: pushing downhill...')

                if maximum_speed == true
                    alfa = -sign(alfa)*max_step_size;
                else
                    alfa = -alfa; 
                end            

                if get_min_mode_ == false
                        get_min_mode = false;
                end

            else

                %disp('get_min_mode will be set to true so min mode is calculated...')
                        get_min_mode = true; 

            end 

            % Maximum step size limitation:
            propose_displace = norm(alfa*d);        

            if propose_displace > max_step_size
                % reduce propose step. The reduction is weighted by the value
                % of propose_displace and the final step is smaller
                % than the max_step_size.
                if propose_displace > 100*max_step_size
                    % disp('Warning! extremely large proposed step size(larger than 100*max_step_size)')
                    propose_displace = 100*max_step_size;
                end
                % keeps sign of alfa
                si = sign(alfa);
                porcent = 20;
                w = (porcent*max_step_size/100) + ...
                    (-porcent*max_step_size/100*...
                    (propose_displace - max_step_size))/(99*max_step_size);

                alfa = si *(1/norm(d))*(max_step_size - w);
            end                

        end

        displace = alfa*d;          
       
        % Remember previous step (always). Note this is NOT the trial step
        x_old = x; 
        g_old = g;
        e_m_old = e_min;
        v_m_old = v_min;

        x(1:n) = x(1:n) + displace;        

        % Keep record of the displacements
        step_size = step_size + displace;
        j = j + 1;   

        %if ( abs(dot(g/norm(g),d/norm(d)))>=cg_err*10 )
            
            if (j < ls_iter_max) && (alfa^2*ganma_d > ls_err^2) 

            % ***** Continuing Line search ****
            % ***** Continuing Line search **** 

                if plot_path
                    x_trials(:,col) = x;
                    col =col + 1;
                end

                % Determine whether to use old minimum mod or not                
                if saddle_search==true 

                    if get_min_mode_lineSearch == true
                        % calculate min mode if needed
                        if save_lanczos_call==false % Calculate min mod
                            e_min = [];
                            v_min = [];
                         else % use old min mode if displace smaller than tol
                            contador = contador +1;
                            [e_min, v_min] = analize_step_lenght(e_min, v_min, displace);
                        end

                    else
                        % dont calculate min mode in LS except last step

                    end

                end

                % ======== Function call #3 ===============================================
                % ======== Function call #3 ===============================================               
                [f, g, e_min, v_min, fcount, hess_count] = ...
                        get_func_and_grad_v8_morse(x, saddle_search, n, box,...
                    get_min_mode, e_min, v_min, in_vecinity_ini_minimum, i+j); % call #3 
                % ======== Function call #3 ===============================================
                % ======== Function call #3 ===============================================  
                %fcn=3;%error track to see where inside the loop F is evaluated.
                %fprintf(fileID,'%2u\t',fcn);
                %fprintf(fileID,'%6f\t', [3 abs(dot(g/norm(g),d))]);
                %fprintf(fileID,'\n');

                minimizer_func_calls = minimizer_func_calls +1;
                func_calls = func_calls +fcount;
                hess_func_calls = hess_func_calls + hess_count;

                if write_to_file
                    %given_step_size = norm(step_size);
                    if saddle_search
                        fprintf(fileID,'%2s ', 'LS');
                        fprintf(fileID,'%2u\t', j);
                        %fprintf(fileID,'%6f\t', [given_step_size (f-E_0) norm(g) e_min]);
                        %fprintf(fileID,'%6f\t', [given_step_size (f-E_0) max(abs(g)) e_min]);
                        fprintf(fileID,'%6f\t', [alfa (f-E_0) max(abs(g)) e_min]);
                        fprintf(fileID,'\n');            
                    else       
                        fprintf(fileID,'%2u\t',i);
                        %fprintf(fileID,'%6f\t', [given_step_size max(abs(g)) f]);
                        fprintf(fileID,'%6f\t', [alfa max(abs(g)) f]);
                        fprintf(fileID,'\n');
                   end               
                end           

                H_x_d =(g-g_old)/alfa;% diagonal elements of the Hessian matrix 
                denominator = d'*H_x_d;% approximation of matrix-vector multiplication H*d

                if (denominator < 0) % DUDAAA REVISAARRRRRR

                    % ** denominator < 0 means "+" curvature in the original function, **
                    % ** if min mode transformation is applied and if the function
                    % is multiply by -1

                    %disp('LS failure due to Negative curvature: pushing downhill...')
                    if maximum_speed == true
                        alfa = (-alfa/abs(alfa) )*max_step_size;
                    else
                        alfa = -alfa; 
                    end
                    if get_min_mode_ == false
                            get_min_mode = false;
                    end

                else

                    %disp('get_min_mode will be set to true so min mode is calculated...')
                            get_min_mode = true; 

                end 

                if  (saddle_search==true)

                    % Check if basing of atraction's boundary has been crossed.
                    if ( sign(e_min_old)+sign(e_min)==0 )

                        if save_l_zero == true 

                            x_0 = x_lamda_zero(x_old, x, e_m_old, e_min);
                           
                            % save even more data:
                            if write_to_file
                                name_it = ['l_zero_', num2str(labindex), '.mat'];
                                save([working_path, name_it], 'x_0','x_old','g_old',...
                                'e_m_old', 'v_m_old','x', 'g', 'e_min', 'v_min');
                                write_to_hdd(x_0, working_path, 'x_zero.con')
                            end

                            %count_crossing = count_crossing +1; % file name index           
                            abort_now = false; % send a signal to stop the simulation
                            save_l_zero = false; % to avoid overwritting

                        end

                        if RESET == true % stop line search and reset CG
                           cross_bound = true;

                           if (e_min<0) && (use_same_formula == true) %  means negative area was enter
                                % NOTE: in the case use_same_formula = false it makes no sense
                                % to set in_vecinity_ini_minimum = false
                                in_vecinity_ini_minimum = false;
                           end

                           if e_min_old < 0 %means attempting to exit SP basin of atraction

                                discarded_steps = discarded_steps + 1; % counter  
                                if discarded_steps<n_intentos
                                    discarded_steps_ = discarded_steps_ + 1;
                                    % disp(['Step discarded!(e_min= ', num2str(e_min), ' Atempting to reenter basin of atraction...'])
                                    if write_to_file
                                        fprintf(fileID,['step= ', num2str(norm(displace)), ' discarded!(e_min= ', num2str(e_min), ' Atempting to reenter basin of atraction...']);
                                        fprintf(fileID,'\n');
                                    end

                                    % Reject proposed step, Restart from previous step
                                    % Discard the x, g, e_min and v_min, and step backward.     
                                    x = x_old; % step backward to enter again
                                    step_size = step_size - displace; % for print out
                                    g = g_old; % Taken tha gradient at the previous step
                                    e_min = e_m_old;
                                    v_min = v_m_old;

                                else
                                    % keep record of eigenvalue and eigenvector 
                                    e_min_old = e_min;
                                    v_min_old = v_min;                            
                                end

                           else %means entering SP basin of atraction

                             % keep record of eigenvalue and eigenvector 
                             e_min_old = e_min;
                             v_min_old = v_min;

                           end

                           % disp('Boundary has been crossed. STOPPING LS  ...')
                           if write_to_file
                            fprintf(fileID,'Boundary has been crossed. STOPPING LS  ...');
                            fprintf(fileID,'\n');
                           end                       

                        else
                            cross_bound = false;
                            % keep record of eigenvalue and eigenvector 
                            e_min_old = e_min;
                            v_min_old = v_min;                        
                        end 

                    end

                end
                
                if ( abs(dot(g/norm(g),d/norm(d)))<=cg_err*10 ), break, end

            else % ********* stop Line search ***********

            % ***** Stopping Line search ****
            % ***** Stopping Line search ****             

                % Records the path
                if plot_path
                    x_trials(:,col) = x;
                    col =col + 1;

                    x_for_plot(:,col_plot) = x;
                    col_plot = col_plot + 1;
                end

                % Determine whether to use old minimum mod or not
                if saddle_search==true 

                    %if get_min_mode_lineSearch == true
                        % calculate min mode if needed
                        if save_lanczos_call==false % Calculate min mod
                            e_min = [];
                            v_min = [];
                         else % use old min mode if displace smaller than tol
                            contador = contador +1;
                            [e_min, v_min] = analize_step_lenght(e_min, v_min, displace);                    
                        end

                end

                % ======== Function call #4 =============================================== 
                % ======== Function call #4 ===============================================            
                [f, g, e_min, v_min, fcount, hess_count] = ...
                    get_func_and_grad_v8_morse(x, saddle_search, n, box,...
                get_min_mode, e_min, v_min, in_vecinity_ini_minimum, i+j); % call #4
                % ======== Function call #4 =============================================== 
                % ======== Function call #4 =============================================== 
                %fcn=4;%error track to see where inside the loop F is evaluated.
                %fprintf(fileID,'%2u\t',fcn);
                %fprintf(fileID,'%6f\t', [4 abs(dot(g/norm(g),d))]);
                %fprintf(fileID,'\n');

                minimizer_func_calls = minimizer_func_calls +1;
                func_calls = func_calls +fcount;
                hess_func_calls = hess_func_calls + hess_count;

                H_x_d =(g-g_old)/alfa;% diagonal elements of the Hessian matrix 
                denominator = d'*H_x_d;% approximation of matrix-vector multiplication H*d

                if (denominator < 0) % DUDAAA REVISAARRRRRR

                    % ** denominator < 0 means "+" curvature in the original function, **
                    % ** if min mode transformation is applied and if the function
                    % is multiply by -1

                    %disp('LS failure due to Negative curvature: pushing downhill...')
                    if maximum_speed == true
                        alfa = (-alfa/abs(alfa) )*max_step_size;
                    else
                        alfa = -alfa; 
                    end
                    if get_min_mode_ == false
                            get_min_mode = false;
                    end
                else

                    %disp('get_min_mode will be set to true so min mode is calculated...')
                            get_min_mode = true; 

                end 

                if  (saddle_search==true)

                    % Check if basing of atraction's boundary has been crossed.
                    if ( sign(e_min_old)+sign(e_min)==0 )  

                      if save_l_zero == true  

                        x_0 = x_lamda_zero(x_old, x, e_m_old, e_min);
                        
                        % save even more data:
                        if write_to_file
                            name_it = ['l_zero_', num2str(labindex), '.mat'];
                            save([working_path, name_it], 'x_0','x_old','g_old',...
                            'e_m_old', 'v_m_old','x', 'g', 'e_min', 'v_min');
                            write_to_hdd(x_0, working_path, 'x_zero.con')
                        end
                        
                        %count_crossing = count_crossing +1; % file name index                               
                        abort_now = false; % send a signal to stop the simulation
                        save_l_zero = false; % to avoid overwritting

                      end

                      if RESET == true % stop line search and reset CG  

                        cross_bound = true;

                        if (e_min<0) && (use_same_formula == true) %  means negative are was enter
                            % NOTE: in the case use_same_formula = false it makes no sense
                            % to set in_vecinity_ini_minimum = false
                            in_vecinity_ini_minimum = false;
                        end

                        if e_min_old < 0 %means attempting to exit basin of atraction

                            discarded_steps = discarded_steps + 1; % counter
                            if discarded_steps<n_intentos
                                discarded_steps_ = discarded_steps_ + 1;
                                % disp('Step discarded! Atempt to reenter basin of atraction...')
                                if write_to_file
                                    fprintf(fileID,['step= ', num2str(norm(displace)), ' discarded!(e_min= ', num2str(e_min), ' Atempting to reenter basin of atraction...']);
                                    fprintf(fileID,'\n');
                                end

                                % Reject proposed step, Restart from previous step
                                % Discard the x, g, e_min and v_min, and step backward.     
                                x = x_old; % step backward to enter again
                                step_size = step_size - displace; % for print out
                                g = g_old; % Taken tha gradient at the trial step
                                e_min = e_m_old;
                                v_min = v_m_old;

                            else
                                % keep record of eigenvalue and eigenvector 
                                e_min_old = e_min;
                                v_min_old = v_min;                            
                            end

                        else %means entering SP basin of atraction

                         % keep record of eigenvalue and eigenvector 
                         e_min_old = e_min;
                         v_min_old = v_min;                        

                        end                    

                        % disp('Boundary has been crossed. CG will reset  ...')
                        if write_to_file
                            fprintf(fileID,'Boundary has been crossed. CG will reset  ...');
                            fprintf(fileID,'\n');
                        end 

                      else 
                        cross_bound = false;
                        % keep record of eigenvalue and eigenvector 
                        e_min_old = e_min;
                        v_min_old = v_min;                        
                      end 

                    end

                end

                if plot_path
                    x_for_plot(:,col_plot) = x;
                    col_plot = col_plot + 1;
                end   

                break;

            end
            
        %else
            
            %break;

        %end
       % *********************END LINE SEARCH*******************************%  
       
    end    
    % *********************END LINE SEARCH*************************%
    % *********************END LINE SEARCH*************************% 

    LS_total_iter = LS_total_iter + j;
    given_step_size = norm(step_size);    
    
    r = -g;
    ganma_old = ganma_new;
    ganma_new = dot(r,r);
    %ganma_mix = dot(r,r_old);  
    Y = r - r_old;
    nomi = dot(Y,r);
    
    % beta_formula:
    % 1->Hestenes-Stiefel, 2->Polak-Ribiere, 3->Liu-Storey
    % 4->Dai-Yuan, 5->Fletcher-Reeves, 6->conjugate descent
    % 7->Hestenes-Stiefel+Dai-Yuan, 8->Polak-Ribiere+Fletcher-Reeves, 9->Liu-Storey+conjugate descent 
    % suitable for practical computations:    
    if beta_formula==1
            beta = nomi/dot(Y,d);
            beta = max(beta,0);  % Hestenes and Stiefel        
    elseif beta_formula==2
            beta = nomi/dot(r_old,r_old);
            beta = max(beta,0);  % Polak-Ribiere
    elseif beta_formula==3
            beta = nomi/abs(dot(r_old,d));
            beta = max(beta,0);  % Liu and Storey
% globally convergent:
    elseif beta_formula==4 
            denomi = dot(Y,d);
            beta = ganma_new/denomi;% Dai and Yuan       
    elseif beta_formula==5
            beta = ganma_new/ganma_old;% Fletcher and Reeves    
    elseif beta_formula==6
            beta = ganma_new/abs(dot(r_old,d));% Conjugate Descend                      
% combined methos:        
    elseif beta_formula==7
            denomi = dot(Y,d);
            beta1 = nomi/denomi;% Hestenes and Stiefel
            beta2 = ganma_new/denomi;% Dai and Yuan
            beta = max(0,min(beta1,beta2));% Combined        
    elseif beta_formula==8
            beta1 = nomi/dot(r_old,r_old);% Polak-Ribiere
            beta2 = ganma_new/ganma_old;% Fletcher and Reeves   
            beta = max(0,min(beta1,beta2)); % Combined
    elseif beta_formula==9
            beta1 = nomi/abs(dot(r_old,d));% Liu and Storey
            beta2 = ganma_new/abs(dot(r_old,d)); % Conjugate Descend                     
            beta = max(0,min(beta1,beta2));  % Combined          
    end

    % Compute new search direction
    if (saddle_search==true) && (get_min_mode==false)% get search direction 'd'
        
        d = JaMaHa(xmin, x, -g, alf, n);
    
    else
        
        
        if modified_CG==true
            % Use modified forula for 'd'
            d = r*( 1 + beta*(dot(g,d)/ganma_new) ) + beta*d; 
        else
            % Use clasic approach fo 'd'
            d = r + beta*d;
        end
        
        if normalize_direction
            d = d/norm(d);
        end  
         
    end    
    
    k = k + 1;
    i = i +1;
    r_old = r;
    
    if track_energy 
        ener_index = ener_index + 1;
        energy(ener_index) = f;   
    end  

    % Iteration print out     
    if write_to_file
        if saddle_search            
            fprintf(fileID,'%2u\t',i);
            %fprintf(fileID,'%6f\t', [given_step_size (f-E_0) norm(g) e_min]);
            fprintf(fileID,'%6f\t', [given_step_size (f-E_0) max(abs(g)) e_min]);
            fprintf(fileID,'\n');            
        else       
            fprintf(fileID,'%2u\t',i);
            fprintf(fileID,'%6f\t', [given_step_size max(abs(g)) f]);
            fprintf(fileID,'\n');
       end               
    end

    % === BEGIN RESETTING CG ===   
    if (k == n)  % reset CG
        
        if (saddle_search==true) && (get_min_mode==false)

            d = JaMaHa(xmin, x, -g, alf, n);

        else

            if normalize_direction % Search direction    
                d = r/norm(r);
            else
                d = r;
            end        

        end
        k = 0;    
        if write_to_file
            fprintf(fileID,'CG iter = degree of freedom. CG reset');
            fprintf(fileID,'\n');
        end                    
   
    end
    
    % if (dot(r,d) <= 0) % reset CG
    %     if normalize_direction
    %         d = r/norm(r);
    %     else
    %         d = r;
    %     end
    %     k = 0;
    %     if write_to_file
    %         fprintf(fileID,'uphill direction rejected. CG reset');
    %         fprintf(fileID,'\n');
    %     end              
    % end    

    if (dot(r,d) <= e_tol*norm(r)*norm(d)) % reset CG
        
        d = r;
        if normalize_direction
            d = d/norm(d);
        end
        k = 0;

        if write_to_file
            fprintf(fileID,'uniform descent condition not fullfilled. CG reset');
            fprintf(fileID,'\n');
        end              
 
    end      
    
    if  cross_bound && ( RESET ) % reset CG
        if normalize_direction
            d = r/norm(r);
        else
            d = r;
        end
        k = 0;
        %disp('Computed direction reset to => ') % error  tracking
        %disp(d)% error tracking          
        if plot_path
            x_for_plot(:,col_plot) = x;
            col_plot = col_plot +1;
        end
        %disp('*******************************************************************')
        %disp('CG reset due to crossing the basing of atraction boundary')
        if write_to_file
            fprintf(fileID,'CG reset due to crossing the basing of atraction boundary');
            fprintf(fileID,'\n');
        end             
%         disp('*******************************************************************')
        cross_bound = false;
    end
    
    % test for convergence to a minimum during SP search:
    if (saddle_search) && (max(abs(g))<=abs_grad_tol) && (e_min>=0) && (avoid_stuck_in_min)
        %g = g*1000;
        g = 1000*(g/norm(g));
        % Reset CG
        if (get_min_mode==false)

            d = JaMaHa(xmin, x, -g, alf, n);

        else

            if normalize_direction % Search direction    
                d = r/norm(r);
            else
                d = r;
            end        

        end        
        k = 0;           
        if write_to_file
            fprintf(fileID,'Stuck in minimum! The system will be forced to move! {g = 1000*(g/norm(g));}');
            fprintf(fileID,'\n');
            fprintf(fileID,'CG reset');
            fprintf(fileID,'\n');            
        end
        bandera = true;
       
    end    
    
    % ===END RESETTING CG ===
    
end % Conjugate gradient loop

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++%
                % End Conjugate Gradient loop %
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++%



    % ------------ BEGING CONVERGE PRINT OUT 
    if (max(abs(g)) < abs_grad_tol) 
        
        x_min = x;
        final_energy = f;
        
        if saddle_search 
            
            if (e_min < 0) % accept the convergence
                %disp('Saddle search converged!')
                if write_to_file
                    fprintf(fileID,'Saddle search converged!');
                    fprintf(fileID,'\n');
                end  
            % ******* Recalculate min mode if save_lanczos_cal is true

                if save_lanczos_call % eigenvector are NOT recalculated, old are used.

                    %only if convergence is achieved with save_lanczos_call = true 
                    
                    % ======== Function call #5 =============================================== 
                    % ======== Function call #5 =============================================== 
                    [~, ~, ~, v_min, fcount, hess_count] = ...
                    get_func_and_grad_v8_morse(x, saddle_search, n, box,...
                    get_min_mode, [], [], in_vecinity_ini_minimum, i+j); % call #5
                    % ======== Function call #5 =============================================== 
                    % ======== Function call #5 =============================================== 
                    %fcn=5;%error track to see where inside the loop F is evaluated.
                    %fprintf(fileID,'%2u\t',fcn);
                    %fprintf(fileID,'\n');
                    
                    minimizer_func_calls = minimizer_func_calls +1;
                    func_calls = func_calls +fcount;
                    hess_func_calls = hess_func_calls + hess_count;   

                end

            % ******* END Recalculate min mode if save_lanczos_call is true

            else % reject convergence because e_min >0                 
                final_energy = [];
                x_min = x;
                v_min = [];
                %disp('Saddle search did not converge!')
                if write_to_file
                    fprintf(fileID,'Warning! Stuck in minimum!');
                    fprintf(fileID,'Saddle search did not converge!');
                    fprintf(fileID,'\n');
                end  
            end

        else

            %disp('Minimization converged!')
            if write_to_file
                fprintf(fileID,'Minimization converged!');
                fprintf(fileID,'\n');
            end   

        end
        e_string = ['Energy at initial state (minimum) = ', num2str(E_0)];
        %final_energy = f;
        F_e_string = ['Final energy = ', num2str(f)];
        EE_string = ['E_min-E = ', num2str(f-E_0)];
        if write_to_file
            fprintf(fileID,e_string);
            fprintf(fileID,'\n');
            fprintf(fileID,F_e_string);
            fprintf(fileID,'\n');
            fprintf(fileID,EE_string);
            fprintf(fileID,'\n');
        end      

    else % not converge        
        final_energy = [];
        x_min = x;
        v_min = [];
    end
    % ------------ END CONVERGE PRINT OUT

    % ------------ BEGING NOT CONVERGE PRINT OUT

    if i == cg_iter_max % all cg iter are given
        
      if  (max(abs(g)) > abs_grad_tol) % maximum g component is bigger than tolerance
          
        if saddle_search
            %disp('Saddle search did not converge. Increase number of iterations')
            if write_to_file
                fprintf(fileID,'Saddle search did not converge. Increase number of iterations');
                fprintf(fileID,'\n');
            end         
        else
            %disp('Minimization did not converge. Increase number of iterations')
            if write_to_file
                fprintf(fileID,'Minimization did not converge. Increase number of iterations');
                fprintf(fileID,'\n');
            end        
        end
        
        final_energy = [];
        x_min = x;
        v_min = [];
        
      end
      
    end

    if f-E_0 > dE_max

        if write_to_file
            fprintf(fileID,'Saddle Search Terminated High Energy');
            fprintf(fileID,'\n');
        end 

        final_energy = [];
        x_min = x;
        v_min = [];         

    end

    % ------------ END NOT CONVERGE PRINT OUT

    if write_to_file
    
        t1 = ['Number of LS iterations:  ', num2str(LS_total_iter)];
        t2 = ['Number of Hessian function calls:  ', num2str(hess_func_calls)];
        t3 = ['Number of minimizer function calls:  ', num2str(minimizer_func_calls)];
        t4 = ['Total number of function calls:  ', num2str(func_calls)];
        t5 = ['Steps discarded due to atempt exit basin of attraction:  ', num2str(discarded_steps_)];
        ti = toc(cg);
        tim = ['Elapsed time is ', num2str(ti), ' seconds '];
        dat = datestr(now);
        end_simu = ['Simulation finished on:  ', dat ];        
        
        fprintf(fileID,t1);
        fprintf(fileID,'\n');
        fprintf(fileID,t2);
        fprintf(fileID,'\n');
        fprintf(fileID,t3);
        fprintf(fileID,'\n');
        fprintf(fileID,t4);
        fprintf(fileID,'\n');
        fprintf(fileID,t5);
        fprintf(fileID,'\n');
        fprintf(fileID,'Line search method: Newton-Raphson');
        fprintf(fileID,'\n');
        fprintf(fileID,tim);
        fprintf(fileID,'\n');
        fprintf(fileID,end_simu);        
        fclose(fileID); % close the file tha keeps screen output
        
        % writes min/sp to hdd
        if saddle_search
            f_name = 'sp.con';
            % Writes to HDD the min mode
            write_to_hdd(v_min, working_path, 'v_min.con')
        else
            f_name = 'min.con';
        end
        write_to_hdd(x_min, working_path, f_name)

        % writes value of objective function
        write_to_hdd(f, working_path, 'obj.con')

        % writes number of function calls
        write_to_hdd(func_calls, working_path, 'fcs.con')         
        
    end      
    % Delete all empty rows from the paths arrays
    if plot_path
                x_for_plot(:,((col_plot):1:cg_iter_max)) = [];
                x_trials(:,((col):1:cg_iter_max*ls_iter_max)) = [];
                % writes optimization path to hdd
                write_to_hdd(x_for_plot, working_path, 'x_for_plot')
                % writes optimization path to hdd
                write_to_hdd(x_trials, working_path, 'x_trials')            

    end
    % Delete all empty colums from the energy vector
    if track_energy 
        ener_index = ener_index + 1;
        energy(ener_index:length(energy)) = []; 
        % writes value of objective function at each iter
        write_to_hdd(energy, working_path, 'energy.con')     
    end       

    function [e_min, v_min] = analize_step_lenght(e_min_, v_min_, distance)
        % function to analyze whether to reclaculate or not the minimum
        % mode. The function returns to empty variables when min mode needs
        % to be recalculated.
        
            if contador == 1 % means first time
                

                    %d = d/norm(d); % be sure d is normalized
                    sum_distance = distance; %initialize vector sum_distance
                    if norm(sum_distance) > tol; % recalculate e_min and v_min
                        %disp(['Acumulated Distance =  ', num2str(norm(sum_distance)), ' => recalculate e_min and v_min'])
                        e_min = [];
                        v_min = [];
                        % Reset sum_distance and contador to begin adding from this point
                        sum_distance = 0*d; 
                        contador = 0;
                        
                    %draw_circle(x(1),x(2),tol);
                    %plot(x(1), x(2), 's b'); %initial position                        
                        
                    else
                    %******
                    % Use old e_min, v_min so do not empty the variables
                        e_min = e_min_;
                        v_min = v_min_;                    
                    %disp(['First proposed step smaller than ',num2str(tol), ' => Use old minimum mode'])
                    %disp(['Acumulated Distance =  ', num2str(norm(sum_distance)), ' => Use old minimum mode'] )
                    %******                   
                    end

            else % means is not the first time
                    %d = d/norm(d); % be sure d is normalized

                    sum_distance = sum_distance + distance; % Add all distances along the line
                    if norm(sum_distance) > tol; % recalculate e_min and v_min
                        %disp(['Acumulated Distance =  ', num2str(norm(sum_distance)), ' => recalculate e_min and v_min'])
                        e_min = [];
                        v_min = [];
                        % Reset sum_distance and contador to begin adding from this point
                        sum_distance = 0*d; 
                        contador = 0;
                        
                    %draw_circle(x(1),x(2),tol);
                    %plot(x(1), x(2), 's b'); %initial position                        
                        
                    else
                    %******
                    % Use old e_min, v_min so do not empty the variables
                        e_min = e_min_;
                        v_min = v_min_;                      
                    %disp(['Acumulated Distance =  ', num2str(norm(sum_distance)), ' => Use old minimum mode'] )
                    %******                   
                    end           

            end        
        
    end
        

end % main function
