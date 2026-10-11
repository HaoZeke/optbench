	function [x_min, x_for_plot, x_trials, final_energy, v_min,...
    minimizer_func_calls, hess_func_calls, func_calls, LS_total_iter, i, x_0 ] = LBFGS_morse_v6(x, n,...
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
% What is new in LBFGS_nr_morse:
%   . LBFGS implementation on top of the CG method. Line search remains
%   same.
% What is new in LBFGS_morse:
%   . LS is removed and lenght of vector is used as a step size.
%   . memory size 'L' is read from iniConfig file
%   . fixed bug when stuck in minimum
%       author: Manuel Plasencia Gutierrez

%% BEGIN Initialization

finite_diff_step    = iniVar.finite_diff_step; % Finite diff to stimate 2nd order derivative 
cg_iter_max         = iniVar.cg_iter_max; % Maximun number of CG iterations     
ls_iter_max         = iniVar.ls_iter_max; % Maximum number of Line search iterations
cg_err              = iniVar.cg_err; % CG error tolerance 
ls_err              = iniVar.ls_err; % Line search error tolerance
max_step_size       = iniVar.max_step_size ; % delta is defined in globalsadd
RESET               = iniVar.RESET; % BEGING RESET may be switched only for comparisson 
              % and must be kept true in final code
get_min_mode_       = iniVar.get_min_mode_; % true => clasical approach of the minimum mode
                      % false => improved approach of the minimum mode
                      % first time gradient is requested in a sp 
                      % search we assume we are in area with 
                      % positive curvature  
normalize_direction = iniVar.normalize_direction; % if true the search direction is normalized
save_lanczos_call   = iniVar.save_lanczos_call; % if true the old min mode is used every time
                           % the proposed step lenght is smaller than a 
                           % tolerance. True RECOMMENDED for expensive func
%save_lanczos_factor = 2.5;
save_lanczos_factor = iniVar.save_lanczos_factor; % Values between 0 and 20
                         % every propose step size smaller than 
                         % 'save_lanczos_factor' percent of 'max_step_size'  
                         % will use the oldeigenvalue.
maximum_speed       = iniVar.maximum_speed; % true means max_step_size will be use in areas with 
                      % negative curvature. This implies for SP-searches
                      % avandon the positive area using max_step_size
                      % instead of calculating the step lenght.
dE_max              = iniVar.dE_max;
%dE_max = Inf;                      
%dE_max = 24.0; % maximum delta energy value. SP searchs above this values 
              % are aborted
obligate_move       = iniVar.obligate_move; % in SP-searches the gradient at the initial 
                      % displacement can be very small therefore
                      % the system will not progress, to avoid such a 
                      % non-progress situation the gradient is multiply by 
                      % a factor when obligate move is true 
use_same_formula    = iniVar.use_same_formula;% true means using same min. mode formula when first exit + area
alf                 = iniVar.alf; % if equal zero -gradient is used as direction
n_intentos          = iniVar.n_intentos; % Attempts to force the system staying inside SP basin of attraction.
avoid_stuck_in_min  = iniVar.avoid_stuck_in_min; % if true the CG algorithm will reject 
                           % convergence in area with e_min>0 and force the
                           % system to keep searching for SPs

e_tol               = iniVar.e_tol; % tolerance to evaluate uniform descent condition in CG
write_to_file       = iniVar.write_out_to_hdd; % write screen output to a file
L                   = iniVar.memory_size;               % number of updates stored

save_l_zero = true; % save data about the point where lowest eigen value is zero.   
track_energy = false; % true keeps record of energy values

% ----------- DO NOT CHANGE FOLLOWING VALUES -----------------------------------
write_this = ''; % initialize a text variable
write_this2 = '';
bandera = false;% flag to bypass trial step whe initial disp is very close to the minimum
contador = 0; % for checking min mode
sum_distance = 0; % for checking min mode
cross_bound = false; % for the saddle point search
%k=0; % Counter for resetting CG when k=n
i=0; % Counter for CG iterations
LS_total_iter = 0; % LS iteration counter
func_calls = 0; % counter for number of function evaluations
hess_func_calls = 0; % counter for number of fun eval to get min mode
minimizer_func_calls = 0; % counter for number of fun evalfrom minimizer
discarded_steps = 0; % count number of step posible discarded due to atempt to exit basin of atraction 
discarded_steps_ = 0; % count number of step actually discarded due to atempt to exit basin of atraction 
in_vecinity_ini_minimum = true; % true means in the positive area and false means negative area
%j=0; % LS counter
%count_crossing = 0; % index for file naming x_lamda_zero
%x_0 = []; % x_0 is evaluated only if x_lamda_zero is requested.
x_0 = xmin; % x_0 is over written only if x_lamda_zero is requested.(this is to avoid crash when search beging in negative area)
abort_now = false; % flag to abort the search if needed
cg=tic;% start the clock only for CG_sec
%fcn=Inf;%error track to see where inside the loop F is evaluated.
g = ones(size(x(1:n)))*1000; % just to start the while-loop
e_min = [];
v_min = [];


% storage for vectors s, y, and scalars rho that define BFGS matrices
%n      = number of free atoms
  
BFGSs   = zeros(n,L);
BFGSy   = zeros(n,L);
BFGSrho = zeros(L,1);
BFGSa   = zeros(L,1);
Lc      = 0;               % current number of updates stored.
H0      = 0.01;
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
        n = length(x);
        x_trials = zeros(n, cg_iter_max*ls_iter_max);
        col = 1; % column index to store in the x_trials array

        x_for_plot = zeros(n, cg_iter_max);
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
    
    if saddle_search % saddle search requested then relax stopping criteria
        
       % absolute tolerance gradient
        %abs_grad_tol = 0.01;  % morse
        abs_grad_tol = cg_err;
        
    else % minimization requested
        
       % absolute tolerance gradient 
        %abs_grad_tol = 0.01; % morse
        abs_grad_tol = cg_err;
       % Overwrite maximum energy value
        dE_max = Inf;
    
    end    

    % load the energy at the minimum where simulation began
    E_0 = current_min;
    f = E_0; % just to start the while-loop
    %if isempty(E_0), E_0 = f; end
    if isempty(E_0), E_0 = 0.0; f = E_0; end


%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++%
                % Begin LBFGS loop %
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++%

while (i < cg_iter_max) && (max(abs(g)) > abs_grad_tol) && (f-E_0 < dE_max) && (abort_now==false)           

    % Gets f and G
    
    % Determine whether to use old minimum mod or not
    if saddle_search==true 
        if (i==0)
            e_min = [];
            v_min = [];            
        else

            % calculate min mode if needed
            if save_lanczos_call==false % Calculate min mod
                e_min = [];
                v_min = [];
             else % use old min mode if displace smaller than tol
                contador = contador +1;
                [e_min, v_min] = analize_step_lenght(e_min, v_min, dx);                    
            end
        end
    end    
    
    % ======== Function call #1 ===============================================         
    % ======== Function call #1 ===============================================     
        [f, g, e_min, v_min, fcount, hess_count] = get_func_and_grad_v8_morse...
            (x, saddle_search, n, box, get_min_mode, e_min, v_min, in_vecinity_ini_minimum, i); % call #1
    % ======== Function call #1 =============================================== 
    % ======== Function call #1 =============================================== 
    
    minimizer_func_calls = minimizer_func_calls +1;
    func_calls = func_calls +fcount;
    hess_func_calls = hess_func_calls + hess_count; 

    if saddle_search % saddle search requested then relax stopping criteria
        
        if (e_min<0) && (i==0)
            write_this2 = 'Warning! Inital guess in area with negative lowest eigenvalue';
            get_min_mode = true;
            if (use_same_formula == true) %  initial displacement was placed in the negative area
                % NOTE: in the case use_same_formula = false it makes no sense
                % to set in_vecinity_ini_minimum = false
                in_vecinity_ini_minimum = false;
            end
            
        end
         
    end    
        
   % Keeps track of the energy
    if track_energy 
        ener_index = ener_index + 1;
        energy(ener_index) = f;   
    end 

    % prints in the screen/file heading of table with progress 
    if write_to_file  
       if i==0 
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
       else
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
    end   
    
    if (i>0), updateLBFGS(x(1:n), x_c(1:n), g, g_c); end

    %dx = getStep(g);
    dx = getStep(g);
    if bandera==true
       dx = max_step_size*(dx/norm(dx));
       bandera=false;
    end

    % save current, position 
    x_c  = x;

    x(1:n) = x(1:n) + dx;% update x            
%=========================================================
    if (i>0)
        
            if  (saddle_search==true)

                % Check if basing of atraction's boundary has been crossed.
                if ( sign(e_m_old)+sign(e_min)==0 )  

                  if save_l_zero == true  

                    x_0 = x_lamda_zero(x_c, x, e_m_old, e_min);

                    % save even more data:
                    if write_to_file
                        name_it = ['l_zero_', num2str(labindex), '.mat'];
                        save([working_path, name_it], 'x_0','x_c','g_c',...
                        'e_m_old', 'x', 'g', 'e_min', 'v_min');
                        write_to_hdd(x_0, working_path, 'x_zero.con')
                    end

                    %count_crossing = count_crossing +1; % file name index                               
                    abort_now = false; % send a signal to stop the simulation
                    save_l_zero = false; % to avoid overwritting

                  end

                end

            end 
            
    end
%====================================================
    % save current gradient
    g_c      = g; 
    
    % keep record of eigenvalue  
    e_m_old  = e_min;

    % Keep record of the displacements
    step_size = dx;                     

    % Records the path
    if plot_path
        x_trials(:,col) = x;
        col =col + 1;

        x_for_plot(:,col_plot) = x;
        col_plot = col_plot + 1;
    end

    given_step_size = norm(step_size);          
  
    i = i +1;    

    if track_energy 
        ener_index = ener_index + 1;
        energy(ener_index) = f;   
    end  

    % === BEGIN RESETTING CG ===   
    
    if  cross_bound && ( RESET ) % reset CG
        if normalize_direction
            d = -g/norm(-g);
        else
            d = -g;
        end
        %k = 0;
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
        
        g = 1000*(g/norm(g));       
        if write_to_file
            fprintf(fileID,'Stuck in minimum! The system will be forced to move! {g = 1000*(g/norm(g));}');
            fprintf(fileID,'\n');
        end
        bandera = true;
       
    end    
    
    % ===END RESETTING CG ===
    
end % LBFGS loop

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++%
                % End LBFGS loop %
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++%
LS_total_iter = i; % number of LBFS steps is pased.


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
                    
                    % ======== Function call #4 =============================================== 
                    % ======== Function call #4 =============================================== 
                    [~, ~, ~, v_min, fcount, hess_count] = ...
                    get_func_and_grad_v8_morse(x, saddle_search, n, box,...
                    get_min_mode, [], [], in_vecinity_ini_minimum, i); % call #4
                    % ======== Function call #4 =============================================== 
                    % ======== Function call #4 =============================================== 

                    
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
                            sum_distance = 0*distance; 
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
                            sum_distance = 0*distance; 
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

        function [v_out] = maxMotionAppliedV(v_in , maxMotion)
            v_out = v_in;
            max_ = norm(v_in);
            if (max_ > maxMotion), v_out = v_in * maxMotion/max_; end
        end

        function updateLBFGS(x, x_c, g, g_c)
            
            % update BFGS marices
            if ( Lc == L ) % storage limit reached  
               %disp([' storage limit reached ', num2str(Lc)]) 
               BFGSs(:,1:L-1) = BFGSs(:,2:L);
               BFGSy(:,1:L-1) = BFGSy(:,2:L);
               BFGSrho(1:L-1) = BFGSrho(2:L);
               Lc             = L-1;
            end
            Lc = Lc+1;
            BFGSs(:,Lc) = (x - x_c);
            BFGSy(:,Lc) = (g - g_c);
            BFGSrho(Lc) = 1/dot( BFGSy(:,Lc), BFGSs(:,Lc) );             
            
        end
    
        function dx = getStep(g)
           my_return = false; % initialization to stop execution of the function
           % Get curvature along search direction. Do trial step here
           % to estimate for first approximation:
            
            if  (i>0)
                
                if bandera==true % take max step along gradient
                    C = -0.01; % tis is just a trick to force max_step_size
                    bandera=false;
                else
                    %C = (fPrev-f).dot(fPrev-f)/dr.dot(fPrev-f); from eon
                    C = dot( BFGSy(:,Lc), BFGSy(:,Lc))/dot( BFGSs(:,Lc), BFGSy(:,Lc) ) ;
                end
                
                if C<0% reset LBFGS and take max_step_size
                    
                   Lc = 0; % reset LBFGS
                   dx = maxMotionAppliedV(-g*1000 , max_step_size);
                   %disp(['dx should be = ' , num2str(norm(dx))])
                    %-ERROR TRACKING
                    if write_to_file
                        fprintf(fileID,'Negative curvature.  Reseting lbfgs and taking max step ');
                        fprintf(fileID,'\n');
                        %fprintf(fileID, ['dx should be = ' , num2str(norm(dx))]);
                        %fprintf(fileID,'\n');
                    end
                    %- ERROR TRACKING  
                    
                   my_return = true;
        
                end
            end 
           
            if  (i==0) % first iteration

                if bandera==true % take max step along gradient
                   C = -0.01; % tis is just a trick to force max_step_size
                   bandera=false;
                else

                    % do trial step (func eval #2) 
                    d = -g/norm(g);
                    x_finite = ([x(1:n)+ finite_diff_step*d; x(n+1:length(x))]);        
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
                                            in_vecinity_ini_minimum, i);% call #2
                    % ======== Function call #2 ===============================================     
                    % ======== Function call #2 =============================================== 

                    minimizer_func_calls = minimizer_func_calls +1;
                    func_calls = func_calls +fcount;
                    hess_func_calls = hess_func_calls + hess_count;

                    %H_x_d =(g1-g)/finite_diff_step;% diagonal elements of the Hessian matrix                 
                    %nominator = dot(g,d);
                    %denominator = d'*H_x_d;% approximation of matrix-vector multiplication H*d
                    %alfa =  -nominator/denominator; 

                    %ganma_k = denominator; % Initial curvature
                    %C = dg.dot(f.normalized())/parameters->finiteDifference;
                    C = dot( (g1-g), d)/finite_diff_step;
                end

                if C<0 % reset LBFGS and take max_step_size

                    %disp('Negative curvature calculated via FiniteDiff. Reseting lbfgs and taking max step ')
                    Lc = 0; % reset LBFGS
                    dx = maxMotionAppliedV(-g*1000 , max_step_size);
                    %disp(['dx should be = ' , num2str(norm(dx))])
                    %-ERROR TRACKING
                    if write_to_file
                        fprintf(fileID,'Negative curvature calculated via FiniteDiff. Reseting lbfgs and taking max step');
                        fprintf(fileID,'\n');
                        %fprintf(fileID, ['dx should be = ' , num2str(norm(dx))]);
                        %fprintf(fileID,'\n');
                    end
                    %- ERROR TRACKING                    
                    
                    my_return = true;
                end
                
            end
            
            if my_return == true; 
%                 %-ERROR TRACKING
%                 if write_to_file
%                     fprintf(fileID,['function should exit now with dx = ', num2str(norm(dx))]);
%                     fprintf(fileID,'\n');                    
%                 end
%                 %- ERROR TRACKING                

                return; 
                
            end
            
%             if my_return == true; 
%                 %-ERROR TRACKING
%                 if write_to_file
%                     fprintf(fileID,'function return es una MIERDA');
%                     fprintf(fileID,'\n');                    
%                 end
%                 %- ERROR TRACKING                            
%             end            
           % => END get curvature along search direction (trial step)
            H0 = 1/C;
%             disp(['H0 =  ', num2str(H0)])

           % L-BFGS two-loop recursion: 
            q = g;
% disp(['antes loop norm(q) =  ', num2str(norm(q))])
% disp(['Lc =  ', num2str(Lc)])            
            for ii = Lc:-1:1
               BFGSa(ii) = BFGSrho(ii) * dot( BFGSs(:,ii), q);
               q        = q - BFGSa(ii)*BFGSy(:,ii);
            end
% disp(['despues loop norm(q) =  ', num2str(norm(q))])
           % Compute Search direction (d = - H*g):
            r = H0*q;

            for ii = 1:Lc
               scal = dot( BFGSy(:,ii), r);
               r    = r + (BFGSa(ii) - BFGSrho(ii)*scal)*BFGSs(:,ii);
            end    

            % Compute search direction
            p = -r;  
            
            distance = norm(p);
            %disp(['norm(p) =  ', num2str(distance)])
  
            if (distance >= max_step_size)% reset LBFGS and take max_step_size
        
                Lc = 0; % reset LBFGS
                dx = maxMotionAppliedV(H0*-g , max_step_size);
                %-ERROR TRACKING
                if write_to_file
                    fprintf(fileID,['reset memory, proposed step too large (', num2str(distance),') ']);
                    fprintf(fileID,'\n');
                    %fprintf(fileID, ['dx should be = ' , num2str(norm(dx))]);
                    %fprintf(fileID,'\n');
                end
                %- ERROR TRACKING                 
                
                my_return = true;

            end   

            vd = dot( p/distance, -g/norm(g));
            if vd<0 % reset LBFGS and take max_step_size
         
                Lc = 0; % reset LBFGS
                dx = maxMotionAppliedV(H0*-g , max_step_size);
                %-ERROR TRACKING
                if write_to_file
                    fprintf(fileID,' Search direction is not a descent direction. Reseting lbfgs and taking max step.');
                    fprintf(fileID,'\n');
                    %fprintf(fileID, ['dx should be = ' , num2str(norm(dx))]);
                    %fprintf(fileID,'\n');                    
                end
                %- ERROR TRACKING                 
                
                my_return = true;
            end
            
            if my_return == true; 
%                 %-ERROR TRACKING
%                 if write_to_file
%                     fprintf(fileID,['function should exit now with dx = ', num2str(norm(dx))]);
%                     fprintf(fileID,'\n');                    
%                 end
%                 %- ERROR TRACKING                

                return; 
                
            end
            
%             if my_return == true; 
%                 %-ERROR TRACKING
%                 if write_to_file
%                     fprintf(fileID,'function return es una MIERDA');
%                     fprintf(fileID,'\n');                    
%                 end
%                 %- ERROR TRACKING                            
%             end
            
            dx = maxMotionAppliedV(p , max_step_size);
%                 %-ERROR TRACKING
%                 if write_to_file
%                     fprintf(fileID,['my_return never validated dx should be dx = ', num2str(norm(dx))]);
%                     fprintf(fileID,'\n');                    
%                 end
%                 %- ERROR TRACKING             
            
        end

end % main function
