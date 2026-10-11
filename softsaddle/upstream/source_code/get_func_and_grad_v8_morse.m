function [f, df, e_min, v_min, fcount, lanc_fcount] = ...
                                              get_func_and_grad_v8_morse...
 (x, saddle_search, n, boxx, get_min_mode, e_min_old, v_min_old, in_vecinity_ini_minimum, current_cg_iter)
% NOTE:
% libmorse:
%           returns force (force = -gradient)
%           force is a colum vector dimension=N
% liblanczos:
%           returns gradient, colum vector dimension=n
%
% Variables
    % n number of variables
    % N number of atoms
    N = n/3; % number of free atoms
    N_T = 343; % total number of atoms  6_layer x 56_atoms + 7_atoms = 343

% Initializing
    lanc_fcount = 0;
    fcount = 0;
    df_ = x; % Just to initialize the df_ variable
    
    %disp(get_min_mode)% Error track
                     
 if saddle_search==false % minimization requested
    
    % ********* BEGIN Calling external function
    [f, ~, g, ~] = calllib('libmorse', 'force', N_T, x, df_, boxx);% func eval
    fcount = fcount + 1;
    % ******** END Calling external function

    df = -g(1:n); % takes only the gradient component of free atoms
    e_min = []; % this values needs to be compute only for sp search
    v_min = []; % this values needs to be compute only for sp search
        
 else % SP search requested

    % ++++++++ BEGIN Computing minimum mode (lanczos)
           
    if get_min_mode % clasical minimum-mode
        
        % Lanczos initialization
        g_dummy = ones(n,1)/2; % required by lanczos
        v_ = g_dummy; % initializing v (it doesn't matter the value)
        f_ = -1.0; % initializing f_

        if isempty(v_min_old) % update min mode

            % disp('Calling Lanczooooooos...')
            % call lanczos to get min mode        
            [e_min, ~, g, v_min, f] = calllib...
                          ('lib', 'lanczos', N, x, g_dummy, v_, f_);

            % disp(['lanczos e_min =  ', num2str(e_min)])             
            reporta = 0; % 0=> not write report; 1=> write a short report
            [~,given_iter] = calllib('lib', 'report', 0.,reporta);
            lanc_fcount = given_iter +1; % func calls inside lanczos
            fcount = fcount + lanc_fcount; % total func calls  
            

        else % use the old/previous min mode

            %disp('Using old eigenvalue and eigenvector. Clever you!')
            v_min = v_min_old;
            e_min = e_min_old;
            %disp(['old e_min =  ', num2str(e_min)]) 
            % gradient and function values
            [f, ~, df, ~] = calllib('libmorse', 'force', N_T, x, df_, boxx);% func eval
            fcount = fcount + 1;

            g = -df(1:n); % takes only the gradient component of free atoms

        end       
        
     % ------------ BEGIN Effective Gradient ------------           
        norma = norm(v_min);% Normalizing min mode(ask JC whether min mode is returned normalized)
        v_min = v_min/norma;          
        d_product = dot(g, v_min);  

        if (e_min >= 0) && (in_vecinity_ini_minimum==true)

            % disp('Warning!! positive eigenvalue found')
            g_eff = -d_product * v_min; % clasic approach
            %g_eff = -g;

        else  % if e_min < 0 or e_min>0 and not in vecinity of initial minimum

            g_eff = g - (2*d_product * v_min);

        end
     % ------------ END Effective Gradient ------------             

    else  % get_min_mode=false
        
        %disp('Warning! applaying crazzy ideas...')
        if current_cg_iter==0 % if first CG_iter and first LS_iter
             
            % call Lanczos
            % Lanczos initialization
            g_dummy = ones(n,1)/2; % required by lanczos
            v_ = g_dummy; % initializing v (it doesn't matter the value)
            f_ = -1.0; % initializing f_
   
            if isempty(v_min_old) % update min mode

                % disp('Calling Lanczooooooos...')
                % call lanczos to get min mode        
                [e_min, ~, g, v_min, f] = calllib...
                              ('lib', 'lanczos', N, x, g_dummy, v_, f_);

                % disp(['lanczos e_min =  ', num2str(e_min)])             
                reporta = 0; % 0=> not write report; 1=> write a short report
                [~,given_iter] = calllib('lib', 'report', 0.,reporta);
                lanc_fcount = given_iter +1; % func calls inside lanczos
                fcount = fcount + lanc_fcount; % total func calls  


            else % use the old/previous min mode

                %disp('Using old eigenvalue and eigenvector. Clever you!')
                v_min = v_min_old;
                e_min = e_min_old;
                %disp(['old e_min =  ', num2str(e_min)]) 
                % gradient and function values
                [f, ~, df, ~] = calllib('libmorse', 'force', N_T, x, df_, boxx);% func eval
                fcount = fcount + 1;

                g = -df(1:n); % takes only the gradient component of free atoms

            end                         
            
            %g = v_min; % return min mode as gradient so it is used as direction
            % ------------ BEGIN Effective Gradient ------------           
                norma = norm(v_min);% Normalizing min mode
                v_min = v_min/norma;          
                d_product = dot(g, v_min);  

                if (e_min >= 0) %&& (in_vecinity_ini_minimum==true)

                    % disp('Warning!! positive eigenvalue found')
                    %g_eff = -d_product * v_min; % clasic approach
                    g_eff = -g; % testing

                else 

                    %g_eff = g - (2*d_product * v_min);
                    g_eff = -g; % testing

                end
            % ------------ END Effective Gradient ------------                     
            
        else
        
            % ********* BEGIN Calling morse potential 
            [f, ~, force, ~] = calllib('libmorse', 'force', N_T, x, df_, boxx);% func eval
            fcount = fcount + 1;   

            g = -force; % return gradient vector
            % ******** END Calling morse potential 
            
            e_min = +1;% I need the sign in CG_sec.m to see whether
                       % boundaries of basisn of atraction have been crossed
            v_min = []; % this values needs to be compute only for sp search 
            
            g_eff = -g(1:n); % returns -gradient
        
        end
                    
    end

 
    df = g_eff; % <=
        
 end

  %disp(['get_func_and_grad_v6.m -> e_min =  ', num2str(e_min)])
end
