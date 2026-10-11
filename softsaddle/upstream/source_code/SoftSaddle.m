function [obj_funct, x_global_min, fm_global_list,  ...
    m_global_list, fsp_global_list, sp_global_list, table_connected ...
    ] = SoftSaddle
%% *************************************************************************
% Main function for the Improved MMF method as described in this publication: 
% 
%     Improved Minimum Mode Following Method for Finding First Order Saddle Points
%     Manuel Plasencia Guti?rrez, Carlos Arg?ez, and Hannes J?nsson
%     Journal of Chemical Theory and Computation Article ASAP
%     DOI: 10.1021/acs.jctc.5b01216
% 
%  Manuel Plasencia Gutierrez 2013-2015
%  email: mpg2@hi.is
% *************************************************************************
% 
%
%
%
%
%
%
% Updates and debugging records:
% What is new in version 4:
%	A visit from a minimum is considered finished when:
%	a) a given number of unique SP is at least found or
%	b) a given number of unsuccessfull atempt have been made(this is in v4 too) 
%
% What is new in version 5_morse:
%   a) Load a dedicated library for morse potential
%   b) libraries are loaded here and not in MyOwn
%
% What is new in version 6_morse:
%   a) It is friendly with the 7 pt atoms island test problem
%   b) takes better care of connected SPs and statistics
% What is new in version 7_morse:
%   a) It simplify the code and minimize from every unique SP
%   b) a random atom can be picked to be displaced
% What is new in version 71_morse:
%   a) It fixs bugs related to bookkeeing simulation results
%   b) matlabpool opens here only once, intead of opening in MyOwn 
%      function several times.
% What is new in version 72_morse:
%   a) Improve speed by checking for repeated SPs only after all attempts.   
%   b) A maximum execution time can be given(only applies for the SP search loop)
% What is new in version 73_morse:
%   a) A maximum array size is defined to aboy out of memory, if a larger
%   array is required the simulation run several batches and process them
%   before running the  next
% What is new in version 74_morse:
%   a) More efficient code. Bug related to folder structure fixed
% What is new in version 76_morse:
%   a) Can read a batch of initial displacements for reproducing
%   somebadyelse results.
% What is new in version 77_morse:
%   a) Keeps records of all SP found and write a report with info about
%   been unique, repeated, bad, etc.
% What is new in version 78_morse:
%   a) tmp folder allows to execute multiple matlab sesions simultaniusly
%   without conflicting each other. (tmp folder is in shared memory and
%   uses and unique random ID)
% What is new in version 79_morse:
%   a) all variables to set up the simulation is readed from an iniConfig
%   file, stored in a matlab structure and pass it to all child function.
%   This will allow to run several batch of parallel run with different
%   variables values each.
% What is new in version 80_morse:
%   a) The hypershpere has variable radius.
% What is new in version 81_morse:
%   a) enclose eerything in a loop to perform a single parameter automatic
%   repetition.
% What is new in version 82_morse:
%   a) It is posible to choose between automatic_self_learning and NOT 
%   automatic_self_learning, the former one allows for an scheduled 
%   variation of the hyper radius.
%   b) in cases were a SP search do not cross lamda zero surface, position  
%   of lamda zero (x_0) is returned with position of the minimum instead of
%   being empy. This aboid crash. Consecuently when calculating the
%   distances from minimum to lambda zero surfaces, all zero distances are
%   squezeed out of the array.
% What is new in version 83_morse:
%
% What is new in version 84_morse:
%   Incorporate the code to minimize the points on the hypersphere
% What is new in version 85_morse:
%   .Incorporate a new stopping criterium in order to compete searching for
%    the reported 27 SPs with E<=1.5 eV.
%   . Updated to matlab R2014b (similar to goust) 
%   . new stopping criterium based on number of connected SPs with E<ref_energy
% What is new in version 86_morse:
%   . update Hyer R only with connected SPs
% What is new in version 86_morse_CLEAN:
%   . all non used features are removed in attempt to speed code. Similar to GOUST.
% What is new in version 87_morse_:
%   . New formula to update HyperR based on the efficiency of the batch
%   . Can be selected minimization  method: LBFGS or CG in iniConfig.m
%   . When reading from database, 10 repetition can be read.
%   . Fixed bug: when no new connected SPs was found
% What is new in version 88_morse_:
%   . Uses davidsson library upon request to get min mode .
%   . new_displacement_per_tune can be set from iniConfig file and works
%   for both, reading from database and not.
%   . Variable 'read_displacement_batch' is now named
%   'disp_by_hypersphere_method'
%   . Saves the variable 'iniConfig.mat' with the actual simulation set up,
%   which is usefull to double check overwriten values in the iniConfig.m
%   file when doing several repetitions.
%   . Stop criterion 'number of connected SPs with E<ref_energy' has been
%   commented to avoid premature stop
% What is new in version 90_morse:
%   . Points on the hypersphere can be selected randomly
%   . uses dis_generator_morseV3.m, which is a bit more efficient 
% What is new in version 90_morse_web:
%   . Simulation will stop if the 27 SPs bellow 1.5 ev are found.
% **************************************

%% load variable structure form iniConfig.m file:
    my_iniFile = 'iniConfig.m'; 
    my_inipath = pwd; % gets path to curent matlab folder(assumtion ini file
                      % is located there. If not hard code the path in this variable)    
    iniVar = get_initial_variables(my_inipath, my_iniFile);    
    
    %do_repetition=true;
    do_repetition = iniVar.do_repetition;
    
    
    if do_repetition

        % SUDO repetition just to repeat the code using different random numbers.
        %n_repetition=10;% number of parameters to be generated
        n_repetition=iniVar.n_repetition;
        values_for_repetition = (1:n_repetition);  
        my_repetition_array = zeros(n_repetition,7);
        
    else
        
        n_repetition = 1;

    end
    
% BEGING LOOP FOR TUNNING:
for repetition=1:n_repetition
    
    % -----Reset random seed
        reset_rnd = false;
        if reset_rnd
            disp('+++IMPORTANT---> Reseting the global random number stream to its initial settings. <---IMPORTANT+++')
            s = RandStream('mt19937ar','Seed',0);
            RandStream.setGlobalStream(s);
        end
    % ----     
    
    folderID=repetition;
    % **** BEGING parameter to be tuned:
    if do_repetition        
        v_tmp = values_for_repetition(repetition);
        disp(' ')
        disp(['Repetition: ', num2str(repetition), '/', num2str(n_repetition), ' -> ', 'parameter value =  ',num2str(v_tmp)])
        disp(' ')
    end
    % **** END parameter to be tuned    
    
%% Declaration (set by user):
    desired_number_sp           = iniVar.desired_number_sp;
    max_attempts                = iniVar.max_attempts;
    RUNTIME                     = iniVar.RUNTIME;
    absTol                      = iniVar.absTol;
    delta                       = iniVar.delta;
    sigma                       = iniVar.sigma;
    disp_by_hypersphere_method  = iniVar.disp_by_hypersphere_method;
    a_to_disp                   = iniVar.a_to_disp;
    pick_rand_atoms             = iniVar.pick_rand_atoms;
    displace_radius             = iniVar.displace_radius;
    get_ini_guess               = iniVar.get_ini_guess;
    bench_energy                = iniVar.bench_energy;
    distribution                = iniVar.distribution;
    hyper_rad                   = iniVar.hyper_rad0;
    self_learning_hyper_rad     = iniVar.self_learning_hyper_rad;% true means the radius will update depending on distances to lambda_zero and found SPs.
    automatic_self_learning     = iniVar.automatic_self_learning;% false means a predifined scheduled for vary hyper R is used.
    self_learning_scheme        = iniVar.self_learning_scheme;
    mix_prev_hyper_rad          = iniVar.mix_prev_hyper_rad;
    read_from_database          = iniVar.read_from_database;
    n_cores                     = iniVar.parall_search_per_min;
    real_n_cores                = iniVar.real_n_cores; % to avoid requesting all available
                                            % CPU in the computer 
    ref_energy                  = iniVar.ref_energy;% this value will be used as a stopping criterium to compete iin the web benchmark.
    new_displacement_per_tune   = iniVar.new_displacement_per_tune;
    sort_displacements          = iniVar.sort_displacements;
    
%% Static Declaration (set by developer):
    reproduce_results = true; % copy source code along with result
    use_scratch=false; % define whether to save in home directory or scratch(local)

%% Dinamic Declaration:
% beging ------ path to save all files -----------------------------------  
    user_name = char(java.lang.System.getProperty('user.name')); % get username to use in folder structure
    f_name = iniVar.f_name;
    project_name = [f_name,'/running_test_', ...
                   num2str(n_cores), 'core/',...
                   num2str(folderID),'repetition/'];
    
    if use_scratch==true
        % SCRATCH DIRECTORY:
        my_paths_ = [...
                    '/scratch/          ';...  
                    'dev/shm/           '];
        disp(['Results will be saved to /scratch/', user_name, '/... '])
        my_paths = cellstr(my_paths_);
        %my_paths{1} = [my_paths{1}, user_name, '/']; 
        my_paths{1} = fullfile(my_paths{1}, user_name, ' ');
    else
        % LOCAL DIRECTORY linux (sol):
        my_paths_ = [...
                    '/SoftSaddle_results/';...  
		    'dev/shm/            '];
        my_paths = cellstr(my_paths_);
        my_paths{1}=fullfile(my_inipath, my_paths{1});
    end
    my_paths{2} = tempname(my_paths{2}); % tmp folder in memory(ramdon/unique name)

% create working folder
    working_folder = fullfile(my_paths{1}, project_name); % folder in HDD
    err_code = system(['mkdir -p ', working_folder]);
    if (err_code ~= 0) 
        disp('Warning! creating working folder..')
    end  

% Modify the paths for saving matlab functions
    folder_name = 'config_to_reprod/'; % folder to save copies of matlab func.
    my_paths_for_func = [working_folder, folder_name];
% end ------ path to save all files --------------------------------------

%% copy source code along with results:    
    if reproduce_results        

        current_fun = mfilename('fullpath'); % gets the name of the current function
        my_list = matlab.codetools.requiredFilesAndProducts(current_fun);
        % creates folder
        err_code = system(['mkdir -p ', my_paths_for_func]);
        if (err_code ~= 0)
            disp('Warning! Problem creating folder for saving matlab functions..')
        end
        % coppy all files
        for i=1:length(my_list)
            origin_f = my_list{i};
            % fullpath omits the .m suffix. exist() still resolves it,
            % so cp must see the suffix on disk.
            if ~isfile(origin_f) && isfile([origin_f '.m'])
                origin_f = [origin_f '.m'];
            end
            err_code = system(['cp ', origin_f, ' ', my_paths_for_func]);
            if (err_code ~= 0)
            disp('Warning! Problem copying matlab functions..')
            end
        end
        % copy iniConfig.m file wih initial variables declaration
        my_ini_config = fullfile(my_inipath, my_iniFile);
        system(['cp ', my_ini_config, ' ', my_paths_for_func]);
       
    end   

%% Initialization
    obj_funct = [];
    x_global_min = [];     
    n_m_vectors_old = 0;
    N_searches = 0; % counter for No of SPs searches
    all_bellow_bench = 0; % counter for all searches that found a SP bellow bench_energy
    m_LS_total_iter_ = 0;% Line Search iterations counter   
    m_CG_total_iter_ = 0;% Conjugate Gradient iterations counter
    sp_LS_total_iter_ = 0;% Line Search iterations counter  
    table_connected = [];% will be overwrited when SP is found
    sp_minimizer_func_calls = 0; % counter
    sp_hess_func_calls = 0; % counter
    sp_func_calls = 0; % counter
    n_ref_energy = 0; % count number of times ref_energy is found
    number_bench = 27;
    c_hyper = 0; % counter for array of distances to x_0
    c_hyper_sp = 0; % counter for array of distances to connected SPs
    n_same_R = 1; % count number of times same R is used.
    n_so_far        = 0; % counter for SPs found in a batch
    n_conn_batch    = 0;
    fsp_global_list = [];
    n_connected_sp  = 0;
    n_ts            = 0;
    unique_sp_sofar = 0;
    sp_global_list= []; 
    
%% Read initial configuration    
    if get_ini_guess
    % BEGING Initial guess and index for zero-forces

        filename = 'displace_bech.mat'; % relative path 
        tmp_a_var = load(filename, '-mat', 'displacement'); % matlab structure
        m2x2_atm = tmp_a_var.displacement;

        remove_atoms = false;
        % BEGIN Removing layer of frozen atoms    
        if remove_atoms
            index_atms = m2x2_atm(:,4)==1; % index of frozzen atoms
            m2x2_atm(index_atms,:)= [];
            N = size(m2x2_atm,1);
        end
        % END Removing layer of frozen atoms

        arrange_atoms = true;
        % BEGIN moving layer of frozen atoms to the end of the array    
        if arrange_atoms
            index_atms = m2x2_atm(:,4)==1; % index of frozzen atoms
            frozzen_atoms = m2x2_atm(index_atms,:);
            index_atms = m2x2_atm(:,4)==0; % index of free atoms
            free_atoms = m2x2_atm(index_atms,:);
            m2x2_atm = [free_atoms; frozzen_atoms]; % rearrenge the array
            %n_frozzen = size(frozzen_atoms,1);
            n_free = size(free_atoms,1);
            N = size(m2x2_atm,1);
        end
        % END moving layer of frozen atoms to the end of the array 
        n = 3*n_free; % number of variable due to free atoms  

        % Initial guess
            x_(1:3:3*N) = m2x2_atm(:,1);
            x_(2:3:3*N) = m2x2_atm(:,2);
            x_(3:3:3*N) = m2x2_atm(:,3);
            x = x_';
        % Clear memory
        clear data textdata %my_file my_path            
        clear x_ m2x2_atm displacement free_atoms frozzen_atoms index_atms
    % END Initial guess and index for zero-forces

    end % get ini_guess   
    
%% Read initial displacements (rather than generate them)
    if disp_by_hypersphere_method==true
        maximum_N_searches= n_cores*max_attempts;
        %read_from_database=false;
        
        if read_from_database
            
            if new_displacement_per_tune
            
                % load displcements from database:
                path_to_my_disp = fullfile('../../source_code/data/disp_on_hypersphere_database/', [num2str(maximum_N_searches), '_points/'], num2str(repetition));% 
                file_name_displacements = 'pos_triang.mat';% has'V', array of all displacements on the unitary hypersphere.
                load(fullfile(path_to_my_disp, file_name_displacements)); 
                disp(['Number of displacements successfuly loaded  = ', num2str(length(V(:,1)))])
                clear Ue_i Ue
            
            else
               
                if repetition==1
                    % load displcements from database:
                    path_to_my_disp = fullfile('../../source_code/data/disp_on_hypersphere_database/', [num2str(maximum_N_searches), '_points/'], num2str(repetition));% 
                    file_name_displacements = 'pos_triang.mat';% has'V', array of all displacements on the unitary hypersphere.
                    load(fullfile(path_to_my_disp, file_name_displacements)); 
                    disp(['Number of displacements successfuly loaded  = ', num2str(length(V(:,1)))])
                    clear Ue_i Ue 
                    V_temporal = V;% I dont remember if I overwrite V bellow to I save it in a temporal variable.
                else
                    disp('Warning! new_displacement_per_tune = false, therefore same initial configurations will be used')
                    V=V_temporal; 
                end                  
                
            end
            
        else % do the minimization on the hypersphere
            
            if new_displacement_per_tune
                % Get V at every tune iteration. Displacements will all be different
                disp(['Wait while ', num2str(maximum_N_searches), ' displacements are generated on the hyperspherical surface..']) 
                time_triangulation = tic;
                V=triangulation_hight_v2(maximum_N_searches, 3*a_to_disp, 2E3, [working_folder, 'disp_on_hypersphere/']); 
                toc(time_triangulation);
                disp(['Number of displacements successfuly generated  = ', num2str(length(V(:,1)))])
            else
                % Get V only once and use it for every tune iteration.
                % Displacements will same:
                % Get V at every tune iteration. Displacements will all be different
                if repetition==1
                    disp(['Wait while ', num2str(maximum_N_searches), ' displacements are generated on the hyperspherical surface..']) 
                    V=triangulation_hight_v2(maximum_N_searches, 3*a_to_disp, 2E3, [working_folder, 'disp_on_hypersphere/']); 
                    disp(['Number of displacements successfuly generated  = ', num2str(length(V(:,1)))])
                    V_temporal = V;% I dont remember if I overwrite V bellow to I save it in a temporal variable.
                else
                    disp('Warning! new_displacement_per_tune = true, therefore same initial configurations will be used')
                    V=V_temporal; 
                end
                
            end
           
        end
        
        if sort_displacements
          % Sort the displacements:
          sorted_disp_index = sort_initial_disp(V); 
        else
            % populate the index list randomly:
            % randsample(n,k) returns a k-by-1 vector of values sampled uniformly at random, without replacement, from the integers 1 to n
            sorted_disp_index = randsample(maximum_N_searches,maximum_N_searches,false);
            disp('Displacements are randomly taken from the hypersphere!')
        end
    end    

%% Pre - allocating memory

    max_array_size = 100000;
    [a_size, ~] = get_opt_array_size(max_array_size, n_cores, max_attempts);

    l = n; % Number of variables due to free atoms 

    m_global_list = NaN*zeros(3*N,a_size);% minima
    fm_global_list = NaN*zeros(a_size,1);% minima
                                                  
    e_sp_ = zeros(a_size,1);
    x_sp_ = zeros(3*N,a_size);
    v_min_ = zeros(l,a_size); % rows are equal to variables due to free to move atoms. 
    sp_CG_total_iter_ = zeros(max_attempts*n_cores,1); % array big enough to keep data only at each minimum individually. 
                                                       % the array is saved and then overwrited in following minima.
    hyper_distances = zeros(a_size, 2); % keeps distances to lambda zero and SPs
    R_efficiency_array = zeros(max_attempts,2);
% ++++++++++++ Check and open the pool if necesary
    mypool=gcp('nocreate');
    if isempty(mypool)

        disp('No open parpool was detected. Attempt to open a parpool for parallel run.')
        if isempty(real_n_cores)
            mypool=parpool;
        else
            mypool=parpool(real_n_cores); 
        end
        
        if strcmp(iniVar.lib_name, '') || strcmp(iniVar.lib_path, '')
            disp('Library name or path not especified. Searching for liblanczos.so in in default matlab directory ')
            iniVar.lib_name = 'liblanczos';
            addAttachedFiles(mypool,{[iniVar.lib_name, '.so'],'matlab.h'})
        else
            disp('Minimum mode will be computed using: ')
            libreria = fullfile(iniVar.lib_path, [iniVar.lib_name, '.so']);
            disp(libreria)
            addAttachedFiles(mypool,{libreria,'matlab.h'})  
            addpath(fullfile(iniVar.lib_path)) % add path so library can be found without especifing the full path
        end
        
        addAttachedFiles(mypool,my_list) % copy all source code to memory
        
    else

        poolsize = mypool.NumWorkers;
        if poolsize<n_cores

            disp(['Matlabpool is open with ', num2str(poolsize), ' workers but more workers requested. Please wait..'])
            delete(mypool);
            if isempty(real_n_cores)
                mypool=parpool(n_cores);
            else
                mypool=parpool(max(n_cores, real_n_cores));
            end
            
            if strcmp(iniVar.lib_name, '') || strcmp(iniVar.lib_path, '')
                disp('Library name or path not especified. Lanczos will be used instead!')
                iniVar.lib_name = 'liblanczos';
                addAttachedFiles(mypool,{[lib_name, '.so'],'matlab.h'})
            else
                dip('Minimum mode will be computed using: ')
                libreria = fullfile(iniVar.lib_path, iniVar.lib_name);
                disp(libreria)
                addAttachedFiles(mypool,{libreria,'matlab.h'})            
            end
            
            addAttachedFiles(mypool,my_list) % copy all source code to memory
        else
           disp(['Matlabpool is open with ', num2str(poolsize), ' workers. SoftSaddle is set up to ', num2str(real_n_cores), ' workers']) 
        end

    end   
% ++++++++++++++++++++++++++++++++++++++++++++++++++                                                       
                                                       
%% Minimize from initial guess:
    reloj = tic; %start the clock

    disp('*******************************************')
    disp(['** Hello ', user_name,'. Simulation started at:   ', datestr(now) ' **'])
    disp('*******************************************')    


% 1) Minimize from initial guess:

    [m_global_list_, ~, ~, fm_global_list_, ~, ...
        m_minimizer_func_calls, m_hess_func_calls, m_func_calls, ~, ~, ~, ~] =...
        MyOwn_v10_morse(x,l,N,false, 0, 0, 0, project_name, my_paths, [], iniVar, []);

%% Chek if first minimization has converged

    if isempty(fm_global_list_)
        disp('=======> BAD LUCK!! Initial minimization NOT converged. Begin again <=======')
    else % do everything

        m_global_list(:,1)  = m_global_list_;   % record first minimum (position)
        fm_global_list(1)   = fm_global_list_;  % record first minimum (energy) 
        current_min         = m_global_list_;   % Position of First minimum 
        current_min_f       = fm_global_list_;  % Energy of First minimum       
        
        % Clear memory
        clear fm_global_list_ % delete temporal variables   

        % initialize
        attempts        = 0;  
        n_sp_found      = 0;
        e_sp_(:)        = NaN;
        x_sp_(:,:)      = NaN;
        v_min_(:,:)     = NaN; 
        run_count_      = e_sp_;
        connected_sp_   = e_sp_; % Colum vector with 0=>NO 1=> YES connected

        % Place Hyper-sphere arround FIRST minimum:
        if disp_by_hypersphere_method==true
            [V_disp_] = place_hypersphere(V, hyper_rad, a_to_disp, current_min); % first hyper-sphere   
            % variable declaration:
            hyper_array = zeros(max_attempts,9); % keep hypersphere statistics
            min_d_sp=0; ave_d_sp=0; max_d_sp=0; min_d_x_0=0; ave_d_x_0=0; max_d_x_0=0;
            % Creates folder to save all distances:
            path_to_save_hyperR = fullfile(working_folder, 'hyperR_data');
            err_code = system(['mkdir -p ', path_to_save_hyperR]);
            if (err_code ~= 0)
                disp('Warning! Problem creating folder for hyper-R data ..')
            end
            
        end

       % repeat SP searches until at least desired_number_sp are found or a
       % max_attempts times.
        while (attempts < max_attempts) && (n_sp_found < desired_number_sp) && (toc(reloj) < RUNTIME) && (n_ref_energy<number_bench)   

            spmd  % n_cores Parallel SP - searches

                run_count_ini = labindex+n_cores*attempts; % for folder structure
                if labindex <= n_cores

                    if disp_by_hypersphere_method == false    
                        % generates displacements
                        displacement_rand = dis_generator_morseV3...
                        (current_min, sigma, pick_rand_atoms, a_to_disp, displace_radius, distribution);
                    else

                        % read displacemnts:
                        displacement_rand = V_disp_(:, sorted_disp_index(run_count_ini));% take diplacements from array loaded previously

                    end

                    % SP-searches from random displacements:
                    [x_sp, ~, ~, e_sp, v_min, m_fc, h_fc, T_fc, run_count, sp_LS_total_iter, sp_CG_total_iter, x_0] = MyOwn_v10_morse...
                        (displacement_rand, l, N,true, 0, 1,...
                        run_count_ini, project_name, my_paths, m_global_list_, iniVar, current_min_f);
                end

            end

            % Read composites from the spmd blok:
            
            % gets index of empty composite fields which means SP not converged  
                i_empt = false(1, n_cores); % index of not converged SPs    
                for a=1:n_cores, i_empt(a)=isempty(e_sp{a});end
            
            % gets folder index    
                run_count_a = cell2mat(run_count(~i_empt)); % succesfull searches

            % (FEV) update function calls list
                sp_minimizer_func_calls = sp_minimizer_func_calls + sum(cell2mat(m_fc(1:n_cores)));
                sp_hess_func_calls = sp_hess_func_calls + sum(cell2mat(h_fc(1:n_cores)));
                sp_fcs = cell2mat(T_fc(1:n_cores)); % array with SP func call for each search in the batch
                sp_func_calls = sp_func_calls + sum(sp_fcs);
                
            % Count Total number of geometry steps:
                sp_LS_total_iter_ = sp_LS_total_iter_ + sum( cell2mat(sp_LS_total_iter(1:n_cores)) );% Line Search
                ini_pos = attempts*n_cores + 1; end_pos = (attempts+1)*n_cores; % to use in array Conjugate Gradient
                sp_CG_total_iter_(ini_pos:end_pos) = cell2mat(sp_CG_total_iter(1:n_cores));% array Conjugate Gradient         
                %sp_CG_total_iter_ = sp_CG_total_iter_ + sum( cell2mat(sp_CG_total_iter(1:n_cores)) );% Conjugate Gradient
                %Total_LS_CG = Total_LS_CG + LS_total_iter_ + CG_total_iter_ ;% Total geometry steps                
            
            % Count SPs before removing not unique;
                e_sp_indicator = cell2mat(e_sp(~i_empt)); % will take non empty fields
                if isrow(e_sp_indicator), e_sp_indicator = e_sp_indicator'; end
                %e_sp_indicator_all(~i_empt) = e_sp_indicator; % empty fields are represented by zero
                all_bellow_bench = all_bellow_bench + sum( e_sp_indicator <= (bench_energy + current_min_f) );
                n_sp = length(e_sp_indicator); % Number of SPs earches converged in batch
                
            if n_sp==0    
                disp(' No SP found in this batch!! ')                
            else % continue     

             % a) ///////////////// Remove repeated SPs (batch)
                converged_batch_xsp = cell2mat(x_sp(~i_empt));
                converged_batch_x_0 = cell2mat(x_0(~i_empt));

                [x_sp_tmp, e_sp_tmp, v_min_tmp, run_count_tmp, x_0_tmp] = ...
                are_vectors_equal_morse_v6 (...
                                    converged_batch_xsp,...
                                    e_sp_indicator,...
                                    absTol,...
                                    cell2mat(v_min(~i_empt)),...
                                    run_count_a,...
                                    converged_batch_x_0);
                clear   all_batch_xsp converged_batch_xsp                                 
    
             % Check agaist previous batches for repetitions:
                [x_sp_tmp_unique, e_sp_tmp_unique, v_min_tmp_unique, run_count_tmp_unique, x_0_unique]= ...
                                    are_batches_equal_morse...
                                (x_sp_tmp, e_sp_tmp, absTol, v_min_tmp, run_count_tmp, x_0_tmp,...
                                x_sp_, e_sp_);    
    

             % Gets NaN index
                ind_nan = isnan(e_sp_tmp_unique);% index of NaN
                n_so_far = sum(~ind_nan); % numbers of UNIQUE SPs in the previus batch
                ini_a = 1 + n_sp_found;
                end_a = n_sp_found + n_so_far;            
               
                if n_so_far ~=0 % This batch discovered new SPs

                    % Check whether all unique in the batch are connected or not:
                    % Minimizations from from all unique SPs along min mode:            
                    % ***************************************************************   
                        [ x_min1, x_min2, energy1, energy2, connected_sp,...
                        m_fc_m1, h_fc_m1, T_fc_m1, m_LS_total_iter1, m_CG_total_iter1, ...
                        m_fc_m2, h_fc_m2, T_fc_m2, m_LS_total_iter2, m_CG_total_iter2] =...
                                      check_if_connected...
                        ( v_min_tmp_unique(:,~ind_nan), x_sp_tmp_unique(:,~ind_nan), delta, absTol, ...
                        current_min, l, N,run_count_tmp_unique, project_name, my_paths, iniVar);
                    % ***************************************************************    
                    % Print number of unique SPs and connected:
                        n_conn_batch = sum(connected_sp); % number of unique and connected SPs in a batch
                        disp(['Connected SPs -> ', num2str(n_conn_batch)])


                    % update FEV list
                        m_minimizer_func_calls = m_minimizer_func_calls + sum(m_fc_m1) + sum(m_fc_m2);
                        m_hess_func_calls = m_hess_func_calls + sum(h_fc_m1) + sum(h_fc_m2);
                        m_func_calls = m_func_calls + sum(T_fc_m1) + sum(T_fc_m2);   

                    % update Total number of geometry steps:
                        m_LS_total_iter_ = m_LS_total_iter_ + sum( m_LS_total_iter1 )+sum( m_LS_total_iter2 );% Line Search  
                        m_CG_total_iter_ = m_CG_total_iter_ + sum( m_CG_total_iter1 )+sum( m_CG_total_iter2 );% Conjugate Gradient            

                    % update global list with minima:
                        e_m = [energy1 energy2]; % energy
                        n_m_vectors = length(e_m);% numbers of non empty composites

                        fm_global_list(1+n_m_vectors_old:(n_m_vectors_old + n_m_vectors)) = e_m; % energy
                        m_global_list (:,(1+n_m_vectors_old:(n_m_vectors_old + n_m_vectors))) =...
                                         [x_min1 x_min2]; % position                        

                    % Validate the global arrays with unique SPs
                        e_sp_(ini_a:end_a)          = e_sp_tmp_unique(~ind_nan);        % energy
                        x_sp_(:, (ini_a:end_a))     = x_sp_tmp_unique(:,~ind_nan);      % Position             
                        v_min_(:, (ini_a:end_a))    = v_min_tmp_unique(:,~ind_nan);     % min mode
                        run_count_(ini_a:end_a)     = run_count_tmp_unique(~ind_nan);   % folder estructure 
                        connected_sp_(ini_a:end_a)  = connected_sp;                     % connected or not

                    % Gets NaN index
                        ind_fin = isfinite(e_sp_(1:end_a));% index of not NaN
                        n_sp_found = sum(ind_fin); % numbers of unique SPs

                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                        % Check for Unique Connected SPs with E<=1.5eV:
                        ind_all_connected_so_far =  connected_sp_==1 ;
                        ind_ref_energy = find( e_sp_(ind_all_connected_so_far)<=(ref_energy + current_min_f) );
                        n_ref_energy = length(ind_ref_energy);% validate one stopping criterium
                        disp(['====>   Number of SPs bellow bench enery = ', num2str(n_ref_energy) '   <===='])
                        if n_ref_energy >= number_bench
                          disp(['====>   The maximum Number of SPs bellow bench enery = ', num2str(n_ref_energy) ' has been found. Simulation will stop.  <===='])
                          % Save the data: To get the SPs bellow bench, use the index (ind_ref_energy) into the global array. 
                        end
                        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                else
                    
                    disp('This batch did NOT discover new SPs ')
                    n_conn_batch = 0;
                    
                end
                
            end

            attempts = attempts +1;

% BEGIN:-:-:-:-:-:-:-:-:-Learning from all searches done so far::-:-:-:-:-:-:-:-:-
            if disp_by_hypersphere_method == true
                
                eff_batch = 0; % efficiency of the batch is set to zero and will be calculated inside the if down below
                my_non_zero = length(find(hyper_distances(:,2)));
                
                if (n_conn_batch == 0) && (my_non_zero == 0) && (automatic_self_learning)% last batch did NOT discover new connected SPs 
                    
                    disp('No new and connected SPs was found. Previus Radius will be used.') 
                    R_efficiency_array(attempts,:) = [hyper_rad eff_batch];
                    hyper_array (attempts,7) = hyper_rad;
                    
                else
                    
                    if self_learning_hyper_rad == true
                        
                        if automatic_self_learning
                           % validate array for data analysis:
                            disp(['Attempt ', num2str(attempts), ' out of ', num2str(max_attempts)])
                            disp(['hyper_R = ', num2str(hyper_rad), ' -> ', num2str(n_so_far), ' batch-unique ',num2str(n_sp_found), ' sofar-unique' ])
                            %hyper_array (attempts,:) = [min_d_sp ave_d_sp max_d_sp min_d_x_0 ave_d_x_0 max_d_x_0 hyper_rad n_so_far n_sp_found]; 

                           % Calculate efficiency of the batch:
                            %eff_batch = n_sofar/(n_cores*n_same_R); % efficiency of the bach based n UNIQUE SPs
                            eff_batch = n_conn_batch/n_cores; % efficiency of the bach based in CONNECTED SPs
                            disp(['Efficiency of the bach  = ', num2str(eff_batch),'  (based CONNECTED SPs) ', ' R = ',num2str(hyper_rad),' has been used so far  ', num2str(n_same_R), ' times'])
                            R_efficiency_array(attempts,:) = [hyper_rad eff_batch];


                            if n_conn_batch ~=0 % This batch discovered new connected SPs
                                % 1. Distances from minimum to lambdazero surface 
                                x_0_array_ = x_0_unique(:,~ind_nan); % unique(connected and NOT)
                                x_0_array  = x_0_array_(:,connected_sp);% unique connected

                                d_x_0_array = sqrt( sum( bsxfun(@minus, x_0_array, current_min).^2 ) );% array of distances minimum to lamda zero                              
                                d_x_0_array(d_x_0_array==0)=[];% sqeeze out zero distances 

                               % 2. Distance from minimum to unique and connected SPs: 
                                x_sp_tmp_unique_ = x_sp_tmp_unique(:,~ind_nan);% unique(connected and NOT)
                                d_sp_array = sqrt( sum( bsxfun(@minus, x_sp_tmp_unique_(:,connected_sp), current_min).^2 ) );% array of distances minimum to SPs                         
                            else
                               d_x_0_array = [];
                               d_sp_array  = [];
                            end
                            %my_non_zero = length(find(hyper_distances(:,2))); % gets the number of non zero d_sp

                            if isempty(d_x_0_array)
                                disp('Not new lambda zero distances')
                            else
                                % Validate the hyper_distances array:
                                s_array = length(d_x_0_array); % number of non zero distances
                                hyper_distances( (1+my_non_zero):(my_non_zero+s_array),1) = d_x_0_array;% distances to x_0
                            end

                            hyper_distances_ = hyper_distances( hyper_distances(:,1)~=0, 1 );% gets all non-zero values
                            min_d_x_0 = min(hyper_distances_);
                            max_d_x_0 = max(hyper_distances_);
                            ave_d_x_0 = mean(hyper_distances_);
                            disp(['min lamda zero= ', num2str(min_d_x_0), ' average lamda zero = ', num2str(ave_d_x_0), ' max lamda zero = ', num2str(max_d_x_0)]);

                            if isempty(d_sp_array)
                                disp('Not new SPs distances')
                            else  
                               % Validate the hyper_distances array:
                                s_array_sp = length(d_sp_array); % number of non zero distances

                                hyper_distances( (1+my_non_zero):(my_non_zero+s_array_sp),2) = d_sp_array;% Distances to connected SPs
                            end

                            hyper_distances_sp = hyper_distances( hyper_distances(:,2)~=0, 2 );% gets all non-zero values
                            min_d_sp = min(hyper_distances_sp);
                            max_d_sp = max(hyper_distances_sp);
                            ave_d_sp = mean(hyper_distances_sp);
                            disp(['min d_sp = ', num2str(min_d_sp), ' average d_sp = ', num2str(ave_d_sp), ' max d_sp = ', num2str(max_d_sp)]);                     

                           % Save d_sp_array, x_0_array for each attempt:
                            my_data_name = [num2str(attempts),'_hyperR_data.mat'];
                            save(fullfile(path_to_save_hyperR, my_data_name), 'd_sp_array', 'x_0_array');

                            %half_d = min_d_x_0 + abs( (min_d_sp-ave_d_x_0)/2);   
                            half_d = min([ave_d_x_0 min_d_sp]);   

                            hyper_array (attempts,:) = [min_d_sp ave_d_sp max_d_sp min_d_x_0 ave_d_x_0 max_d_x_0 hyper_rad n_so_far n_sp_found];

                            % Rewrite hyper_rad (radius of hypersphere)
                            %if automatic_self_learning

                            old_hyper_rad = hyper_rad; %Record previous hyper_rad

                                if self_learning_scheme==1 % use minimum distance to lambda zero
                                    hyper_rad = min_d_x_0; 
                                elseif self_learning_scheme==2 % use average distance to lambda zero
                                    hyper_rad = ave_d_x_0;
                                elseif self_learning_scheme==3 % use maxima distance to lambda zero
                                    hyper_rad = max_d_x_0;
                                elseif self_learning_scheme==4 % use average of minima distances
                                    hyper_rad = (min_d_x_0 + min_d_sp)/2;
                                elseif self_learning_scheme==5 % use average of average distances
                                    hyper_rad = (ave_d_x_0 + ave_d_sp)/2; 
                                elseif self_learning_scheme==6 % use average of maxima distances
                                    hyper_rad = (max_d_x_0 + max_d_sp)/2;
                                else
                                    % experimental formulas here:
                                    %hyper_rad = (hyper_rad + half_d)/2; % 
                                    hyper_rad = min([hyper_rad half_d]);
                                end

                            if mix_prev_hyper_rad
                                hyper_rad = (hyper_rad + old_hyper_rad)/2;
                            end

                            % Decide here whether to use old_hyper_rad or hyper_rad based in efficiency:
                            if eff_batch >= 0.2 % Stay at old_hyper_rad
                                hyper_rad = old_hyper_rad; 
                                n_same_R = n_same_R + 1;
                                %R_efficiecy_array(attempts,:) = [hyper_rad eff_batch];
                            else

                                disp(['Previous hyper R was used ', num2str(n_same_R), ' times.'])
                                n_same_R = 1; % reset counter
                                % valiadte array of R and Probablities:
                                %R_efficiecy_array(attempts,:) = [old_hyper_rad eff_batch];
                                %R_efficiecy_array(attempts,:) = [hyper_rad 1]; % makes efficiency=1 so new hyper_rad is used
                                % Call the function to select R based on its probability:
                                %hyper_rad = get_R(R_efficiecy_array(:,2),R_efficiecy_array(:,1), hyper_rad);
                            end
                            
                            

                        else % automatic_self_learning= false

                                if attempts == 1

                                   disp('automatic_self_learning is OFF. Hyper R varies by user input schedule. ')
                                   % All this need to be done only once:

                                   %min_h_R=0.1; max_h_R=2.3; n_h_R=10; 
                                   %array_hyper_R_ = linspace(min_h_R,max_h_R,n_h_R); 

                                   %array_hyper_R_ = [0.1000 0.3800 0.6600 0.9400 1.2100 1.4900 ...
                                   %     1.7700 2.0500 2.3300 2.6100 2.8900 3.1600  3.4400 3.7200];
                                   %my_schedule= [1 1 3 5 5 5 5 9 5 3 2 2 2 2]; % number of times an hyperR will be used (array with n_h_R elements)

                                    array_hyper_R_ = [0.5 1.0 1.5 2.0 2.5];
                                    %array_hyper_R_ = linspace(0.1, 2.0, 20);

                                    %lo_E=2; % Number of times first R is going to be used
                                    %my_schedule= [lo_E (n_cores*max_attempts-lo_E)];
                                    my_schedule = [10 10 10 10 10]*2;
                                    %my_schedule = ones(1,50)*10;
                                   array_hyper_R = zeros(max_attempts,1); % define at top of code.

                                   ini_ = 1;
                                   for ik=1:length(my_schedule)
                                       fin_ = sum( my_schedule(1:ik) );
                                       array_hyper_R( ini_:fin_ )= repmat(array_hyper_R_(ik),my_schedule(ik),1);
                                       ini_ = fin_ + 1;
                                   end
                                   hyper_rad = array_hyper_R(attempts+1); % Validates hyper R

                                else

                                   ki = attempts+1;
                                   if ki<=max_attempts
                                       hyper_rad = array_hyper_R(ki); % Validates hyper R 
                                   end

                                end 

                        end

                        % rebuild the hypersphere:
                        [V_disp_] = place_hypersphere(V, hyper_rad, a_to_disp, current_min);

                    end
                    
                end

                %R_efficiency_array(attempts,:) = [hyper_rad eff_batch];
                
            end   
% END:-:-:-:-:-:-:-:-:-Learning from all searches done so far::-:-:-:-:-:-:-:-:-        

        end

        N_searches = N_searches + n_cores*attempts; % Number of SPs searches
        disp(['Given attempts in SP =  ', num2str(attempts)]) % error tracking
        disp(['SPs =  ', num2str(n_sp_found), ' in ', num2str(N_searches), ' searches']) % error tracking

        % Check whether any SP has been found or not 
        if n_sp_found == 0    
            disp(' No SP found in any batch. Terribly bad luck!')      
        else % Continue     

            % ------  Delete all NaN entries from the arrays ----
            valid_entries = isfinite(e_sp_);
            connected_sp_(~valid_entries) = [];

            to_squeeze_m = ~isfinite(fm_global_list);
            fm_global_list(to_squeeze_m) = [];
            m_global_list(:,to_squeeze_m) = [];       

            % ------  end Delete all NaN entries from the arrays ----            
            
            
          % Populate the global list 
            fsp_global_list         = e_sp_(valid_entries) - current_min_f;   % delta energy       
            sp_global_list          = x_sp_(:,valid_entries);                       % position
            runcount_global_list    = run_count_(valid_entries);                  % folder estructure
            
            
            % Save the table of SP index and connection info to HDD
            table_connected = [runcount_global_list connected_sp_ fsp_global_list];
            % Save all variables in the workspace
            %working_folder = [my_paths{1}, project_name,'0_0_0/'];
            name_to_table = 'connected.mat';

            % save table in matlab format
            save([working_folder, name_to_table], 'table_connected');   
            % Save table to txt file
            write_to_hdd(table_connected, [my_paths{2}, '/'] , 'connected.con')             
            

        %end
        % BEGIN Data depending on current minimum:
            % print statistics about CG steps:
            % Calculus:
            M = mean(sp_CG_total_iter_); % mean or average value
            S = std(sp_CG_total_iter_); % Standar deviation
            fprintf('%-6s\t', 'min_CG_iter','average_CG_iter', 'max_CG_iter', 'std');
            fprintf('\n');
            fprintf('%3.2f\t',[min(sp_CG_total_iter_) M max(sp_CG_total_iter_) S]);
            fprintf('\n');     
            disp('******=*****')   
            % path of current minimum:
            %path_to_current_min = fullfile(my_paths{2}, '0_0_0/');
            %save(fullfile(path_to_current_min, 'sp_CG_iters.mat'), 'sp_CG_total_iter_')
        % END Data depending on current minimum:

        %i = i+1; 
        unique_sp_sofar = n_sp_found;
        unique_min_sofar = length(fm_global_list);

        n_connected_sp = sum(connected_sp_);
        disp(['SPs = ', num2str(unique_sp_sofar),'( ', num2str(sum(n_connected_sp)), ' )','  min =  ', num2str(unique_min_sofar)  ])   

        % GLOBAL MINIMUM

        global_index = find(fm_global_list==min(fm_global_list)); % index
        obj_funct = fm_global_list(global_index); % energy
        x_global_min = m_global_list(:, global_index); % position

        % Total number of function evaluations
        % total_func_eval = fsp_global_list;
        % Printing a report
        disp(['unique saddle points found = ', num2str(length(fsp_global_list))])
        %disp(['saddle points sampled = ', num2str(length(vis_fsp_global_list))])
        disp(['unique minima found = ', num2str(length(fm_global_list))])
        end
        
    end % end if first minimization has converged    

%% Print out:
    tiempo = toc(reloj);
    disp('^^^ For minimizations: ^^^')
    t1 = ['Number of minimizer function calls:  ', num2str(m_minimizer_func_calls)];
    t2 = ['Number of Hessian function calls:  ', num2str(m_hess_func_calls)];
    t3 = ['Minimizations: Total number of function calls:  ', num2str(m_func_calls)];
    disp(t1);
    disp(t2);
    disp(t3);

    disp('^^^ For SP-searches: ^^^')
    t4 = ['Number of minimizer function calls:  ', num2str(sp_minimizer_func_calls)];
    t5 = ['Number of Hessian function calls:  ', num2str(sp_hess_func_calls)];
    t6 = ['SP-searches: Total number of function calls:  ', num2str(sp_func_calls)];
    disp(t4);
    disp(t5);
    disp(t6);

    disp('^^^ TOTAL (SP-searches+minimizations) ^^^')
    t7 = ['Number of minimizer function calls:  ', num2str(m_minimizer_func_calls+sp_minimizer_func_calls)];
    t8 = ['Number of Hessian function calls:  ', num2str(m_hess_func_calls+sp_hess_func_calls)];
    t9 = ['TOTAL number of function calls:  ', num2str(m_func_calls+sp_func_calls)];
    disp(t7);
    disp(t8);
    disp(t9);

    disp(' ')
    disp('***********  Simulation finished!   **********')
    disp(datestr(now))
    disp(['Elepased time is  ', num2str(tiempo), ' secs'])
    disp('***********  Simulation finished!   **********')
    disp(' ')

    sp_ave_f_call = sp_func_calls/N_searches; % Average function calls        
    T_ave_f_call = (m_func_calls+sp_func_calls)/N_searches; % Average function calls
        
    if (length(fsp_global_list))~=0
        %************************************************
        sp_ave_f_call_connected = sp_func_calls/n_connected_sp; % average function calls per connected SP
        T_ave_f_call_connected = (m_func_calls+sp_func_calls)/n_connected_sp; % average function calls per connected SP
        n_ts = sum( table_connected(logical(connected_sp_),3)<=bench_energy );% gives SPs connected and bellow bench_energy
        n_ts_NOT = sum( table_connected(~logical(connected_sp_),3)<=bench_energy );% gives SPs NOT connected and bellow bench_energy

        %---
        % SPs energies statstics
        energy_sp_connected_ = table_connected(logical(connected_sp_),3); % energy of sp connected
        % round to 4 decimal points
        energy_sp_connected = round(energy_sp_connected_*10000)/10000;
        E_err = 0.001;% tolerance to consider two energy values equal
        for i=1:n_ts
           ind_ex = abs(energy_sp_connected - energy_sp_connected(i)) < E_err; 
           energy_sp_connected( ind_ex ) = energy_sp_connected(i);
        end
        st = E_err;
        values_to_check=min(energy_sp_connected)-st/2:st:max(energy_sp_connected)+st/2;
        N=histc(energy_sp_connected,values_to_check); 
        ValuesOccuringMoreThanOnce=values_to_check(N>=1);
        NumberOfOccurences=N(N>=1);
        energy_table = [ValuesOccuringMoreThanOnce' NumberOfOccurences];
        write_to_hdd(energy_table, working_folder, 'energy_table.con')
        %---

        result_path3 = [working_folder, 'bench_olsen.mat'];
        save(result_path3, 'sp_ave_f_call', 'sp_ave_f_call_connected', 'T_ave_f_call', 'T_ave_f_call_connected', 'n_ts', 'n_ts_NOT',...
        'all_bellow_bench', 'N_searches', 'm_CG_total_iter_', 'm_LS_total_iter_','sp_CG_total_iter_', 'sp_LS_total_iter_', 'unique_sp_sofar', 'n_connected_sp');
        disp('*==========================================================================*')
        disp('*              Displaying benchmark parameters (Olsen et al):              *') 
        disp('*==========================================================================*')

        fprintf('%-6s\t', 'dx_max','dx_ran', 'n_ts', 'v', 'f', 'f_ts', 's', 'v~');
        fprintf('\n');
        %fprintf('%2u\t',delta);
        fprintf('%3.2f\t',...
        [delta sigma n_ts NaN sp_ave_f_call sp_ave_f_call_connected (sp_LS_total_iter_/N_searches) n_ts_NOT]);
        fprintf('\n');    

        % disp(['Average force calls to reach SP (f) ->  ', num2str(sp_ave_f_call)]); 
        % disp(['Average number of force calls needed to reach SPs connected(f_ts) ->  ', num2str(sp_ave_f_call_connected)]);
        % disp(['Total average force calls (min+Sp)(f) ->  ', num2str(T_ave_f_call)]); 
        % disp(['Total average number of force calls needed per SP found connected (min+Sp)(f_ts) ->  ', num2str(T_ave_f_call_connected)]);
        % disp(['No. of SP found within ', num2str(bench_energy), ' eV and connected (n_ts)->  ', num2str(n_ts) ])
        % disp(['No of searches that found one SP within ', num2str(bench_energy), ' eV -> ', num2str(all_bellow_bench)]);
        % disp(['No. of SP found within ', num2str(bench_energy), ' eV and NOT connected (v~)->  ', num2str(n_ts_NOT) ])
        % disp(['Average geometry steps needed to reach SPs(s) ->  ', num2str(sp_LS_total_iter_/N_searches)]);
        disp(['    Average CG geometry steps needed to reach SPs ->  ', num2str(sum(sp_CG_total_iter_)/N_searches)]); 
        disp(['    Average LS geometry steps needed to reach SPs ->  ', num2str(sp_LS_total_iter_/N_searches)]);  
        % disp(['Average geometry steps needed in total(min+sp)(s) ->  ', num2str( (m_LS_total_iter_ + sp_LS_total_iter_)/N_searches)]);
        % disp(['    Average CG geometry steps needed in total (min_sp) ->  ', num2str( (m_CG_total_iter_ + sp_CG_total_iter_)/N_searches)]); 
        % disp(['    Average LS geometry steps needed in total (min_sp) ->  ', num2str( (m_LS_total_iter_ + sp_LS_total_iter_)/N_searches)]);         

        disp('*==========================================================================*')        
    end

%% Save all variables in the workspace:

    save([working_folder, 'statistic.mat'], ...
        'm_minimizer_func_calls', 'm_hess_func_calls', 'm_func_calls', ...
        'sp_minimizer_func_calls', 'sp_hess_func_calls', 'sp_func_calls'...
        ,'tiempo', 'hyper_distances');
    
    if disp_by_hypersphere_method==true    
       save([working_folder, 'hyper_breathing.mat'], 'hyper_array', 'R_efficiency_array');
    end
    
    save([working_folder,'iniConfig.mat'] , 'iniVar'); % saves the actual parmeters used to run the simulation
                                                       % because iniConfig
                                                       % is overwriten when
                                                       % doing several
                                                       % repetitions.
    
    if n_sp_found ~= 0    
        save([working_folder,'wokspace.mat'] , ...
            'obj_funct', 'x_global_min', 'fm_global_list', 'm_global_list', ...
            'fsp_global_list', 'sp_global_list', 'ind_ref_energy',...
            'runcount_global_list' );
        write_out_to_hdd           = iniVar.write_out_to_hdd; %Function 'independent_save_search_result2' needs that results are save to hdd.
                                                              % therefore I
                                                              % call it only if
                                                              % write_out_to_hdd=true

        if write_out_to_hdd
           % Save result report:
            independent_save_search_result2([my_paths{2},'/'], N_searches, n_free)
                    % get standard deviation of func eval:
            %[mean_spsf, std_spsf, all_spsf] = get_standard([my_paths{2},'/'], N_searches);
            [~, ~, ~] = get_standard([my_paths{2},'/'], N_searches);
        end

    end                            
   % ******* Copy everything from memory to HDD
    disp('Copying data from memory to working directory...')
    err_code = system(['cp -r -f ', my_paths{2}, '/* ', working_folder]);
    if (err_code ~= 0) 
        disp('Warning! Failed to copy data to working folder..')
    else
        % *************  delete temporal folder
        % remove tmp folder
        err_code = system(['rm -r -f ', my_paths{2}]);
        if err_code ~= 0
            disp('Warning! failed to remove temporal folder..')
        end
    end
   
    % Print results from repetition loop:
    if do_repetition
    % tune-parameter	f	unique	connected	n_ts	s
        disp('')
        disp('RESULTS FROM REPETITIONS LOOP:')

        fprintf('%-6s\t', 'rep_par', 'f', 'unique', 'connected', 'n_ts','s', 'sp_below1.5eV');
        fprintf('\n');
        %fprintf('%3.2f\t',...
        %    [v_tmp sp_ave_f_call unique_sp_sofar n_connected_sp n_ts (sp_LS_total_iter_/N_searches) n_ref_energy ]);
        %fprintf('\n'); 
        
       % validate an array with results from all repetitions:
        my_repetition_array(repetition,:) = [v_tmp sp_ave_f_call unique_sp_sofar n_connected_sp n_ts (sp_LS_total_iter_/N_searches) n_ref_energy ];
        disp( my_repetition_array(repetition,:));
        disp('')
        % for particular case CG comparison:
        %   my_repetition_array(repetition,:) = [v_tmp unique_sp_sofar n_connected_sp NaN NaN n_ts...
        %     min(all_spsf) mean_spsf max(all_spsf) std_spsf ...
        %     min(sp_CG_total_iter_) M max(sp_CG_total_iter_) S (sp_LS_total_iter_/N_searches)];

    end

%% ++++ Close matlabpool and exit ++++:
         
    disp('All done!!')      
   
end% END LOOP FOR TUNNING  

    % Print and Save results from repetition:
    if do_repetition
        % print array to screen:
        disp(my_repetition_array);
        % save array to hdd:
            % save array in matlab format
            save([working_folder, 'my_repetition_array.mat'], 'my_repetition_array');   
            % Save array in text format
            write_to_hdd(my_repetition_array, working_folder , 'my_repetition_array.con')        
        
    end
    
    disp('closing matlabpool...') 
    delete(mypool);
    disp('MATLAB will be exit')     
    exit    

end % main function

function [V_disp_] = place_hypersphere(V, radius, a_to_disp, min0_)
        
        comp = 3*a_to_disp; % number of componnents of atoms to be displaced
        min0begin = min0_(1:comp); % take componnents of atoms to be displaced
        min0end = min0_(comp+1:end); % take componnents of not to be displaced atoms        
        % expand solution to hypersphere radius = sigma
        V=V*radius;
        disp(' ')
        disp(['^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^hypersphere radius = ', num2str(radius)]); % ERROR tracking
        % Translate back the origen of coordinate system.
        V_disp = V'; % transpose so each column is a displacement vector.
        V_disp = bsxfun(@plus, V_disp, min0begin); % set the previous coordinate system origen back (see function testing_triangulation_hight.m )
        %disp('Run simulation from V_disp to compare')
        % put coordinates of not to be displaced atoms at the end of the array:
        frozz_at = ones(length(min0end), length(V_disp(1,:)) ); % array with colums = n_displacements, rows= n_comp frozzen atoms
        frozz_at = bsxfun(@times, frozz_at, min0end);
        V_disp_ = [V_disp; frozz_at]; % append both array to get the all vectors in full size    
        clear V_disp frozz_at frozz_at  V      
        
end


