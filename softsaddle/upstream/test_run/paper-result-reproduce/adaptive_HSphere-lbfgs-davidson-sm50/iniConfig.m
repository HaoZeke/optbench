%================================================
% = 	 SoftSaddle: Improved MMF method        =
% = 	Simulation Initial configuration file	=
% = 	Parameter values to set up simulation:	=
%================================================

% *****************
% *  Simulation:  *
% *****************

    desired_number_sp = 10500; % simulation will stop when desired_number_sp is reached
    parall_search_per_min=25; % Number of simultaneous SP-searches in a batch
    max_attempts = 220; % number of batches
    real_n_cores=25; % Number of computer cores dedicated to the parallel searches (real_n_cores >= parall_search_per_min)

    RUNTIME = 86400*10; % simulation will stop when this time (in seconds) is reached
    absTol = 0.1; % tolerance to concider two geometry configuration equal
    delta = 0.5; % maximum displacement allow in one iteration
    sigma = 0.3; % standard deviation for gaussian
    a_to_disp = 7; % number of atoms in the island, to be displaced 
    pick_rand_atoms = false; 
    displace_radius = 3.3; 
    get_ini_guess = true; % reads initial configuration form file
    bench_energy = 4.0; % a report is printed showing number of atoms within this energy
    ref_energy = 1.5;% this energy will be used as stopping criterium for the web benchmark. 
    distribution = 1; % type of distribution for random displacements

    f_name = 'morse'; % potential name
    write_out_to_hdd = false; % write detailed results to file    
    new_displacement_per_tune = true; % true=> each repetition uses different starting points for SP-searches

% *******************
% *  Hyper sphere:  *
% *******************    
    
    disp_by_hypersphere_method = true; % If true the initial displacements are generated on the hyperspherical surface
    %----------------------------------:
    hyper_rad0 = 0.1; % initial hiper radius
    read_from_database = true; % if true the points evenly distributed on the hypersherical surface are retreived from a database
    self_learning_hyper_rad = true; % true -> uses variable hyper R. flase -> uses constant hyper R = hyper_rad0.
    automatic_self_learning = true;% false means a predifined scheduled for vary hyper R is used.
                                   % true means a distances to sp and lamda zero surfaces are used to vary hype R.
    self_learning_scheme = 2; % different ways to stimate change in hyper R
    mix_prev_hyper_rad = true; % true -> average prev hyper R in the schemes(smoother transition from one hyper R to next one)
    sort_displacements = true; % true means disp are sorted by furthest distances and chosen accordingly
                               % false means the points are chosen ramdomly. 
                                   
% *****************
% *  Minimizer:   *
% *****************

    %min_method = 'CG'; % LBFGS or CG
    min_method = 'LBFGS'; % LBFGS or CG
    memory_size = 60; % for LBFGS method    
    finite_diff_step = 1e-4; % Finite diff to stimate 2nd order derivative 
    cg_iter_max = 1000; % Maximun number of CG iterations     
    ls_iter_max = 1; % Maximum number of Line search iterations
    cg_err = 1e-3; % CG error tolerance 
    ls_err = 1e-2; % Line search error tolerance
    max_step_size = delta ; % delta is defined in globalsadd
    
    RESET = false; % if true CG is reset when lambda zero boundary is crossed. 
    get_min_mode_ = true;
    normalize_direction = true; % if true the search direction is normalized
    maximum_speed = false;  
    dE_max = 480.0; % If E goes beyon this value the saerch is aborted.
    obligate_move = true; % True means the SP-search will be forced to initiate when the initial configuration
			  % is very close to the minimum where the gradient is smaller that the stopping tolerance. 

    use_same_formula = false;% true means using same MMF formula when first exit + area
    alf = []; % if equal zero -gradient is used as direction to escape +area                                        
    n_intentos = 0; % Attempts to force the system staying inside SP basin of attraction.
    avoid_stuck_in_min = true; % True means convergence in area with +eigenvalue will be rejected and 
			       % the system forced to keep searching for SPs.
			       % False means convergence in area with +eigenvalue will be rejected and
			       % the search will stop with a non-converged message.

% *****************************************
% *  Eigenvectors: Lanczos, Davidson, etc *
% *****************************************
lib_name='libdavidson';
lib_path='~/SoftSaddle/libraries/davidsson_lib/28102015_2/';
%lib_name=''; % for lanczos
%lib_path=''; % for lanczos
    lanc_iter = 8; % maximum iterations to compute minimum mode
    lanc_abs_tol = 0.01; % covergence tolerance for eigenvalue
    lanc_finite = finite_diff_step; % finite difference step
    save_lanczos_call = true; % if true, the minimum mode is not calcualted when the step is smaller than a given percent of maximum step size.
    save_lanczos_factor = 50.0; % procent of maximum step size (recomended ~50% for max_step_size=0.5)
    
% ******************
% * Conjugate Grad *
% ******************

    beta_formula=9; % 1->Hestenes and Stiefel, 2->Polak-Ribiere, 3->Liu and Storey
                  % 4->Dai-Yuan, 5->Fletcher-Reeves, 6->conjugate descent
                  % 7->Hestenes-Stiefel + Dai-Yuan, 8->Polak-Ribiere + Fletcher-Reeves, 
                  % 9->Liu-Storey + conjugate descent
    modified_CG = false; % Uses a modified formula to compute search direction.

    e_tol = 1e-8; % tolerance to evaluate uniform descent condition in CG
    get_min_mode_lineSearch = true;

% ******************************************
% *  Repetitions for statistical analysis  *
% ******************************************
do_repetition = true;
n_repetition = 10;


% End
