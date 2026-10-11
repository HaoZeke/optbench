% Script to load data from SoftSaddle run and process it according to the web benchmark.
% Nf = average total number of fev
% er = standar deviation
% Ns = total number of saddle searches
% Nu = number of unique connected saddles
% ro = fraction of saddles found that were connected to the initial minimum
%      for details about the formula see the codes in website: http://optbench.org/saddle-search.html#pt-heptamer-island
%      or the file: ~/Downloads/eon-dimer-lbfgs-5-rotations-20-bowl_breakout/eon_r2213_lbfgs_bowl_breakout_20/eon_r2213_lbfgs_confine_20_1/done.py

% Path to run directory where iniConfig.m is located
my_path     = pwd;

% Path to results
my_path2    = 'SoftSaddle_results/morse/running_test_25core/';


my_file1    = 'bench_olsen.mat';
my_file2    = 'statistic.mat';

n_repetitions     = 10;
all_data    = zeros(n_repetitions,4);

for i=1:n_repetitions

    myguess = [num2str(i), 'repetition'];
    load(fullfile(my_path, my_path2, myguess, my_file1))
    load(fullfile(my_path, my_path2, myguess, my_file2))

    all_data(i,1) = m_minimizer_func_calls + sp_func_calls; %Nf
    NOT_connected = unique_sp_sofar - n_connected_sp;
    all_data(i,2) = 1 - NOT_connected/N_searches;           %ro    
    all_data(i,3) = N_searches;                             %Ns
    all_data(i,4) = n_connected_sp;                         %Nu 


end


disp('')
disp('======================================================')
disp('      Nf          ro          Ns            Nu')
disp('======================================================')
disp(all_data)
disp('======================================================')
disp(mean(all_data))


write_to_hdd([all_data; mean(all_data)], my_path , 'result_27.con')
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Save entries according to web-benchmark:
force_calls = mean(all_data(:,1));
force_calls_mean_error = std(all_data(:,1));
fraction_connected = mean(all_data(:,2));
total_searches = mean(all_data(:,3));
unique_saddles = mean(all_data(:,4));

values = [force_calls force_calls_mean_error fraction_connected total_searches unique_saddles];

entries = {'force_calls', 'force_calls_mean_error', ...
           'fraction_connected', 'total_searches', 'unique_saddles'};

fileID = fopen(fullfile(my_path,'benchmark.dat'),'w');
formatSpec = '%s %2.1f\n';

% Add entries
for i = 1:numel(values)
    C = {entries{i}, values(i)};
    fprintf(fileID,formatSpec, C{:});
end
% extra entries:
extra={...
'code SoftSaddle                       ';...
'date March 2017                       ';...
'contributor Manuel Plasencia Gutierrez'};
[nrows,ncols] = size(extra);
for row = 1:nrows
    fprintf(fileID,'%s\n', extra{row,:});
end
fclose(fileID);


exit(); 

