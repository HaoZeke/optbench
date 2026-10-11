function [a_size, t] = get_opt_array_size(max_array_size, n_cores, attempts)
%
% Input:
%   max_array_size -> Maximum number of column to have in a matrix (The
%   number of rows is given by the number of atoms)
%   n_cores -> Number of simultanious SP-searches ( this number has to be
%   less or equal to number of available cpus for the main code to run)
%   attempts -> Number of times to repeat a batch of Sp searches
% Output:
%   a_size -> closest number to max_array_size that can be used for the
%   given n_cores.
%   t -> Number of times to repeat a batch of n_cores searches to fill the
%   entire array.


% max_array_size = 500; % this require 1GB memory in function 
% 'are_vector_equal.m' for vectors with 1024 components  

N_searches = n_cores*attempts;
if N_searches <= max_array_size
    a_size = N_searches;
    t = 1;
    %disp(['optimum array size = ', num2str(a_size)])
else
    %disp('calculating optimum array size...')
    t = floor(max_array_size/n_cores);
    a_size = n_cores*t;
    %disp(['optimum array size = ', num2str(a_size)])
end
