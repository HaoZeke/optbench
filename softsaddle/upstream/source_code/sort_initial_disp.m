function my_index = sort_initial_disp(V)

% Sort initial displacements points acording to largest pair wise distance.
% INPUT: a matrix 'V'[rows, columns] where rows are vectors(points) and
% columns are variables or vector components. (V is m x n)
% OUTPUT: an m x 1 vector 'my_index' having the index of farthest away
% points on V.  For example V(my_index(1),:) and V(my_index(2),:) are the
% two points with largest distance between them.


% load all disp:
%     myPath = '/Users/manuel/Documents/MATLAB/NO_DeleteMe/optima_disp/1000_points/';
%     %myPath = '/Users/manuel/Documents/MATLAB/NO_DeleteMe/optima_disp/120c_points/';
%     myFileName = 'pos_triang.mat';
%     myFile = fullfile(myPath, myFileName);
%     load (myFile, '-mat', 'V' )
%     clear myPath myFileName myFile

    [m, ~] = size(V); % number of points(rows-> n vectors)
    my_index = zeros(m,1); % creates an array lenght: 1 x number of points
   
% Get pair-wise distances:
    pair_wise_dist = pdist(V);
    min_d = min(pair_wise_dist);% only for information printout
    Z = squareform(pair_wise_dist); % square matrix (m x m) with distances
% first two points on list:
    [maxima_along_rows,i_array] = max(Z);
    [max_d,my_index(2)] = max(maxima_along_rows); 
    my_index(1) = i_array(my_index(2)); 

    counter = 3;
for c=1:m-2
    S = sum( Z(my_index(my_index~=0), 1:m) );% dist. between points on the list and the others 
    S(my_index(my_index~=0)) = 0; % zeroing the contribution of already taken points
    [~, next_farther_point]  = max( S );
    my_index(counter) = next_farther_point;
    counter = counter + 1;
end

% checking for repetitions:
t = unique(my_index);
if size(t)~=size(my_index)
    disp('Warning! Sorted list of points has repetitions')
end

disp(['Done sorting initial displacements.. (max. pairwise d=',num2str(max_d), ', min. pairwise d=', num2str(min_d), ')' ])




    


