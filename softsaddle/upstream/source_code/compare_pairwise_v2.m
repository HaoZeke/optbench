function [ index ] = compare_pairwise_v2( V_list, absTol )
% ************************************************************************ 
% *  Global Optimization Using Saddle Traversals (GOUST)                 *
% *  Minimum mode following method for Saddle point searches.            *
% *    compare_pairwise_v2 -> gets index of equal vectors in the given   *
% *    array                                                             *
% *                                                                      *
% *      Author: Manuel Plasencia Gutierrez, 2015                        *
% *      email: mpg2@hi.is                                               *
% ************************************************************************


    [n, ~] = size(V_list); 

    %D = pdist(V_list'); % Vector with pair waise distances
    %sD = squareform(D); % squuare matrix with pair wise distances
    sD = squareform(pdist(V_list));
    %disp('Original square matrix of distances: ')
    %disp(sD)
    %disp(' ')
    
    sD( tril(true(n)) ) = Inf; % makes lower diagonal elements equal infinite
    %disp('transformed square matrix of distances: ')
    %disp(sD)
    %disp(' ')
    
    [r,c] = find(sD<=absTol);
    index = [r c];

% %
% tic;
% [r, c] = size(vv');
% distanceMatrix = zeros(r);
% distanceMatrix(tril(true(r),-1)) = D;
% %distanceMatrix = distanceMatrix + distanceMatrix'
% toc;
% disp(distanceMatrix)
% disp('------------------------------------------')
% disp(['size od D = ', num2str(size(D))])
% toc;
% disp('done without crashing!!!')
% disp(' ')
% disp('+++++++++++++++++++++++++++++++++++++++++++')
% disp(' ')


end

