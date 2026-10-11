function [ index ] = compare_V1andV2_pairwise( V_list1, V_list2, absTol )
% ************************************************************************ 
% *               Improved MMF method (GLOBALSADD)                       *
% *  Minimum mode following method for Saddle point searches.            *
% *    compare_V1andV2_pairwise -> Compares pairwise between two sets of *
% *    observations.                                                     *
% * Input: V_list1 and V_list2 can be vector or matrix arrays.           *        
% * Rows of V_list1 and V_list2 correspond to observations, columns      *
% * correspond to variables. They must have same number of columns.      *
% *                                                                      *
% * Output: index is n by 2 array with index of repeated observation     *
% *         between V_list1 and V_list2. Example:                        *
% *           index = [2 6                                               *
% *                    4 3]                                              *
% * means observation 2 in V_list1 is equal to observation 6 in V_list2  * 
% * and observation 4 in V_list1 is equal to observation 3 in V_list2    *
% *                                                                      *
% *                                                                      *
% *      Author: Manuel Plasencia Gutierrez, 2015                        *
% *      email: mpg2@hi.is                                               *
% ************************************************************************


    D = pdist2(V_list1, V_list2); % Vector with pair waise distances    
    [r,c] = find(D<=absTol);
    index = [r c];


end

