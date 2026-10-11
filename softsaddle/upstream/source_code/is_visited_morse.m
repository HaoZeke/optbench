function [visited]= is_visited_morse(vector, vector_list, absTol)
            
% "vector_list" is a list in a matrix form, where colums are vectors to be
% compared for similarity with "vector"
% vector is a column vector same size as each column in "vector_list"
% visited = true => vector is already in vector_list 
% visited = false => vector is NOT  in vector_list 
% 

    [r, ~] = size(vector_list); % get array dimensions c = number of min/sp
    
if ~isvector(vector)  % if not a vector 
  %raise an error and stop    
    err = MException('ResultChk:OutOfRange', ...
        'expecting a vector and matrix received');
    throw(err)
    % Assumes is empty becasue minimization did not converged
    
elseif length(vector)==r % No problem, do everything
    
    %absError =  bsxfun(@minus, vector_list, vector);
    diff_comp =  bsxfun(@minus, vector_list, vector);
    
    
%     % 2.) Square of the difference in x, y and z between every atom and its corrsponding 
%     X = ( diff_comp((1:3:r),:) ).^2;
%     Y = ( diff_comp((2:3:r),:) ).^2;
%     Z = ( diff_comp((3:3:r),:) ).^2;
%     distance = X + Y + Z; % Square of the distance between atoms
    
    % Square of the distance between atoms
    distance =  ( diff_comp((1:3:r),:) ).^2 + ...
                ( diff_comp((2:3:r),:) ).^2 + ...
                ( diff_comp((3:3:r),:) ).^2 ; 
    
    
    %relError(~isfinite(relError)) = 0;   % Sets Inf and NaN to 0
    same = all( distance < absTol^2);

    if sum(same)==0 % not visited
        visited = false;
        % disp('Has NOT been visited')
    else
        visited = true;
        % disp('Has been visited')
    end

else % is vector but different sise than expected: error and stop
    
    err = MException('ResultChk:OutOfRange', ...
        'Wrong vector length: expecting colum vector');
    throw(err)
    
end
