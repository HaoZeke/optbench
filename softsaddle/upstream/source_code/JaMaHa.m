function [direction] = JaMaHa(xmin, xdisp, g, alf, n)

% function JaMaHa, computes a direction to follow in the convex area of the
% potential energy surface. The direction is a mixture between gradient 'g'
% and the vector determined by the position of the minimum with the position 
% of the current point (lets call it vector normal to hypersphere)
%  cases: a) alf = 1 directions = pure vector normal to hypersphere
%         b) alf = 0 directions = pure gradient
% Improved on 12/March/2015:
% smooth mix between gradient vector and vector normal to hypersphere
    
    a_ = (xdisp - xmin)/norm((xdisp - xmin)); % vector normal to hypersphere(unitary vector)
    a = a_(1:n);
    g = g/norm(g); % gradient vector (unitary)

    if isempty(alf), alf=0; end% pure gradient
    if (alf<0), alf=0; end % pure gradient
    if (alf>1), alf=1; end % pure vector normal to hypersphere
            
    % JaMaHa    
    b_eff = (1-alf)*g + alf*a; % mixing grad and vector normal to hypersphere     
    direction = b_eff/norm(b_eff); % direction to follow

end