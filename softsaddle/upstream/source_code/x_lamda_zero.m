function [x_0] = x_lamda_zero(xpos, xneg, epos, eneg)

% function x_lamda_zero, computes position of the point on the energy
% surface where the lowest eigenvalue is equal to zero.
% output: x_0 -> n-diemsional vector with position of lamda = 0
% input: xpos -> n-dimensional vector with position of last positive lowest 
%                eigenvalue before it changes sign.
%        xneg -> n-dimensional vector with position of first negative 
%                lowest eiegenvalue.
%        epos and eneg are respective lowest eigen values.


% Distance between xpos and xneg
d_lamda = sqrt( sum( (xpos - xneg).^2) );
% Distance to lamda zero relative to distance between xpos and xneg
d_lamda_zero = epos*d_lamda/(epos - eneg);

v = (xneg - xpos)/norm((xneg - xpos)); % unitary vector
x_0 = xpos + d_lamda_zero*v;

% % save x to file
% %my_name = ['x_zero_', num2str(count_crossing), '.con'];
% my_name = 'x_zero.con';
% write_to_hdd(x_0, working_path, my_name)


end