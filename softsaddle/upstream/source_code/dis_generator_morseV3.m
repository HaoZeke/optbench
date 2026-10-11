function displacement_rand = dis_generator_morseV3(x, sigma,pick_rand_atoms, a_to_disp, displace_radius, distribution)
%% Creates an initial displacement from a given minimum. 
% Variables to be passed:
%
%     1. x -> column vector withall atoms components 
%     x=load ('/Volumes/familia/backup_info/bck_compressed/sol/data/MATLAB/mat_codes/min_mod/MyOwn_general/morse/bench/running_test_12core/10guess/1/0_0_0/min.con');
%
%     2. sigma -> The standard deviation of the Gaussian displacement in
%     each degree of freedom for the selected atoms.
%
%     3. pick_rand_atoms -> If FALSE, each displacement will include all of
%     the degrees of freedom of all of the listed atoms in
%     displace_atomlist. If TRUE one of the atoms in displace_atomlist will
%     be selected at random for each displacement. Then all atoms up to
%     displace_radius distance away from the displaced atom will be
%     included in the displacement.
%
%     4. a_to_disp -> Number of atoms free to be displaced (taken from the
%     top of the list)
%
%     5. displace_radius -> Atoms within this distance of the epicenter
%     will be displaced
%      
%     6. distribution -> shoses between Gaussian or uniform distribution.
%     (see more info inside nested function bellow)
%
% Variables to be returned:
%
%     1. displacement_rand -> displacement column vector with all atoms
%     components

if pick_rand_atoms==true
    
% 1) Displace one atom:
    a = randi(a_to_disp);% Integer between 1 and a_to_disp taken from uniform dist.
    ini_v = (a-1)*3 + 1;
    end_v = ini_v + 2;
    x(ini_v:end_v) = displacement_generator(x(ini_v:end_v), sigma);
    %disp(['Atom to be epicenter and displaced ->  ', num2str(a)])
    %disp(['Epicenter coordinates index ->  ', num2str((ini_v:end_v))])% error tracking
% 2) Calculate distance from epicenter to all other atoms to be displaced
    % Square of the distance between atoms
    distance = ...
    ( x(ini_v) - x(1:3:3*a_to_disp) ).^2 + ...
    ( x(ini_v+1) - x(2:3:3*a_to_disp) ).^2 + ...
    ( x(end_v) - x(3:3:3*a_to_disp) ).^2; 
    %disp(['Distances are =  ', num2str(distance')]) % error track.    
% 3) Compare distances to the given tolerance:
    same =  distance < displace_radius^2;
    same(a) = 0; % to be sure epicenter is not displaced again    
    a_list = (1:a_to_disp); % list of all atoms free to be displaced
    % a_list(same) gives atoms to be displaced
    %disp(['Adicional atoms to be displaced due to close to epicenter -> ', num2str(a_list(same))])    
    ini_v = (a_list(same)-1)*3 + 1;
    L = 3*length(ini_v);
    d_index = zeros(L,1);% Allocate memory for index array
    d_index(1:3:L) = ini_v; % x conmponent of atom to be displaced
    d_index(2:3:L) = ini_v+1; % y conmponent of atom to be displaced
    d_index(3:3:L) = ini_v+2; % z conmponent of atom to be displaced
    %disp(['Index of adicional atoms to be displaced due to close to epicenter ->  ', num2str(d_index')])    
    x(d_index) = displacement_generator(x(d_index), sigma);    
    
else
    
    x(1:3*a_to_disp) = displacement_generator(x(1:3*a_to_disp), sigma, distribution);
    
end
    
    displacement_rand = x; % Validate function

 function displacement = displacement_generator(x_, sigma, distribution)
    % x_             -> components of atoms to be displaced 
    % distribution=1 -> Gausian
    % distribution=2 -> Uniform dist in n-1 subspace (number will be 
    %                   uniformly distributed arround an empty hypersphere 
    %                   with radius sigma)
    % distribution=3 -> Uniform (a number between -1.0 and 1.0 is generated
    %                   from uniform distribution, then multiply by sigma)
    % distribution=4 -> Similar to 2 but relative to each displaced atom
    %                   rather to relative to the system. Uniform dist in 
    %                   n-1 subspace (number will be uniformly distributed 
    %                   arround an empty hypersphere with radius sigma from each atom)
    
    
    ra = randn(size(x_));
    if distribution==1 % Gausian distribution
        
        displacement = ra.*sigma + x_;
        
    elseif distribution==2 %Uniform dist in n-1 subspace
        
        displacement = ra.*sigma;
        %r = sqrt(sum(displacement.^2));
        r = norm(displacement);
        displacement = sigma.*(displacement./r) + x_; % uniform distribution
        
    elseif distribution==3 % Uniform
        
        ini=-1; fin=1;
        r = ini + (fin-ini).*ra;
        displacement = sigma.*r + x_; 
        
    else
        
        displacement = ra.*sigma;

        for i=1:3:3*a_to_disp
            % displaced distance per atoms 
            r = sqrt(sum(displacement(i:i+2).^2));
            % uniform distribution
            displacement(i:i+2) = sigma.*(displacement(i:i+2)./r) + x_(i:i+2); 
        end        
        
    end
    
 end

end
% [x displacement]
