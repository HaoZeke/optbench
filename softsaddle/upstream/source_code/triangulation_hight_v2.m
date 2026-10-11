function [V] = triangulation_hight_v2(N, comp, Nitr, saving_path)

% Declaration:
    s=1; Etol=1E-6; %Nitr=2E3; 
    % N -> number of particles to arrange
    R =1; % radius
    %n_free_a = 175; % -> (56_atoms *3_layers + 7_atoms)*3_components = 525 variables
    %a_to_displace=7; % n of atoms to be displaced
    %comp = 3*a_to_displace; % number of atom componnents
    

% load 1500 displacements to use as initial guess:

    
    % create saving directory:
    system(['mkdir -p ', saving_path]);
    %min0_ = load([disp1500_path, '0_0_0/min.con']); % load the minimum
    %min0 = min0_(1:comp); % take only componnents of atoms free to move (as for al disp vectors)
    %load([disp1500_path, 'all_disp.mat']) % Load 1500 displacement
    
    fileID = fopen([saving_path, 'log.out'],'w'); 
    fprintf(fileID, 'Number of particles = ');
    fprintf(fileID,'%2u\t',N);
    fprintf(fileID,'\n');
    
% Translate to origen of coordinate system.
    %V_ = bsxfun(@minus, all_disp((1:comp),(1:N)), min0); % set minimum in coordinate origen
    
% Get random initial points:
%----------------------------------
        sigma=1;
        displacement = randn(comp,N).*sigma;
        r = sqrt(sum(displacement.^2));
        V_= zeros(size(displacement));
        for o=1:N
            V_(:,o) = sigma.*(displacement(:,o)./r(o)) ; % uniform distribution
        end
%----------------------------------
    
% normalize:
    V_L2=sqrt(sum(V_.^2));  
    V_=bsxfun(@rdivide,V_,V_L2);
    clear V_L2
    
% Traspose for compatibility
    V = R.*V_';
    
% =*=*=*=**=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=

% Compute geodesic distances between the points
    DOT=V*V';       % dot product 
    DOT(DOT<-1)=-1; DOT(DOT>1)=1;
    GD=acos(DOT);   % geodesic distance

% Evaluate the energy functional 
    GD(1:(N+1):end)=Inf; % set diagonal entries of GD to Inf
    Ue_ij=1./((GD.^s)+eps);
    Ue_i=sum(Ue_ij,2);
    Ue=sum(Ue_i);

%% Iteratively optimize the position of the particles along the negative 
% gradient of the energy functional using an adaptive Gauss-Seidel 
% update scheme -----------------------------------------------------------

    fprintf(fileID, 'Particle positions are being optimized ...\n');
    t0=clock;
    i=0; 

    a=ones(N,1);    % step sizes used during position updates
    a_min=1E-14;    % minimum step size
    a_max=0.2;      % maximum step size
    dE=Inf;

    while i<Nitr && dE>Etol

        i=i+1;

        % Sort the particles according to their energy contribution
        [~,idx_sort]=sort(Ue_i,'descend');

        % Update the position of individual particles
        for k=1:N

            j=idx_sort(k);

            idx_j=true(N,1); % particle indices, except the current one 
            idx_j(j)=false;

            DOTj=DOT(idx_j,j); 
            GDj=GD(idx_j,j); 

            % Compute the gradient for the j-th particle
            dVj=bsxfun(@times,s./(sqrt(1-DOTj.^2)+eps),V(idx_j,:));                
            dVj=bsxfun(@rdivide,dVj,(GDj.^(s+1))+eps);
            if i<5 % remove contributions from particles which are too close
                idx_fail=sum(dVj.^2,2)>(1E4)^2;
                dVj(idx_fail,:)=[];
                if isempty(dVj) % perturb initial positions
                    V=V+randn(size(V))/1E5;
                    V_L2=sqrt(sum(V.^2,2));
                    V=R.*bsxfun(@rdivide,V,V_L2);
                    DOT=V*V';
                    GD=acos(DOT);
                    break
                end 
            end
            dVj=sum(dVj,1);

            % Only retain the tangential component of the gradient
            dVj_n=(dVj*V(j,:)')*V(j,:);
            dVj_t=dVj-dVj_n;        

            % Adaptively update position of the j-th particle
            m=0; 
            Uj_old=sum(Ue_ij(j,:));
            while true

                % Update the position
                Vj_new=V(j,:)-a(j)*dVj_t;

                % Constrain the point to surface of the sphere
                Vj_new=R.*(Vj_new/norm(Vj_new));

                % Recompute the dot products and the geodesics
                DOTj=sum(bsxfun(@times,V,Vj_new),2);
                DOTj(DOTj<-1)=-1; DOTj(DOTj>1)=1;
                GDj=acos(DOTj);
                GDj(j)=Inf;

                Ue_ij_j=1./((GDj.^s)+eps);

                % Check if the system potential decreased
                if sum(Ue_ij_j)<=Uj_old

                    V(j,:)=Vj_new;

                    DOT(j,:)=DOTj';
                    DOT(:,j)=DOTj;

                    GD(j,:)=GDj';
                    GD(:,j)=GDj;

                    Ue_ij(j,:)=Ue_ij_j';
                    Ue_ij(:,j)=Ue_ij_j;

                    if m==1, a(j)=a(j)*2.5; end
                    if a(j)>a_max, a(j)=a_max; end

                    break

                else

                    if a(j)>a_min
                        a(j)=a(j)/2.5;
                        if a(j)<a_min, a(j)=a_min; end
                    else
                        break
                    end

                end

            end

        end

        % Evaluate the total energy of the system 
        Ue_i=sum(Ue_ij,2);
        Ue(i+1)=sum(Ue_i);

        % Progress update
        if i==1
            fprintf(fileID,'Iteration#\tEnergy Score\n');
            fprintf(fileID,'%u\t        %.3f\t\n',0,Ue(1));
        else
            if mod(i,50)==0
            fprintf(fileID,'%u\t        %.3f\t\n',i,Ue(end));
            end
        end

        % Change in energy
        if i>=10, dE=(Ue(i-2)-Ue(i+1))/10; end

        % Reset the step sizes
        if mod(i,20)==0, a=a_max*ones(N,1); end

    end

    clear DOT GD Ue_ij
    if Nitr==0, fprintf(fileID,'%u\t        %.3f\t\n',i,Ue(1)); end
    if mod(i,50)~=0, fprintf(fileID,'%u\t        %.3f\t\n',i,Ue(end)); end
    fprintf(fileID,'Optimization completed after %u iterations. Elapsed time: %5.1f sec\n',i,etime(clock,t0));
%% look excell file (quality meassurements)

    r = norm( sum(V) ); % should be close to zero(see wikipedia)
    nearest_1 = min(pdist(V));
    a = acosd( (1 + 1 - nearest_1^2)/2 );% cosine law to get the angle given 3sides of triangle
    save([saving_path, 'quality.mat'], 'r', 'nearest_1','a' )
% => ** 
    fprintf(fileID, '|sum(ri)| = %.3f\t\n', r);
    %disp(['|sum(ri)| = ', num2str(r)])
    fprintf(fileID, '1-nearest = %.3f\t\n', nearest_1);
    %disp(['1-nearest = ', num2str(nearest_1)])
    fprintf(fileID, 'alfa = %.3f\t\n', a);
    %disp(['alfa = ', num2str(a)])

% =*=*=*=**=*=*=*=*=*=*=*=*=*=*=*=*=*==*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=
    save([saving_path, 'pos_triang.mat'], 'V', 'Ue_i','Ue' )
    fprintf(fileID, ['pos_triang.mat -> saved to: ', saving_path]);
    %disp ('pos_triang.mat -> saved successfuly to:') 
    %disp(saving_path)
    fclose(fileID);
 %disp('All done!!')

    
