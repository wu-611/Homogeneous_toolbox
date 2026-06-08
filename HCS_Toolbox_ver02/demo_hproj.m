%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Example of compuation of homogeneous projection
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tol=1e-6;                                                                  % compuational tolerance

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Shape matrix of the wighted Euclidean norm
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
P=[1 -1; -1 2];                                                             

if norm(P-P')>tol disp('Error: shape matrix P must be symmetric'); return; end;

if min(real(eig(P)))<tol disp('Error: shape matrix P must be positive definite'); return; end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Generator of the linear dilation in R^2
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

Gd=[5 0; 1 3]; 

%monotonicy check
if min(real(eig(P*Gd+Gd'*P)))<tol disp('Error: dilation must be monotone'); return; end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Computation of homogeneous projection 
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

x=[0.5;-1.5];                                                                  % vector x in R^2
disp(['Vector = [', num2str(x(1)),';',num2str(x(2)),']']);



z=hproj(x,Gd,@(x)hnorm(x,Gd,P));                                                           % homogeneous projection of x 
                                                                           % on the unit sphere x'*P*x=1
                                                                           % computed by bisection method
disp(['Hom. projection (bisection method)= [', num2str(z(1),'%.8f'),';',...
    num2str(z(2),'%.8f'),']']);


z=hproj(x,Gd,@(x)hnorm_newton(x,Gd,P));                                             % homogeneous projection of x 
                                                                           % on the unit sphere x'*P*x=1
                                                                           % computed by Newton method
disp(['Hom. projection (Newton method)= [', num2str(z(1),'%.8f'),';',...
    num2str(z(2),'%.8f'),']']);








