%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Example of compuation of homogeneous norm induced by a linear dilation
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

tol=1e-6;                                                                  % compuational tolerance

%%  
%% Shape matrix of the wighted Euclidean norm
%%   
P=[1 -1; -1 2];                                                             

if norm(P-P')>tol disp('Error: shape matrix P must be symmetric'); return; end;

if min(real(eig(P)))<tol disp('Error: shape matrix P must be positive definite'); return; end;

%%  
%% Generator of the linear dilation in R^2
%%   

Gd=[3 0; 1 1]; 
%monotonicy check
if min(real(eig(P*Gd+Gd'*P)))<tol disp('Error: dilation must be monotone'); return; end;

%%  
%% Computation of homogeneous norm of a vector x
%%   

x=[1;-1];                                                                  % vector x in R^2
disp(['Vector x = [', num2str(x(1)),';',num2str(x(2)),']']);

                                                         
disp(['Norm ||x||=sqrt(x^{T}*P*x) =', num2str(sqrt(x'*P*x),'%.8f')]);      % weighted Euclidean norm of x                                                                     


nx=hnorm(x,Gd,P);                                                          % homogeneous norm of x 
disp(['Hom. norm (bisection method) of x = ', num2str(nx,'%.8f')]);        % computed by bisection method


nx=hnorm_newton(x,Gd,P);                                                   % homogeneous norm of x 
disp(['Hom. norm (Newton method) of x = ', num2str(nx,'%.8f')]);           % computed by  Newton method

%%  
%% Computation of homogeneous norm of the scaled vector y=d(ln(2))*x
%%   

y=expm(Gd*log(2))*x;                                                       % scaled vector y=d(ln2)x

disp(['Scaled vector d(ln(2))x = [', num2str(y(1)),';',num2str(y(2)),']']);


disp(['Norm ||d(ln2)x||=',num2str(sqrt(y'*P*y),'%.8f')]);                  % weighted Euclidean norm of y 


ny=hnorm(y,Gd,P);                                                          % homogeneous norm of y 
disp(['Hom. norm (bisection method) of d(ln2)x = ', num2str(ny,'%.8f')]);  % computed by bisection method


ny=hnorm_newton(y,Gd,P);                                                   % homogeneous norm of y 
disp(['Hom. norm (Newton method) of d(ln2)x = ', num2str(ny,'%.8f')]);     % computed by  Newton method


%%  
%% Computation of homogeneous norm of the scaled vector z=2*x
%%   

z=2*x;                                                                     % scaled vector z=2x

disp(['Scaled vector 2x = [', num2str(z(1)),';',num2str(z(2)),']']);


disp(['Norm ||2x||=', num2str(sqrt(z'*P*z),'%.8f')]);                      % weighted Euclidean norm of y 


nz=hnorm(z,Gd,P);                                                          % homogeneous norm of y 
disp(['Hom. norm (bisection method) of 2x = ', num2str(nz,'%.8f')]);       % computed by bisection method


nz=hnorm_newton(z,Gd,P);                                                   % homogeneous norm of y 
disp(['Hom. norm (Newton method) of 2x = ', num2str(nz,'%.8f')]);          % computed by  Newton method