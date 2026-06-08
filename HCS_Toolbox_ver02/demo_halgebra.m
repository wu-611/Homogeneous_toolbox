%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Example of demonstrating operators in homogeneous Euclidean space
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

Gd=[3 0; 1 1]; 
%monotonicy check
if min(real(eig(P*Gd+Gd'*P)))<tol disp('Error: dilation must be monotone'); return; end;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Sum of vectors in homogeneous Euclidean space
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(['--Addition in homogeneous Euclidean space R^2_d--']);                                                                                                   

x=[1;-1];                                                                  % vector x in R^2
disp(['Vector x = [', num2str(x(1)),';',num2str(x(2)),']']);

y=[-0.5;1];                                                                % vector y in R^2
disp(['Vector y = [', num2str(y(1)),';',num2str(y(2)),']']);

z=x+y;
                                                         
disp(['x+y in R^2 =[', num2str(z(1)),';',num2str(z(2)),']']);              % sum of vectors in R^2                                                                      

z=hadd(x,y,Gd,P);
disp(['x+y in R^2_d =[', num2str(z(1)),';',num2str(z(2)),']']);            % sum of vectors in homogeneous Euclidean space R^2_d                                                                      


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Multiplication of a vector by a scalar in homogeneous Euclidean space
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp(['--Multiplication by a scalar in homogeneous Euclidean space R^2_d--']);                                                                                                  
a=0.7;

disp(['Scalar a= ', num2str(a)]);

x=[1;-5];                                                                  % vector x in R^2
disp(['Vector x = [', num2str(x(1)),';',num2str(x(2)),']']);

z=a*x;
                                                         
disp(['a*x in R^2 =[', num2str(z(1)),';',num2str(z(2)),']']);              % multiplication by a scalar in R^2                                                                      

z=hdot(a,x,Gd);
disp(['a*x in R^2_d =[', num2str(z(1)),';',num2str(z(2)),']']);            % multiplication by a scalar in homogeneous Euclidean space R^2_d                                                                    

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%  
%% Inner product in  in homogeneous Euclidean space
%%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

P=eye(2);                                                                  % for comparison with canonical inner product in R^2

disp(['--Inner product in homogeneous Euclidean space R^2_d--']);                                                                                                  

x=[2;0];                                                                   % vector x in R^2
disp(['Vector x = [', num2str(x(1)),';',num2str(x(2)),']']);

y=[0;1];                                                                   % vector y in R^2
disp(['Vector y = [', num2str(y(1)),';',num2str(y(2)),']']);

p=x'*P*y;
                                                         
disp(['<x,y> in R^2 =', num2str(p)]);                                      % inner product of vectors in R^2                                                                    

p=hinner(x,y,Gd,P);
disp(['<x,y> in R^2_d =', num2str(p)]);                                    % inner product of vectors in homogenous Euclidean space R^2_d                                                                     


