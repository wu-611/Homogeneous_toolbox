function [L0, L, Gd, P]=ho_design(A, C, nu, rho)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function [L0, L, Gd, P]=ho_design(A, C, nu)                     
%% computes parameters of Homogeneous Observer (HO)     
%%                                                                      
%%    dz/dt=A*z+f+(L0+|Cz-y|^(nu-1)*d(log(|Cz-y|))*L)*(Cz-y)         
%%                                                                      
%% for  the linear MIMO system                             
%%                                                                      
%%     dx/dt = A*x+f,   y = C*x            (f is a known input signal)                       
%%     
%%  such that the error dynamics err=z -x  is
%%
%% (a)  a finite time stable for nu<0                  
%%                                                                      
%% (b) a nearly fixed time istable for nu>0
%%
%% The input parameters:                                             
%%     A  - system matrix (n x n)                                       
%%     C  - output matrix (k x n)               
%%     nu - homogeneity degree      
%%                                                                      
%% The output parameters:                                                
%%     L0 - gain (n x k) of the linear term of Ho          
%%     L  - gain (n x k) of the nonlinear term of HO       
%%     Gd - anti-Hurwitz matrix (n x n) being a generator of the dilation 
%%                          d(s)=expm(s*Gd),   s - a real number                   
%%                                                                      
%%                                                                      
%%                                                                      
%% The function [K0, K, Gd, P]=hpc_design(A, B, mu, rho)                
%% uses aditional input parameter rho for tuning  of convergence time:  
%% the larger rho, the faster convergence                             
%% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L0=zeros(size(C')); L=zeros(size(C')); Gd=zeros(size(A)); P=zeros(size(A));
tol=0.00001; 

sA=size(A); sC=size(C);

if sA(1)~=sA(2) disp('Error: the system matrix must be square'); return;
end;
if sC(2)~=sA(2) disp('Error:  dimensions of system  and output matrices must agree');return;
end;

if nargin==4 [L0, L, Gd, P]=hpc_design(A',C',-nu,rho);
else [L0, L, Gd, P]=hpc_design(A',C',-nu);
end;
L0=L0';L=L';Gd=Gd';