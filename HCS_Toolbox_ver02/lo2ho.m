function [L0, G0, nu_min, nu_max]=lo2ho(A,C,L)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function [L0, G0, nu_min, nu_max]=lpc2hpc(A,B,K) upgrades 
%% the linear observer (LO) 
%%
%%    dz/dt=A*z+f+L*(C*z-y) 
%%
%% to Homogeneous Observer (HO):     
%%                                                                      
%%   dz/dt=A*z+f+(L0+|Cz-y|^(nu-1)*d(log(|Cz-y|))*(L-L0))*(Cz-y)            
%%                                                                       
%%
%% The input parameters:                                             
%%      A  - system matrix (n x n)                                       
%%      C  - outout matrix (k x n)       
%%      L  - gain matrix (n x k) of LO s.t. A+L*C is Hurwitz  
%%
%% The output parameters:                                             
%%      L0 - gain (m x n) of linear component of HPC                                          
%%      G0 - matrix (n x n) that  defines the generator Gd=eye(n)+nu*G0  
%%           of the dilation d(s)=expm(s*Gd)                     
%%  nu_min - minimal admissible degree of HO (nu_min<=nu)  
%%  nu_max - maximal admissible degree of HO  (nu<=nu_max)
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L0=0;  G0=0; nu_min=0;nu_max=0; 
tol=0.00001; 

sA=size(A); sC=size(C);

if sA(1)~=sA(2) disp('Error: the system matrix must be square'); return;
end;
if sC(2)~=sA(2) disp('Error:  dimensions of system  and output matrices must agree');return;
end;

[L0, G0, P, nu_max, nu_min]=lpc2hpc(A',C',L');
L0=L0'; G0=-G0'; nu_min=-nu_min;nu_max=-nu_max;
