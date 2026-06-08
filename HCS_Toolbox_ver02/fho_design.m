function [L0, L, G0, nu1, nu2]=fho_design(A, C)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function [L0, L, G0, nu1, nu2]=ho_design(A, C)                     
%% computes parameters of Fixed-time Homogeneous Observer (FHO)     
%%                                                                      
%%                
%%               /(L0+|Cz-y|^(nu1-1)*d1(log(|Cz-y|))*L)*(Cz-y) if |Cz-y|<=1
%%  dz/dt=A*z+f+<         
%%               \(L0+|Cz-y|^(nu2-1)*d2(log(|Cz-y|))*L)*(Cz-y) if |Cz-y|>=1
%%                 
%%                                                                      
%% for  the linear MIMO system                             
%%                                                                      
%%     dx/dt = A*x + f,   y = C*x            (f is a known input signal)                           
%%     
%%  such that the error dynamics err=z-x  is fixed-time stable
%%                 
%%                                                                      
%% The input parameters:                                             
%%     A  - system matrix (n x n)                                       
%%     C  - output matrix (k x n)               
%%     nu - homogeneity degree      
%%                                                                      
%% The output parameters:                                                
%%     L0 - gain (n x k) of a linear term of FHO          
%%     L  - gain (n x k) of a nonlinear term of FHO       
%%     G0 - matrix (n x n) defines generators of dilations d1(s) and d2(s) 
%%              Gd1=eye(n)+nu1*G0 and Gd2=eye(n)+nu2*G0             
%%     nu1 - minimal admissible negative homogeneity degree 
%%     nu2 - maximal admissible positive homogeney degree                                                                                                
%%                                                                      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
L0=0; L=0; Gd=0; 
tol=0.00001; 

sA=size(A); sC=size(C);

if sA(1)~=sA(2) disp('Error: the system matrix must be square'); return;
end;
if sC(2)~=sA(2) disp('Error:  dimensions of system  and output matrices must agree');return;
end;

[L0, L, G0, P, nu2, nu1]=fhpc_design(A',C');
L0=L0';L=L'; G0=-G0'; nu1=-nu1;nu2=-nu2;