function [K0, K, Ki, G0, P, mu1, mu2]=fhpic_design(A, B)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function [K0, K, Ki, G0, P]=fhpic_design(A, B) computes parameters                   
%% of Fixed-time Homogeneous Proportional Integral Controller (FHPIC)     
%%                                                                      
%%      u=uh+v,  dv/dt=ui                    <- u=FHPC + integral term  
%%
%%      uh - FHPC     
%%
%%      ui - integrant (as in HPIC) with dependent of x'*P*x (as in FHPC)
%%
%% The input parameters:                                             
%%     A  - system matrix (n x n)                                       
%%     B  - control matrix (n x m)               
%%                                                                      
%% The output parameters:                                                
%%     K0 - feedback gain (m x n) of the linear term of HPIC          
%%     K  - feedback gain (m x n) of the propotional term of HPIC  
%%     Ki - feedback gain (m x n) of the integral term of HPIC 
%%     G0 -  matrix (n x n) defines generators of dilations d1(s) and d2(s) 
%%              Gd1=eye(n)+mu1*G0 and Gd2=eye(n)+mu2*G0                   
%%     P  - positive definite matrix which defines the weighted Euclidean 
%%         norm sqrt(x'*P*x) that induces the canonical homogeneous norm nx  
%%     mu1 - negative homogeneity degree 
%%     mu2 - positive homogeney degree                                                                  
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[K0, K, G0, P,mu1,mu2]=fhpc_design(A,B);
mu1=max(mu1,-0.5);
Ki=-B'*P;

