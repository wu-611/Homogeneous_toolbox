function [K0, K, Ki, Gd, P]=hpic_design(A, B, mu, rho, gamma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function [K0, K, Ki, Gd, P]=hpic_design(A, B, mu) computes                    
%% parameters of Homogeneous Proportional Integral Controller (HPIC)     
%%                                                                      
%%     u=uh+v,  dv/dt=ui                          <- u= HPC + integral term  
%%
%%     uh=K0*x+nx^(1+mu)*K*d(-log(nx))*x          <- HPC 
%%
%%             nx^(1+2*mu)*Ki*d(-log(nx))*x
%%     ui=  ----------------------------------    <- integrant 
%%          x'*d'(-log(nx))*P*Gd*d(-log(nx))*x
%%                                                                      
%% The input parameters:                                             
%%     A  - system matrix (n x n)                                       
%%     B  - control matrix (n x m)               
%%     mu - homogeneity degree (mu=-1 corresponds to sliding mode control)       
%%                                                                      
%% The output parameters:                                                
%%     K0 - feedback gain (m x n) of the linear term of HPIC          
%%     K  - feedback gain (m x n) of the propotional term of HPIC  
%%     Ki - feedback gain (m x n) of the integral term of HPIC 
%%     Gd - anti-Hurwitz matrix (n x n) being a generator of the dilation 
%%                          d(s)=expm(s*Gd),   s - a real number                   
%%     P  - positive definite matrix which defines the weighted Euclidean 
%%         norm sqrt(x'*P*x) that induces the canonical homogeneous norm nx  
%%                                                                      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin==4 [K0, K, Gd, P]=hpc_design(A,B,mu,rho);
elseif nargin==5 [K0, K, Gd, P]=hpc_design(A,B,mu,rho,gamma);
else [K0, K, Gd, P]=hpc_design(A,B,mu);
end;
Ki=-B'*P;

