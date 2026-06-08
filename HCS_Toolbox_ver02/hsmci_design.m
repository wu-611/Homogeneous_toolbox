function [K0, K, Ki, Gd, P]=hsmci_design(A, B, C, rho, gamma)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function [K0, K, Gd, P]=hsmci_design(A, B, C) computes parameters of           
%% Homogeneous Sliding Mode Control with Integral action (HSMCI)         
%%                                                                      
%%     u = K0*x + sqrt(ncx)*K*d(-log(hcx))*C*x + v   (HSMC + integral term)
%%                                          (ncx- homogenenous norm of C*x) 
%%
%%                    Ki*d(-log(hcx))*C*x
%%     dv/dt= --------------------------------------                          
%%            x'*C'*d'(-log(hcx))*P*d(-log(hcx))*C*x                                             
%%
%% which enforces a sliding mode on Cx=0 for the linear MIMO system     
%%                                                                      
%%     dx/dt = A*x + B*u,   x(0) = x0                                   
%%                                                                      
%% in a finite time: Cx(t)=0  for all t>=T0 (settling time dependent of x0)                 
%%                                                                      
%% The input parameters are                                             
%%     A  - system matrix (n x n)                                       
%%     B  - control matrix (n x m)                                      
%%     C -  sliding surface matrix (p x n)                              
%%                                                                      
%% The output parameters                                                
%%    K0 - feedback gain of a linear part of the controller             
%%    K  - feedback gain of the nonlinear part of the controller        
%%    Gd - generator (p x p) of the dilation d(s)=expm(s*Gd)            
%%    P  - positive definite matrix (p x p) which defines the canonical 
%%         homogeneous norm h_norm(v) induced by the norm sqrt(v'*P*v)  
%%                                                                      
%% Remark: HSMCI becomes Super-Twisting 2-SMC for n=m=1                                                                   
%%                                                                      
%%                                                                      
%% The function [K0, K, Ki, Gd, P]=hsmci_design(A, B, C, rho)  uses          
%% the additional parameter rho for tuning of the reaching time
%% (larger rho, faster the convergence)
%%  
%% The function [K0, K, Gd, P]=hsmci_design(A, B, C, rho, gamma)  uses          
%% uses additional parameter gamma to deal with matched perturbations
%% (with a known bound gamma):  
%%                                                                      
%%   p(t,x)=Bg(t,x) :  norm(g(t,x))<=gamma     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
K0=0; K=0; Ki=0; Gd=0; P=0; 
[tK, As, Bs, err]=output_form(A,B,C);
if err>0 return; end;
if nargin==3 
[K0 K Ki Gd P]=hpic_design(As,Bs,-0.5); end;
if nargin==4
    if size(rho,1)+size(rho,2)~=2 disp('Error: the parameter rho must be non-negative real number'); return; end;
    [K0 K Ki Gd P]=hpic_design(As,Bs,-0.5,rho);     
end;
if nargin>=5
      if size(rho,1)+size(rho,2)~=2 disp('Error: the parameter gamma must be non-negative real number'); return; end;
      [K0 K Ki Gd P]=hpic_design(As,Bs,-0.5,rho,gamma); 
    end;

K0=tK+K0*C;


