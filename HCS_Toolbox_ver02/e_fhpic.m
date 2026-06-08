function [uh ui]=e_fhpic(x, K0, K, Ki, G0, mu1, mu2, P, alpha, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function  [uh ui]=fhpic(x, K0, K, Ki, G0, P, mu1, mu2) computes 
%% components of Fixed-time Homogeneous Proportional-Integral Controller (FHPIC)     
%%                                                                      
%%        u = uh + v,  dv/dt=ui   
%%
%%  uh -  FHPC
%%  
%%  ui - - integrant (as in HPIC)  dependent of x'*P*x (as in FHPC)
%%
%% The input parameters are
%%  x      - (p x 1) state vector
%%  K0     - (m x p) matrix of control gains
%%  K      - (m x p) matrix of control gains
%%  Ki      - (m x p) matrix of control gains
%%  G0     - (p x p) matrix which define the generators of dilations 
%%           Gd1 = I + mu * G0   and Gd2 = I +mu * G0
%%  mu1    - local homogeneity degree (close to the origin)
%%  mu2    - local homogeneity degree (close to infinity)
%%  P      - (p x p) matrix such that Gd1*P+P*Gd1'>0, Gd2*P+P*Gd2'>0, P>0 
%%
%% The function [uh ui]=e_fhpic(x, K0, K, Ki, Gd, mu1, mu2, P, alpha, beta)                    
%% uses the saturation parameters alpha and beta to lower/upper bound
%% the canonical homogenenous nx     
%%                                                                      
%%     nx = max(alpha, min(beta,nx)) 
%%                                                                      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if x'*P*x<=1 
    Gd=eye(size(K,2))+mu1*G0;mu=mu1;
else Gd=eye(size(K,2))+mu2*G0;mu=mu2;
end;
hn_fun=@(x)hnorm(x,Gd,P);

if     nargin==10  [uh ui]=e_hpic(x,K0,K,Ki,Gd,mu,hn_fun,alpha,beta);
elseif nargin==9   [uh ui]=e_hpic(x,K0,K,Ki,Gd,mu,hn_fun,alpha);
else               [uh ui]=e_hpic(x,K0,K,Ki,Gd,mu,hn_fun);
end;


