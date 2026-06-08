function u=si_fhpc(h, x, A, B, K0, K, G0, mu1, mu2, P, alpha, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function u=si_fhpc(h, x, A, B, K0, K, G0, mu1, mu2, P) descretizes
%% semi-implicitly Fixed-time Homogeneous Proportional Controller (FHCP)     
%%                                                                      
%%          _ 
%%         /
%%        /  K0*x+nx^(1+mu1)*K*d1(-log(nx))*x       if nx<=1
%%   u = < 
%%        \  K0*x+nx^(1+mu2)*K*d2(-log(nx))*x       if nx>1
%%         \_  
%%
%%  where x - system state  and   nx - homogenenous norm of x

%%
%% The input parameters are
%%  h      - sampling period
%%  x      - (p x 1) state vector
%%  A      - (p x p) system matrix
%%  B      - (p x m) control matrix
%%  K0     - (m x p) matrix of control gains
%%  K      - (m x p) matrix of control gains
%%  G0     - (p x p) matrix which define the generators of dilations 
%%           Gd1 = I + mu * G0   and Gd2 = I +mu * G0
%%  mu1    - local homogeneity degree (close to the origin)
%%  mu2    - local homogeneity degree (close to infinity)
%%  P      - (p x p) matrix such that Gd1*P+P*Gd1'>0, Gd2*P+P*Gd2'>0, P>0 
%%
%% The function u=si_fhpc(h, x, A, B, K0, K, G0, mu1, mu2, P, alpha, beta)                    
%% uses the saturation parameters alpha, beta to lower/upper bound
%% the homogenenous norm nx     
%%                                                                      
%%     nx = max(alpha, min(beta,nx)) 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if x'*P*x<=1 Gd=eye(size(K,2))+mu1*G0;mu=mu1;
else Gd=eye(size(K,2))+mu2*G0;mu=mu2;
end;
hn_fun=@(x)hnorm(x,Gd,P);
if nargin==12       u=si_hpc(h,x,A,B,K0,K,Gd,mu,hn_fun,alpha,beta);
elseif nargin==11 u=si_hpc(h,x,A,B,K0,K,Gd,mu,hn_fun,alpha);
else                u=si_hpc(h,x,A,B,K0,K,Gd,mu,hn_fun);
end;


