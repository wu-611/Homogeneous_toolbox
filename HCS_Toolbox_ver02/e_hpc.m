function u=e_hpc(x, K0, K, Gd,  mu, hn_fun, alpha, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function  u=e_hpc(x, K0, K, Gd, mu, hn_fun) descretizes explicitely
%% Homogeneous Proportional Controller (HPC)     
%%                                                                      
%%     u = K0*x + nx^(1+mu)*K*d(-ln(nx))*x 
%%
%%  where x - system state  and   nx=hn_fun(x) - homogenenous norm of x
%%
%% The input parameters are
%%  x      - (p x 1) state vector
%%  K0     - (m x p) matrix of control gains
%%  K      - (m x p) matrix of control gains
%%  Gd     - (p x p)  the generators of dilation
%%  mu     - homogeneity degree 
%%  hn_fun - function which computes homogeneous norm (e.g., hnorm) 
%%
%% The function  u=e_hpc(x, K0, K, Gd, mu, hn_fun, alpha, beta)                    
%% uses the saturation parameters alpha and beta to lower and upper bound
%% the homogenenous norm nx (i.e., alpha <= nx <= beta) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if K0==0 u=zeros(size(K,1),1);
    else u=K0*x;
end;


if nargin==8 hn=max(alpha,min(beta,hn_fun(x)));
elseif nargin==7 hn=max(alpha,hn_fun(x));
   else hn=hn_fun(x); 
end;


if hn>1e-16 %numerical tolerance
u=u+hn^(1+mu)*K*expm(-log(hn)*Gd)*x;
end;


