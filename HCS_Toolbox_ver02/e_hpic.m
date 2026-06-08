function [uh ui]=e_hpic(x, K0, K, Ki, Gd, mu, hn_fun, alpha, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function  [uh ui]=e_hpic(x, K0, K, Ki, Gd, mu, hn_fun) computes 
%% components of the Homogeneous Proportional-Integral Controller (HPIC)     
%%                                                                      
%%        u = uh + v,  dv/dt=ui   
%%
%%  uh=K0*x + nx^(1+mu)*K*d(-log(nx))*x - homogenenous component
%%  
%%  ui=nx^(1+2*mu)*Ki*d(-log(nx))*x
%%
%%  where x - a system state, nx=hn_fun(x) - the homogenenous norm of x   
%%
%% The input parameters are
%%  x      - (p x 1) state vector
%%  K0     - (m x p) matrix of control gains
%%  K      - (m x p) matrix of control gains
%%  Ki     - (m x p) matrix of control gains
%%  Gd     - (p x p)  the generators of dilation
%%  mu     - homogeneity degree 
%%  hn_fun - function which computes homogeneous norm (e.g., hnorm) 
%%
%% The function [uh ui]=e_hpic(x, K0, K, Ki, Gd, mu, hn_fun, alpha, beta)                    
%% uses the saturation parameters alpha and beta to lower/upper bound
%% the canonical homogenenous norm nx=hn_fun(x)     
%%                                                                      
%%     nx = max(alpha, min(beta,nx)) 
%%                                                                      
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
ui=zeros(size(K,1),1);
if K0==0 uh=ui;
    else uh=K0*x;
end;

if nargin==10 hn=max(alpha, min(beta,hn_fun(x)));
   elseif nargin==9 hn=max(alpha,hn_fun(x));
   else if nargin==8 hn=hn_fun(x); 
   else hn=hn_fun(x); end;
end;

if hn>1e-20 %numerical tolerance
hpx=expm(-log(hn)*Gd)*x; %homogenenous projection of x  
uh=uh+hn^(1+mu)*K*hpx;
ui=hn^(1+2*mu)*Ki*hpx;
end;

