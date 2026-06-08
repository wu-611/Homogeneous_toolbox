function u=e_hsmc(x, C, K0, K, Gd, hn_fun, alpha, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function  u=e_hsmc(x, C, K0, K, Gd, hn_fun) explicitly discretizes                    
%% Homogeneous (unit) Sliding Mode Controller (HSMC)     
%%                                                                      
%%     u = K0*x + K*d(-ln(hn_fun(C*x)))*C*x 
%%
%%  where x - system state  and   hn_fun(C*x) - homogenenous norm of C*x
%%
%%
%% The function  u=e_hsmc(x, C, K0, K, Gd, hn_fun, alpha, beta)                    
%% uses the saturation parameters alpha and beta to lower/upper bound
%% the homogenenous norm nx     
%%                                                                      
%%     ncx = max(alpha, min(beta,hn_fun(C*x)) 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if K0==0 u=zeros(size(K,1),1);
    else u=K0*x;
end;

if nargin==8 u=u+e_hpc(C*x,0,K,Gd,-1,hn_fun,alpha,beta);
   elseif nargin==7 u=u+e_hpc(C*x,0,K,Gd,-1,hn_fun,alpha);
else u=u+e_hpc(C*x,0,K,Gd,-1,hn_fun);
end;


