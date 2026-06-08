function [uh ui]=e_hsmci(x, C, K0, K, Ki, Gd,  hn_fun, alpha, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function [uh ui]=e_hsmci(x, C, K0, K, Ki, Gd, hn_fun) computes                  
%% Homogeneous  Sliding Mode Control with Integral action (HSMCI)       
%%                                                                      
%%     u = K0*x + sqrt(ncx)*K*d(-log(ncx))*C*x + v   
%%                                          
%%
%%                    
%%     dv/dt= Ki*d(-log(ncx))*C*x                                            
%%
%%  where x - system state, ncx=hn_fun(C*x) - homogenenous norm of C*x 
%%
%% The function  u=e_hsmci(x, C, K0, K, Ki, Gd, P, hn_fun, alpha, beta)                    
%% uses the saturation parameters alpha and beta to lower/upper bound
%% the homogenenous norm of C*x  
%%                                                                      
%%     ncx = max(alpha, min(beta,hn_fun(C*x))) 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin==10 [uh ui]=e_hpic(C*x,0,K,Ki,Gd,-0.5,hn_fun,alpha,beta);
   elseif nargin==9 [uh ui]=e_hpic(C*x,0,K,Ki,Gd,-0.5,hn_fun,alpha);
else if nargin==8 [uh ui]=e_hpic(C*x,0,K,Ki,Gd,-0.5,hn_fun);
      else [uh ui]=e_hpic(C*x,0,K,Ki,Gd,-0.5); end;
end;
uh=K0*x+uh;



