function u=si_hpc(h, x, A, B, K0, K, Gd, mu, hn_fun, alpha, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function  u=e_hpc(h, x, A, B, K0, K, Gd, mu, hn_fun) descretizes 
%% semi-implicitly Homogeneous Proportional Controller (HPC)     
%%                                                                      
%%     u = K0*x + nx^(1+mu)*K*d(-ln(nx))*x 
%%
%%  where x - system state  and   nx=hn_fun(x) - homogenenous norm of x
%%
%%
%% The function  u=si_hpc(h, A, B, x, K0, K, Gd, P, mu, alpha, beta)                    
%% uses the saturation parameters alpha and beta to lower and upper bounds
%% of the homogenenous norm nx (i.e., alpha <= nx <= beta) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

u=zeros(size(K,1),1);

warning('off','all');
if nargin==11 hn=max(alpha,min(beta,hn_fun(x)));
   elseif nargin==10 hn=max(alpha,hn_fun(x));
   else hn=hn_fun(x); 
end;

n=size(A,2);I_n=eye(n);
Kh=K0;
if hn>1e-16 %numerical tolerance
Kh=Kh+hn^(1+mu)*K*expm(-log(hn)*Gd);
if norm(Kh)<1e16
u=Kh*inv(I_n-h*(A+B*Kh))*x;end;
end;
warning('on','all');



