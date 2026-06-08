function q=hnorm_newton(x,Gd,P,tol,Nmax,s)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function nx=hnorm_newtow(x, Gd, P) computes   (for a given vector x)                
%% a canonical d-homogeneous norm q induced by the weighted Euclidean norm           
%% nxp=sqrt(x'*P*x), namely:
%%                                                            
%% nx>0   such that  x'*d'(-log(nx))P*d(-log(nx))*x)=1   (if x is non-zero)
%%
%% ***           The coumpation is based on the Newton method           *** 
%%                                                            
%% The input parameters are                                   
%%     x - vector (p x 1) which norm has to be computed  
%%    Gd - anti-Hurwitx matrix (p x p)  being the generator of
%%          the dilation d(s)=expm(s*Gd) for any real s 
%%    P  - positive definite matrix (p x p) such that P*Gd+Gd'*P>0 
%% 
%%
%%
%% Remark: 
%% The function hnorm is not optimized for a real-time compuation of 
%% the homogeneous norm in practice. To design an fast algorithm for  
%% practical application of a homogeneous control please do contanct  
%%            andrey.polyakov@inria.fr (HPC_toolbox developer)
%%
%%
%%
%% The function q=hnorm_newton(x, Gd, P, tol) uses
%% additional parameter tol to define approximation precision
%% (by default tol=1e-6)
%%
%%
%%
%% The function q=hnorm(x, Gd, P, tol, Nmax) uses
%% additional parameter Nmax to define the maximum number of iterations of 
%% the numerical method (Nmax=20, by default)
%%
%%
%%
%% The function q=hnorm_newton(x, Gd, P, tol, Nmax, s) uses
%% the additional parameter s if the initial estimate nx~=exp(s) is known. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin<4 tol=1e-6; end;
if nargin<5 Nmax=20; end;
if nargin<6 s=0; end;

i=1;  
Q=P*Gd;Q=Q+Q'; 
y=expm(-Gd*s)*x; fs=y'*P*y; 
while (i<=Nmax) && (abs(log(fs))>tol) 
     s=s+fs*log(fs)/(y'*Q*y);
     y=expm(-Gd*s)*x;
     fs=y'*P*y;
     i=i+1;
end;
q=exp(s);
