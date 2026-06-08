function znew=si_ho(h, z, y, A, C, f, L0, L, Gd, nu, alpha, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function znew=si_ho(h,z,y,A,C,f,L0,L,Gd,nu) computes (updates)                   
%% the estimate of the state of the linear MIMO system                             
%%                                                                      
%%     dx/dt = A*x+f,   y = C*x            (f is a known input signal) 
%%
%% using implicit discretization of Homogeneous Observer (HO)     
%%                                                                      
%%    dz/dt=A*z+f+(L0+|Cz-y|^(nu-1)*d(log(|Cz-y|))*L)*(Cz-y)  
%%
%% %% The input parameters:  
%%     h  - sampling period
%%     z  - current state estimate 
%%     y  -  measured output
%%     A  - system matrix (n x n)                                       
%%     C  - output matrix (k x n) 
%%     f  - exogeneous signal
%%     L0 - homogenization gain
%%     Gd - generator of dilation
%%     nu - homogeneity degree      
%%
%% The function  znew=si_ho(h, z, y, A, C, f, L0, L, Gd, nu,alpha,beta)                    
%% uses max(alpha, min(beta,|Cz-y|)) instead of |Cz-y| in computation of HO
%% 
%% Remark: the function is not optimized for practical realization in 
%% a digital control/estimation device. Please do contact the toolbox  
%% developer andrey.polyakov@inria.fr concerning practical use of HO 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
nCe=norm(C*z-y);tol=1e-20;
if nargin>=12 nCe=min(nCe,beta); end;
if nargin>=11 nCe=max(nCe,alpha); end;
nCe=max(tol,nCe);
n=size(A,2);I_n=eye(n);
S=L0+expm(log(nCe)*(nu*I_n+Gd-eye(n)))*L;
znew=inv(I_n-h*(A+S*C))*(z+h*f-h*S*y);