function u=e_fho(h, z, y, A, C, f, L0, L, G0, nu1, nu2, alpha, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% The function u=e_fh0(h,z,y,A,C,f,L0,L,G0,nu1,nu2) computes (updates)                   
%% the estimate of the state of the linear MIMO system                             
%%                                                                      
%%     dx/dt = A*x+f,   y = C*x            (y and f are measured) 
%%
%% using implicit discretization of Fixed-time Homogeneous Observer (FHO)    
%%
%%                
%%               /(L0+|Cz-y|^(nu1-1)*d1(log(|Cz-y|))*L)*(Cz-y) if |Cz-y|<=1
%%  dz/dt=A*z+f+<         
%%               \(L0+|Cz-y|^(nu2-1)*d2(log(|Cz-y|))*L)*(Cz-y) if |Cz-y|>=1
%%                 
%%
%%
%% The function u=e_fh0(h,z,y,A,C,f,L0,L,G0,nu1,nu2,alpha,beta)                   
%% uses the saturation parameters alpha, beta to lower/upper bound
%% uses max(alpha,min(beta,|Cz-y|)) instead of |Cz-y| in computation of FHO
%%
%% Remark: the function is not optimized for practical realization in 
%% a digital control/estimation device. Please do contact the toolbox  
%% developer andrey.polyakov@inria.fr concerning practical use of FHO 
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if norm(C*z-y)<=1 Gd=eye(size(G0,2))+nu1*G0;nu=nu1;
else Gd=eye(size(G0,2))+nu2*G0;nu=nu2;
end;

if nargin>=13       
                     u=e_ho(h, z, y, A, C, f, L0, L, Gd, nu, alpha, beta);
  elseif nargin>=12  u=e_ho(h, z, y, A, C, f, L0, L, Gd, nu, alpha);
else                 u=e_ho(h, z, y, A, C, f, L0, L, Gd, nu);
end;


